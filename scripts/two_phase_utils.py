from __future__ import annotations

import csv
import math
import random
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import models, transforms

WEAPON_CLASSES = {
    "handgun",
    "short_rifle",
    "short-rifle",
    "rifle",
    "gun",
    "pistol",
    "revolver",
    "firearm",
    "weapon",
}

EXCLUDED_CLASSES = {"knife"}

DEFAULT_TWO_PHASE_CONFIG: dict[str, Any] = {
    "paths": {
        "splits_dir": "data/splits",
        "raw_images_dir": "data/raw/images",
        "raw_annotations_dir": "data/raw/annotations",
        "two_phase_root": "data/interim/two_phase",
        "carry_runs_dir": "runs/two_phase/carry_classifier",
        "predictions_dir": "runs/two_phase/predictions",
        "evaluation_dir": "runs/two_phase/evaluation",
    },
    "models": {
        "person_detector": "yolo11n.pt",
        "weapon_detector": "runs/two_phase/weapon_crop_detector/weights/best.pt",
    },
    "thresholds": {
        "person_conf": 0.10,
        "person_iou": 0.45,
        "weapon_conf": 0.25,
        "weapon_iou": 0.45,
        "evaluation_iou": 0.50,
        "image_level_nms_iou": 0.50,
    },
    "dataset": {
        "crop_padding": 0.20,
        "crop_padding_x": 0.35,
        "crop_padding_y": 0.25,
        "min_crop_side": 256,
        "max_negatives_per_image": 2,
        "classifier_image_size": 224,
        "match_ioa_threshold": 0.60,
    },
    "training": {
        "batch_size": 32,
        "epochs": 12,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_workers": 0,
        "seed": 42,
        "threshold_recall_floor": 0.70,
    },
    "inference": {
        "default_device": "cpu",
        "person_imgsz": 960,
        "person_max_det": 50,
        "weapon_crop_imgsz": 640,
        "enable_hold_gate": False,
    },
}


class CarryClassifierNet(nn.Module):
    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)


# ---------------------------------------------------------------------------
# MobileNetV3-Small carry classifier  (replaces the custom 3-layer CNN)
# ---------------------------------------------------------------------------

class MobileNetV3CarryClassifier(nn.Module):
    """
    MobileNetV3-Small backbone (ImageNet pre-trained) with a binary head.

    Rationale (from paper analysis):
      - Weapons in 224x224 crops appear as 10-17px objects (87% of pairs < 32x32).
      - A 3-layer CNN trained from scratch cannot learn reliable features at that
        scale from ~200 positive examples.
      - ImageNet pre-trained features (edges, textures, elongated shapes) transfer
        well to weapon silhouettes even at tiny scale.

    Training strategy (see train_carry_classifier.py):
      1. Freeze backbone for the first `backbone_freeze_epochs`.
      2. Train only the head so it adapts to the binary task quickly.
      3. Unfreeze backbone and continue at a lower LR (backbone_lr_factor * lr).
    """

    def __init__(self, image_size: int = 224, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        # Replace the built-in classifier with a simple binary head
        in_features = backbone.classifier[0].in_features  # 576
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(128, 1),
        )
        self.backbone = backbone
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers except the classifier head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def head_parameters(self):
        return self.backbone.classifier.parameters()

    def backbone_parameters(self):
        return self.backbone.features.parameters()


def build_carry_classifier(
    backbone: str = "mobilenet_v3_small",
    image_size: int = 224,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function. Returns the appropriate carry classifier.

    Args:
        backbone: "mobilenet_v3_small" (recommended) or "custom_cnn" (legacy).
        image_size: Input size. Only relevant for custom_cnn.
        pretrained: Whether to load ImageNet weights (MobileNetV3 only).
    """
    if backbone == "mobilenet_v3_small":
        return MobileNetV3CarryClassifier(image_size=image_size, pretrained=pretrained)
    if backbone == "custom_cnn":
        return CarryClassifierNet(image_size=image_size)
    raise ValueError(f"Unknown classifier backbone: {backbone!r}. Choose 'mobilenet_v3_small' or 'custom_cnn'.")


# ---------------------------------------------------------------------------
# SAHI — Slicing Aided Hyper Inference (manual implementation, no extra deps)
# ---------------------------------------------------------------------------

def _sahi_slice_coords(
    image_w: int,
    image_h: int,
    tile_size: int,
    overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """
    Generate (xmin, ymin, xmax, ymax) pixel coordinates for overlapping tiles.

    The last tile in each row/column is snapped to the image edge to avoid
    border strips.
    """
    stride = max(1, int(tile_size * (1.0 - overlap_ratio)))
    slices: list[tuple[int, int, int, int]] = []

    x_starts = list(range(0, max(1, image_w - tile_size + 1), stride))
    if not x_starts or x_starts[-1] + tile_size < image_w:
        x_starts.append(max(0, image_w - tile_size))

    y_starts = list(range(0, max(1, image_h - tile_size + 1), stride))
    if not y_starts or y_starts[-1] + tile_size < image_h:
        y_starts.append(max(0, image_h - tile_size))

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, image_w)
            y1 = min(y0 + tile_size, image_h)
            slices.append((x0, y0, x1, y1))

    return slices


def sahi_predict_on_crop(
    crop_image: Image.Image,
    weapon_model: Any,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    infer_imgsz: int,
    tile_size: int = 320,
    overlap_ratio: float = 0.30,
    min_crop_side: int = 200,
    nms_iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """
    Run YOLO weapon detection on a person crop using SAHI-style tiling.

    Rationale:
      - Weapons appear as ~10-30px in 600x700px crops.  At 640px inference
        they land in YOLO's "tiny" bucket where AP is ~38% below medium.
      - Tiling to 320px tiles and running each at `infer_imgsz` effectively
        doubles/triples the weapon's pixel footprint inside each tile.
      - Detections from all tiles are mapped back to crop coordinates and
        deduplicated with NMS.

    Returns:
        List of detection dicts with keys xmin, ymin, xmax, ymax, confidence.
        Coordinates are in the crop's pixel space (origin = top-left of crop).
    """
    crop_w, crop_h = crop_image.size

    # If the crop is small enough, just run the detector directly
    if crop_w < min_crop_side and crop_h < min_crop_side:
        result = weapon_model.predict(
            source=crop_image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=device,
            imgsz=infer_imgsz,
        )[0]
        return extract_yolo_boxes(result)

    tile_coords = _sahi_slice_coords(crop_w, crop_h, tile_size, overlap_ratio)
    all_boxes: list[dict[str, Any]] = []
    all_scores: list[float] = []

    for tx0, ty0, tx1, ty1 in tile_coords:
        tile = crop_image.crop((tx0, ty0, tx1, ty1))
        result = weapon_model.predict(
            source=tile,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=device,
            imgsz=infer_imgsz,
        )[0]
        for det in extract_yolo_boxes(result):
            # Map tile-local coordinates back to crop space
            all_boxes.append({
                "xmin": float(det["xmin"]) + tx0,
                "ymin": float(det["ymin"]) + ty0,
                "xmax": float(det["xmax"]) + tx0,
                "ymax": float(det["ymax"]) + ty0,
                "confidence": float(det["confidence"]),
                "class_id": int(det.get("class_id", 0)),
            })
            all_scores.append(float(det["confidence"]))

    if not all_boxes:
        return []

    keep = nms_indices(all_boxes, all_scores, iou_threshold=nms_iou_threshold)
    return [all_boxes[i] for i in keep]


# ---------------------------------------------------------------------------
# Zoom-crop helper for carry classifier training
# ---------------------------------------------------------------------------

def zoom_lower_fraction(
    crop_image: Image.Image,
    lower_fraction: float,
) -> Image.Image:
    """
    Return the lower `lower_fraction` of a crop (PIL Image).

    Rationale:
      - Weapons are typically carried at waist/hand level, i.e., the lower
        ~55% of the person bounding box.
      - Zooming into this region increases the weapon's pixel footprint in
        the 224x224 classifier input from ~15px to ~27px (~80% increase).
      - The classifier learns weapon-relevant features (hand+object shape)
        rather than full-body silhouette.

    Args:
        crop_image: PIL Image of the full padded person crop.
        lower_fraction: Fraction of height to keep, from the bottom.
                        0.55 means keep the lower 55%.
    """
    w, h = crop_image.size
    y_start = max(0, int(h * (1.0 - lower_fraction)))
    return crop_image.crop((0, y_start, w, h))


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_dir_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
        return
    for child in sorted(path.iterdir(), reverse=True):
        if child.is_dir():
            remove_dir_if_exists(child)
        else:
            child.unlink()
    path.rmdir()


def merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_two_phase_config(config_path: Path | None = None) -> dict[str, Any]:
    root = project_root()
    path = config_path or (root / "configs" / "two_phase.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Missing two-phase config: {path}")
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return merge_dicts(DEFAULT_TWO_PHASE_CONFIG, loaded)


def resolve_path(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def path_for_csv(path: Path) -> str:
    root = project_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_")


def is_weapon_class(class_name: str) -> bool:
    normalized = normalize_class_name(class_name)
    return normalized in WEAPON_CLASSES


def parse_voc_xml(annotation_path: Path) -> tuple[int | None, int | None, list[dict[str, Any]]]:
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    size = root.find("size")
    width = None
    height = None
    if size is not None:
        width_text = size.findtext("width")
        height_text = size.findtext("height")
        if width_text is not None and height_text is not None:
            width = int(float(width_text))
            height = int(float(height_text))

    objects = []
    for obj in root.findall("object"):
        class_name = normalize_class_name(obj.findtext("name") or "")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            continue

        objects.append(
            {
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return width, height, objects


def filter_weapon_boxes(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [obj for obj in objects if is_weapon_class(obj["class_name"])]


def center_of_box(box: dict[str, Any]) -> tuple[float, float]:
    return ((float(box["xmin"]) + float(box["xmax"])) / 2.0, (float(box["ymin"]) + float(box["ymax"])) / 2.0)


def point_in_box(x: float, y: float, box: dict[str, Any]) -> bool:
    return float(box["xmin"]) <= x <= float(box["xmax"]) and float(box["ymin"]) <= y <= float(box["ymax"])


def clamp_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    xmin_i = max(0, min(int(round(xmin)), width - 1))
    ymin_i = max(0, min(int(round(ymin)), height - 1))
    xmax_i = max(xmin_i + 1, min(int(round(xmax)), width))
    ymax_i = max(ymin_i + 1, min(int(round(ymax)), height))
    return xmin_i, ymin_i, xmax_i, ymax_i


def expand_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
    padding_fraction: float | None = None,
    padding_x_fraction: float | None = None,
    padding_y_fraction: float | None = None,
    min_side: int | None = None,
) -> tuple[int, int, int, int]:
    box_width = xmax - xmin
    box_height = ymax - ymin
    if padding_x_fraction is None:
        padding_x_fraction = padding_fraction if padding_fraction is not None else 0.0
    if padding_y_fraction is None:
        padding_y_fraction = padding_fraction if padding_fraction is not None else 0.0

    pad_x = box_width * float(padding_x_fraction)
    pad_y = box_height * float(padding_y_fraction)

    expanded_xmin = xmin - pad_x
    expanded_ymin = ymin - pad_y
    expanded_xmax = xmax + pad_x
    expanded_ymax = ymax + pad_y

    if min_side is not None and min_side > 0:
        current_width = expanded_xmax - expanded_xmin
        current_height = expanded_ymax - expanded_ymin
        if current_width < min_side:
            extra_x = (float(min_side) - current_width) / 2.0
            expanded_xmin -= extra_x
            expanded_xmax += extra_x
        if current_height < min_side:
            extra_y = (float(min_side) - current_height) / 2.0
            expanded_ymin -= extra_y
            expanded_ymax += extra_y

    return clamp_box(expanded_xmin, expanded_ymin, expanded_xmax, expanded_ymax, width=width, height=height)


def box_area(box: dict[str, Any]) -> float:
    width = max(0.0, float(box["xmax"]) - float(box["xmin"]))
    height = max(0.0, float(box["ymax"]) - float(box["ymin"]))
    return width * height


def box_iou(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    inter_xmin = max(float(box_a["xmin"]), float(box_b["xmin"]))
    inter_ymin = max(float(box_a["ymin"]), float(box_b["ymin"]))
    inter_xmax = min(float(box_a["xmax"]), float(box_b["xmax"]))
    inter_ymax = min(float(box_a["ymax"]), float(box_b["ymax"]))

    inter_width = max(0.0, inter_xmax - inter_xmin)
    inter_height = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0

    union = box_area(box_a) + box_area(box_b) - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def box_ioa(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    inter_xmin = max(float(box_a["xmin"]), float(box_b["xmin"]))
    inter_ymin = max(float(box_a["ymin"]), float(box_b["ymin"]))
    inter_xmax = min(float(box_a["xmax"]), float(box_b["xmax"]))
    inter_ymax = min(float(box_a["ymax"]), float(box_b["ymax"]))

    inter_width = max(0.0, inter_xmax - inter_xmin)
    inter_height = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0

    area_a = box_area(box_a)
    if area_a <= 0:
        return 0.0
    return inter_area / area_a


def crop_matches_weapon(
    crop_box: dict[str, Any],
    weapon_box: dict[str, Any],
    match_ioa_threshold: float,
) -> tuple[bool, bool, float]:
    center_x, center_y = center_of_box(weapon_box)
    center_match = point_in_box(center_x, center_y, crop_box)
    ioa = box_ioa(weapon_box, crop_box)
    return center_match or ioa >= match_ioa_threshold, center_match, ioa


def intersect_boxes(box_a: dict[str, Any], box_b: dict[str, Any]) -> dict[str, float] | None:
    inter_xmin = max(float(box_a["xmin"]), float(box_b["xmin"]))
    inter_ymin = max(float(box_a["ymin"]), float(box_b["ymin"]))
    inter_xmax = min(float(box_a["xmax"]), float(box_b["xmax"]))
    inter_ymax = min(float(box_a["ymax"]), float(box_b["ymax"]))
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return None
    return {
        "xmin": inter_xmin,
        "ymin": inter_ymin,
        "xmax": inter_xmax,
        "ymax": inter_ymax,
    }


def project_box_into_crop(
    box: dict[str, Any],
    crop_box: dict[str, Any],
) -> dict[str, float] | None:
    clipped = intersect_boxes(box, crop_box)
    if clipped is None:
        return None
    return {
        "xmin": float(clipped["xmin"]) - float(crop_box["xmin"]),
        "ymin": float(clipped["ymin"]) - float(crop_box["ymin"]),
        "xmax": float(clipped["xmax"]) - float(crop_box["xmin"]),
        "ymax": float(clipped["ymax"]) - float(crop_box["ymin"]),
    }


def yolo_label_line_for_box(
    box: dict[str, Any],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> str:
    x_center = ((float(box["xmin"]) + float(box["xmax"])) / 2.0) / float(image_width)
    y_center = ((float(box["ymin"]) + float(box["ymax"])) / 2.0) / float(image_height)
    box_width = (float(box["xmax"]) - float(box["xmin"])) / float(image_width)
    box_height = (float(box["ymax"]) - float(box["ymin"])) / float(image_height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def nms_indices(boxes: list[dict[str, Any]], scores: list[float], iou_threshold: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining = []
        for idx in order:
            if box_iou(boxes[current], boxes[idx]) <= iou_threshold:
                remaining.append(idx)
        order = remaining
    return keep


def load_split_manifest(project_root_path: Path, split_name: str) -> pd.DataFrame:
    manifest_path = project_root_path / "data" / "splits" / f"{split_name}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    return pd.read_csv(manifest_path)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_torch_device(device: str) -> str:
    if device.isdigit():
        return f"cuda:{device}"
    return device


def build_classifier_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def extract_yolo_boxes(result: Any, allowed_class_ids: set[int] | None = None) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else [1.0] * len(xyxy)
    classes = result.boxes.cls.cpu().tolist() if result.boxes.cls is not None else [0.0] * len(xyxy)
    boxes = []
    for coords, conf, class_id_raw in zip(xyxy, confs, classes):
        class_id = int(class_id_raw)
        if allowed_class_ids is not None and class_id not in allowed_class_ids:
            continue
        boxes.append(
            {
                "xmin": float(coords[0]),
                "ymin": float(coords[1]),
                "xmax": float(coords[2]),
                "ymax": float(coords[3]),
                "confidence": float(conf),
                "class_id": class_id,
            }
        )
    return boxes


def save_rows_to_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def threshold_range(
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 2))
        current += step
    return values
