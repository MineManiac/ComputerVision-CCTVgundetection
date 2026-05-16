from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

from ultralytics import YOLO

from two_phase_utils import (
    crop_matches_weapon,
    ensure_dir,
    expand_box,
    extract_yolo_boxes,
    filter_weapon_boxes,
    load_image,
    load_split_manifest,
    load_two_phase_config,
    parse_voc_xml,
    path_for_csv,
    project_box_into_crop,
    project_root,
    remove_dir_if_exists,
    resolve_path,
)


BOX_COLORS = {
    "weapon": (255, 50, 50),
    "person": (0, 210, 255),
    "crop": (255, 220, 0),
}


def compact_file_prefix(image_stem: str, image_index: int, person_index: int | None = None, suffix: str = "") -> str:
    digest = hashlib.sha1(image_stem.encode("utf-8")).hexdigest()[:8]
    camera = image_stem.split("-", 1)[0]
    frame_token = image_stem.rsplit("_frame_", 1)[-1] if "_frame_" in image_stem else image_stem[-12:]
    frame_token = "".join(char if char.isalnum() else "_" for char in frame_token)[:16]
    parts = [f"{image_index:03d}", camera, f"frame{frame_token}", digest]
    if person_index is not None:
        parts.append(f"p{person_index:02d}")
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first stage of the two-phase pipeline on a small positive subset "
            "and save visual diagnostics for original frames, person crops, and resized crops."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/two_phase.yaml.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to sample from.")
    parser.add_argument("--num-images", type=int, default=50, help="Number of GT-positive images to inspect.")
    parser.add_argument(
        "--sample-mode",
        default="first",
        choices=["first", "random"],
        help="Use the first positive rows or a seeded random subset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used when --sample-mode random is selected.")
    parser.add_argument("--person-model", default=None, help="Override person detector model or weights path.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--person-conf", type=float, default=None, help="Override person confidence threshold.")
    parser.add_argument("--person-iou", type=float, default=None, help="Override person NMS IoU threshold.")
    parser.add_argument("--person-imgsz", type=int, default=None, help="Override person detector image size.")
    parser.add_argument("--person-max-det", type=int, default=None, help="Override max person detections per image.")
    parser.add_argument(
        "--resize-size",
        type=int,
        default=None,
        help="Output crop size after resize. Defaults to dataset.classifier_image_size from config.",
    )
    parser.add_argument(
        "--save-all-positive-crops",
        action="store_true",
        help="Save every matched person crop instead of only the highest-confidence matched crop per image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/debug/two_phase_crop_subset"),
        help="Directory where diagnostic images and manifest files will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing new diagnostics.",
    )
    return parser.parse_args()


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int) == 1
    return series.astype(str).str.lower().isin({"1", "true", "yes", "weapon"})


def safe_output_dir(root: Path, output_dir: Path) -> Path:
    resolved = resolve_path(root, output_dir).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Output directory must stay inside the project root: {resolved}") from exc
    return resolved


def select_positive_rows(split_df: pd.DataFrame, num_images: int, sample_mode: str, seed: int) -> pd.DataFrame:
    positives = split_df[to_bool_series(split_df["has_weapon"])].copy()
    if positives.empty:
        raise ValueError("No positive images found in the selected split.")
    if sample_mode == "random":
        return positives.sample(n=min(num_images, len(positives)), random_state=seed).reset_index(drop=True)
    return positives.head(num_images).reset_index(drop=True)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    coords = [float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])]
    draw.rectangle(coords, outline=color, width=width)
    text_xy = (coords[0] + 3, max(0, coords[1] - 14))
    draw.text(text_xy, label, fill=color, font=font)


def annotate_original(
    image: Image.Image,
    weapon_boxes: list[dict[str, Any]],
    person_box: dict[str, Any] | None,
    crop_box: dict[str, Any] | None,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    line_width = max(3, round(min(annotated.size) / 300))

    for idx, weapon_box in enumerate(weapon_boxes):
        label = f"GT weapon {idx}"
        draw_box(draw, weapon_box, BOX_COLORS["weapon"], label, line_width, font)

    if person_box is not None:
        confidence = person_box.get("confidence")
        suffix = f" {float(confidence):.2f}" if confidence is not None else ""
        draw_box(draw, person_box, BOX_COLORS["person"], f"person{suffix}", line_width, font)

    if crop_box is not None:
        draw_box(draw, crop_box, BOX_COLORS["crop"], "padded crop", line_width, font)

    return annotated


def resized_crop_with_gt(
    raw_crop: Image.Image,
    weapon_boxes: list[dict[str, Any]],
    crop_box: dict[str, Any],
    resize_size: int,
) -> Image.Image:
    resized = raw_crop.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
    annotated = resized.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    crop_width = max(1.0, float(crop_box["xmax"]) - float(crop_box["xmin"]))
    crop_height = max(1.0, float(crop_box["ymax"]) - float(crop_box["ymin"]))
    scale_x = resize_size / crop_width
    scale_y = resize_size / crop_height

    for idx, weapon_box in enumerate(weapon_boxes):
        projected = project_box_into_crop(weapon_box, crop_box)
        if projected is None:
            continue
        resized_box = {
            "xmin": float(projected["xmin"]) * scale_x,
            "ymin": float(projected["ymin"]) * scale_y,
            "xmax": float(projected["xmax"]) * scale_x,
            "ymax": float(projected["ymax"]) * scale_y,
        }
        draw_box(draw, resized_box, BOX_COLORS["weapon"], f"GT {idx}", 2, font)

    return annotated


def make_panel(
    original_annotated: Image.Image,
    resized_crop_annotated: Image.Image | None,
    title: str,
    status: str,
    panel_original_width: int = 960,
    crop_display_size: int = 448,
) -> Image.Image:
    font = ImageFont.load_default()
    original_ratio = panel_original_width / original_annotated.width
    original_display_height = max(1, round(original_annotated.height * original_ratio))
    original_display = original_annotated.resize(
        (panel_original_width, original_display_height),
        Image.Resampling.BILINEAR,
    )

    right_width = crop_display_size
    panel_width = panel_original_width + right_width + 36
    panel_height = max(original_display_height, crop_display_size + 60) + 70
    panel = Image.new("RGB", (panel_width, panel_height), color=(18, 18, 18))
    draw = ImageDraw.Draw(panel)

    draw.text((16, 12), title, fill=(245, 245, 245), font=font)
    draw.text((16, 32), "red=GT weapon | cyan=person bbox | yellow=padded crop", fill=(210, 210, 210), font=font)
    draw.text((16, 50), status, fill=(210, 210, 210), font=font)

    panel.paste(original_display, (16, 70))
    crop_x = panel_original_width + 32
    draw.text((crop_x, 50), "resized crop", fill=(245, 245, 245), font=font)

    if resized_crop_annotated is None:
        crop_area = Image.new("RGB", (crop_display_size, crop_display_size), color=(35, 35, 35))
        crop_draw = ImageDraw.Draw(crop_area)
        crop_draw.text((18, 18), "No matched person crop", fill=(235, 235, 235), font=font)
        crop_display = crop_area
    else:
        crop_display = resized_crop_annotated.resize(
            (crop_display_size, crop_display_size),
            Image.Resampling.NEAREST,
        )

    panel.paste(crop_display, (crop_x, 70))
    return panel


def matched_weapon_indices(
    crop_box: dict[str, Any],
    weapon_boxes: list[dict[str, Any]],
    match_ioa_threshold: float,
) -> list[int]:
    matches = []
    for weapon_idx, weapon_box in enumerate(weapon_boxes):
        is_match, _, _ = crop_matches_weapon(crop_box, weapon_box, match_ioa_threshold)
        if is_match:
            matches.append(weapon_idx)
    return matches


def write_markdown_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    split: str,
    num_images_requested: int,
    resize_size: int,
) -> None:
    total_images = len({row["image_stem"] for row in rows})
    matched_images = len({row["image_stem"] for row in rows if row["status"] == "matched"})
    miss_images = len({row["image_stem"] for row in rows if row["status"] == "no_matched_crop"})
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Two-Phase Crop Subset Diagnostic",
                "",
                f"- Split: `{split}`",
                f"- Requested positive images: `{num_images_requested}`",
                f"- Positive images inspected: `{total_images}`",
                f"- Images with matched person crop: `{matched_images}`",
                f"- Images without matched person crop: `{miss_images}`",
                f"- Resized crop size: `{resize_size}x{resize_size}`",
                "",
                "Color legend:",
                "",
                "- red: ground-truth weapon box",
                "- cyan: detected person box selected by the pipeline",
                "- yellow: padded crop box before resize",
                "",
                "Files:",
                "",
                "- `panels/`: side-by-side original annotation and resized crop",
                "- `originals/`: original frames with GT/person/crop boxes",
                "- `raw_crops/`: crop before resize",
                "- `resized_crops/`: exact resized crop without overlays",
                "- `resized_crops_annotated/`: resized crop with projected GT weapon boxes",
                "- `diagnostic_manifest.csv`: per-crop metadata",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_two_phase_config(args.config)
    output_dir = safe_output_dir(root, args.output_dir)

    if args.overwrite and output_dir.exists():
        remove_dir_if_exists(output_dir)

    panels_dir = output_dir / "panels"
    originals_dir = output_dir / "originals"
    raw_crops_dir = output_dir / "raw_crops"
    resized_crops_dir = output_dir / "resized_crops"
    resized_annotated_dir = output_dir / "resized_crops_annotated"
    for directory in [panels_dir, originals_dir, raw_crops_dir, resized_crops_dir, resized_annotated_dir]:
        ensure_dir(directory)

    dataset_config = config["dataset"]
    threshold_config = config["thresholds"]
    inference_config = config["inference"]

    padding_x_fraction = float(dataset_config.get("crop_padding_x", dataset_config.get("crop_padding", 0.20)))
    padding_y_fraction = float(dataset_config.get("crop_padding_y", dataset_config.get("crop_padding", 0.20)))
    min_crop_side = int(dataset_config.get("min_crop_side", 1))
    match_ioa_threshold = float(dataset_config.get("match_ioa_threshold", 0.60))
    resize_size = int(args.resize_size or dataset_config.get("classifier_image_size", 224))
    person_conf = float(args.person_conf if args.person_conf is not None else threshold_config["person_conf"])
    person_iou = float(args.person_iou if args.person_iou is not None else threshold_config["person_iou"])
    person_imgsz = int(args.person_imgsz or inference_config.get("person_imgsz", 960))
    person_max_det = int(args.person_max_det or inference_config.get("person_max_det", 50))
    device = args.device or inference_config["default_device"]

    split_df = load_split_manifest(root, args.split)
    selected_df = select_positive_rows(split_df, args.num_images, args.sample_mode, args.seed)

    person_model_name = args.person_model or config["models"]["person_detector"]
    person_model = YOLO(person_model_name)

    rows: list[dict[str, Any]] = []
    for subset_idx, row in selected_df.iterrows():
        image_path = root / Path(row["image_path"])
        annotation_path = root / Path(row["annotation_path"])
        width, height, objects = parse_voc_xml(annotation_path)
        if width is None or height is None:
            raise ValueError(f"Missing size information in XML: {annotation_path}")

        image = load_image(image_path)
        weapon_boxes = filter_weapon_boxes(objects)
        result = person_model.predict(
            source=str(image_path),
            conf=person_conf,
            iou=person_iou,
            classes=[0],
            verbose=False,
            device=device,
            imgsz=person_imgsz,
            max_det=person_max_det,
        )[0]
        detections = extract_yolo_boxes(result, allowed_class_ids={0})

        records = []
        for det_idx, det_box in enumerate(detections):
            crop_xmin, crop_ymin, crop_xmax, crop_ymax = expand_box(
                det_box["xmin"],
                det_box["ymin"],
                det_box["xmax"],
                det_box["ymax"],
                width=width,
                height=height,
                padding_x_fraction=padding_x_fraction,
                padding_y_fraction=padding_y_fraction,
                min_side=min_crop_side,
            )
            crop_box = {
                "xmin": float(crop_xmin),
                "ymin": float(crop_ymin),
                "xmax": float(crop_xmax),
                "ymax": float(crop_ymax),
            }
            matches = matched_weapon_indices(crop_box, weapon_boxes, match_ioa_threshold)
            if matches:
                records.append(
                    {
                        "det_idx": det_idx,
                        "det_box": det_box,
                        "crop_box": crop_box,
                        "crop_coords": (crop_xmin, crop_ymin, crop_xmax, crop_ymax),
                        "matched_weapon_indices": matches,
                    }
                )

        records.sort(key=lambda item: float(item["det_box"].get("confidence", 0.0)), reverse=True)
        if not args.save_all_positive_crops:
            records = records[:1]

        if not records:
            stem = str(row["image_stem"])
            prefix = compact_file_prefix(stem, subset_idx + 1, suffix="miss")
            original_annotated = annotate_original(image, weapon_boxes, person_box=None, crop_box=None)
            original_path = originals_dir / f"{prefix}_original.jpg"
            panel_path = panels_dir / f"{prefix}_panel.jpg"
            original_annotated.save(original_path, quality=95)
            panel = make_panel(
                original_annotated,
                resized_crop_annotated=None,
                title=f"{subset_idx + 1:03d} {stem}",
                status=f"GT-positive image, but no matched person crop. person_detections={len(detections)}",
            )
            panel.save(panel_path, quality=95)
            rows.append(
                {
                    "split": args.split,
                    "image_index": subset_idx + 1,
                    "image_stem": stem,
                    "image_path": path_for_csv(image_path),
                    "annotation_path": path_for_csv(annotation_path),
                    "status": "no_matched_crop",
                    "num_gt_weapon_boxes": len(weapon_boxes),
                    "num_person_detections": len(detections),
                    "person_index": "",
                    "person_confidence": "",
                    "person_xmin": "",
                    "person_ymin": "",
                    "person_xmax": "",
                    "person_ymax": "",
                    "crop_xmin": "",
                    "crop_ymin": "",
                    "crop_xmax": "",
                    "crop_ymax": "",
                    "matched_weapon_indices": "",
                    "original_annotated_path": path_for_csv(original_path),
                    "raw_crop_path": "",
                    "resized_crop_path": "",
                    "resized_crop_annotated_path": "",
                    "panel_path": path_for_csv(panel_path),
                }
            )
            continue

        for record_idx, record in enumerate(records):
            stem = str(row["image_stem"])
            det_idx = int(record["det_idx"])
            det_box = record["det_box"]
            crop_box = record["crop_box"]
            crop_xmin, crop_ymin, crop_xmax, crop_ymax = record["crop_coords"]
            prefix = compact_file_prefix(stem, subset_idx + 1, person_index=det_idx)

            raw_crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
            resized_crop = raw_crop.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
            resized_annotated = resized_crop_with_gt(raw_crop, weapon_boxes, crop_box, resize_size)
            original_annotated = annotate_original(image, weapon_boxes, det_box, crop_box)
            status = (
                f"matched crop {record_idx + 1}/{len(records)} | "
                f"person_conf={float(det_box.get('confidence', 0.0)):.3f} | "
                f"matched_gt={','.join(str(idx) for idx in record['matched_weapon_indices'])}"
            )
            panel = make_panel(
                original_annotated,
                resized_annotated,
                title=f"{subset_idx + 1:03d} {stem}",
                status=status,
            )

            original_path = originals_dir / f"{prefix}_original.jpg"
            raw_crop_path = raw_crops_dir / f"{prefix}_raw_crop.jpg"
            resized_crop_path = resized_crops_dir / f"{prefix}_resized_{resize_size}.jpg"
            resized_annotated_path = resized_annotated_dir / f"{prefix}_resized_{resize_size}_gt.jpg"
            panel_path = panels_dir / f"{prefix}_panel.jpg"

            original_annotated.save(original_path, quality=95)
            raw_crop.save(raw_crop_path, quality=95)
            resized_crop.save(resized_crop_path, quality=95)
            resized_annotated.save(resized_annotated_path, quality=95)
            panel.save(panel_path, quality=95)

            rows.append(
                {
                    "split": args.split,
                    "image_index": subset_idx + 1,
                    "image_stem": stem,
                    "image_path": path_for_csv(image_path),
                    "annotation_path": path_for_csv(annotation_path),
                    "status": "matched",
                    "num_gt_weapon_boxes": len(weapon_boxes),
                    "num_person_detections": len(detections),
                    "person_index": det_idx,
                    "person_confidence": round(float(det_box.get("confidence", 0.0)), 6),
                    "person_xmin": round(float(det_box["xmin"]), 3),
                    "person_ymin": round(float(det_box["ymin"]), 3),
                    "person_xmax": round(float(det_box["xmax"]), 3),
                    "person_ymax": round(float(det_box["ymax"]), 3),
                    "crop_xmin": crop_xmin,
                    "crop_ymin": crop_ymin,
                    "crop_xmax": crop_xmax,
                    "crop_ymax": crop_ymax,
                    "matched_weapon_indices": ";".join(str(idx) for idx in record["matched_weapon_indices"]),
                    "original_annotated_path": path_for_csv(original_path),
                    "raw_crop_path": path_for_csv(raw_crop_path),
                    "resized_crop_path": path_for_csv(resized_crop_path),
                    "resized_crop_annotated_path": path_for_csv(resized_annotated_path),
                    "panel_path": path_for_csv(panel_path),
                }
            )

        print(
            f"[OK] {subset_idx + 1:03d}/{len(selected_df):03d} {row['image_stem']}: "
            f"gt_weapons={len(weapon_boxes)} person_detections={len(detections)} matched_crops={len(records)}"
        )

    manifest_path = output_dir / "diagnostic_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    write_markdown_summary(output_dir, rows, args.split, args.num_images, resize_size)

    print(f"[OK] Saved diagnostics to: {output_dir}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Panels: {panels_dir}")


if __name__ == "__main__":
    main()
