from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ultralytics import YOLO

from two_phase_utils import (
    CarryClassifierNet,
    build_classifier_transform,
    center_of_box,
    ensure_dir,
    expand_box,
    extract_yolo_boxes,
    filter_weapon_boxes,
    load_image,
    load_split_manifest,
    load_two_phase_config,
    nms_indices,
    normalize_torch_device,
    parse_voc_xml,
    project_root,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Sprint 4 two-phase weapon detection pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=["train", "val", "test"], help="Named split manifest to process.")
    group.add_argument("--manifest", type=Path, help="Explicit manifest CSV with image_path and annotation_path.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/two_phase.yaml.")
    parser.add_argument("--person-model", default=None, help="Override person detector model or weights path.")
    parser.add_argument("--weapon-model", type=Path, default=None, help="Path to the Sprint 3 YOLO26n checkpoint.")
    parser.add_argument("--hold-checkpoint", "--carry-checkpoint", dest="hold_checkpoint", type=Path, default=None, help="Path to the hold/no_hold classifier checkpoint.")
    parser.add_argument("--hold-threshold", type=float, default=None, help="Override the hold/no_hold gate threshold saved in the checkpoint.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--output-prefix", default=None, help="Optional prefix for saved output filenames.")
    return parser.parse_args()


def resolve_manifest(args: argparse.Namespace, root: Path) -> tuple[str, pd.DataFrame]:
    if args.split:
        return args.split, load_split_manifest(root, args.split)

    manifest_path = args.manifest
    if manifest_path is None:
        raise ValueError("Expected --split or --manifest.")
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    return manifest_path.stem, pd.read_csv(manifest_path)


def predict_hold_probability(
    model: CarryClassifierNet,
    transform: Any,
    crop_image,
    device: str,
) -> float:
    tensor = transform(crop_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        return float(torch.sigmoid(logits).item())


def resolve_hold_threshold(checkpoint: dict[str, Any], manual_override: float | None) -> tuple[float, str]:
    if manual_override is not None:
        return float(manual_override), "manual_override"
    if "stage1_gate_threshold" in checkpoint:
        return float(checkpoint["stage1_gate_threshold"]), "checkpoint_stage1_gate_threshold"
    if "best_threshold" in checkpoint:
        return float(checkpoint["best_threshold"]), "checkpoint_best_threshold"
    return 0.50, "default_fallback"


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_two_phase_config(args.config)
    split_name, manifest_df = resolve_manifest(args, root)
    if args.max_images is not None:
        manifest_df = manifest_df.head(args.max_images).copy()

    predictions_dir = resolve_path(root, config["paths"]["predictions_dir"])
    ensure_dir(predictions_dir)

    yolo_device = args.device or config["inference"]["default_device"]
    torch_device = normalize_torch_device(yolo_device)
    person_model_name = args.person_model or config["models"]["person_detector"]
    weapon_model_path = args.weapon_model or resolve_path(root, config["models"]["weapon_detector"])
    hold_checkpoint_path = args.hold_checkpoint or resolve_path(root, config["paths"]["carry_runs_dir"]) / "best.pt"

    if not Path(weapon_model_path).exists():
        raise FileNotFoundError(
            f"Missing weapon detector checkpoint: {weapon_model_path}. "
            "Pass --weapon-model with the Sprint 3 YOLO26n best.pt path."
        )
    if not Path(hold_checkpoint_path).exists():
        raise FileNotFoundError(
            f"Missing hold/no_hold classifier checkpoint: {hold_checkpoint_path}. "
            "Run train_carry_classifier.py first or pass --hold-checkpoint."
        )

    person_model = YOLO(person_model_name)
    weapon_model = YOLO(str(weapon_model_path))

    checkpoint = torch.load(hold_checkpoint_path, map_location=torch_device)
    image_size = int(checkpoint.get("image_size", config["dataset"]["classifier_image_size"]))
    hold_threshold, threshold_source = resolve_hold_threshold(checkpoint, manual_override=args.hold_threshold)
    hold_model = CarryClassifierNet(image_size=image_size).to(torch_device)
    hold_model.load_state_dict(checkpoint["model_state_dict"])
    hold_model.eval()
    hold_transform = build_classifier_transform(image_size)

    output_prefix = args.output_prefix or split_name
    prediction_rows: list[dict[str, object]] = []
    image_summary_rows: list[dict[str, object]] = []
    aggregate = defaultdict(int)

    for _, row in manifest_df.iterrows():
        image_path = root / Path(row["image_path"])
        annotation_path = root / Path(row["annotation_path"])
        image = load_image(image_path)
        width, height = image.size

        person_result = person_model.predict(
            source=str(image_path),
            conf=float(config["thresholds"]["person_conf"]),
            iou=float(config["thresholds"]["person_iou"]),
            classes=[0],
            verbose=False,
            device=yolo_device,
        )[0]
        person_boxes = extract_yolo_boxes(person_result, allowed_class_ids={0})

        _, _, objects = parse_voc_xml(annotation_path)
        gt_weapon_boxes = filter_weapon_boxes(objects)
        stage0_image_miss = 0
        if gt_weapon_boxes:
            has_candidate_cover = False
            for gt_weapon_box in gt_weapon_boxes:
                center_x, center_y = center_of_box(gt_weapon_box)
                if any(
                    person_box["xmin"] <= center_x <= person_box["xmax"]
                    and person_box["ymin"] <= center_y <= person_box["ymax"]
                    for person_box in person_boxes
                ):
                    has_candidate_cover = True
                    break
            if not has_candidate_cover:
                stage0_image_miss = 1

        per_image_predictions: list[dict[str, object]] = []
        stage1_rejected = 0
        stage2_passed = 0

        for person_idx, person_box in enumerate(person_boxes):
            crop_xmin, crop_ymin, crop_xmax, crop_ymax = expand_box(
                person_box["xmin"],
                person_box["ymin"],
                person_box["xmax"],
                person_box["ymax"],
                width=width,
                height=height,
                padding_fraction=float(config["dataset"]["crop_padding"]),
            )
            crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
            hold_prob = predict_hold_probability(hold_model, hold_transform, crop, torch_device)
            if hold_prob < hold_threshold:
                stage1_rejected += 1
                continue

            stage2_passed += 1
            weapon_result = weapon_model.predict(
                source=crop,
                conf=float(config["thresholds"]["weapon_conf"]),
                iou=float(config["thresholds"]["weapon_iou"]),
                verbose=False,
                device=yolo_device,
            )[0]
            weapon_boxes = extract_yolo_boxes(weapon_result)
            for weapon_idx, weapon_box in enumerate(weapon_boxes):
                global_xmin = crop_xmin + float(weapon_box["xmin"])
                global_ymin = crop_ymin + float(weapon_box["ymin"])
                global_xmax = crop_xmin + float(weapon_box["xmax"])
                global_ymax = crop_ymin + float(weapon_box["ymax"])
                per_image_predictions.append(
                    {
                        "split": split_name,
                        "image_stem": row["image_stem"],
                        "image_path": row["image_path"],
                        "person_index": person_idx,
                        "weapon_index": weapon_idx,
                        "person_confidence": round(float(person_box["confidence"]), 6),
                        "hold_probability": round(hold_prob, 6),
                        "carry_probability": round(hold_prob, 6),
                        "weapon_confidence": round(float(weapon_box["confidence"]), 6),
                        "xmin": round(global_xmin, 3),
                        "ymin": round(global_ymin, 3),
                        "xmax": round(global_xmax, 3),
                        "ymax": round(global_ymax, 3),
                    }
                )

        if per_image_predictions:
            boxes = [
                {"xmin": pred["xmin"], "ymin": pred["ymin"], "xmax": pred["xmax"], "ymax": pred["ymax"]}
                for pred in per_image_predictions
            ]
            scores = [float(pred["weapon_confidence"]) for pred in per_image_predictions]
            keep_indices = nms_indices(boxes, scores, iou_threshold=float(config["thresholds"]["image_level_nms_iou"]))
            per_image_predictions = [per_image_predictions[idx] for idx in keep_indices]

        aggregate["images_processed"] += 1
        aggregate["persons_detected"] += len(person_boxes)
        aggregate["persons_filtered_out"] += stage1_rejected
        aggregate["persons_passed_stage2"] += stage2_passed
        aggregate["final_weapon_detections"] += len(per_image_predictions)
        aggregate["stage0_miss_images"] += stage0_image_miss

        prediction_rows.extend(per_image_predictions)
        image_summary_rows.append(
            {
                "split": split_name,
                "image_stem": row["image_stem"],
                "image_path": row["image_path"],
                "stage0_image_miss": stage0_image_miss,
                "persons_detected": len(person_boxes),
                "persons_rejected_by_hold_gate": stage1_rejected,
                "persons_filtered_out": stage1_rejected,
                "persons_passed_stage2": stage2_passed,
                "final_weapon_detections": len(per_image_predictions),
            }
        )

    predictions_df = pd.DataFrame(prediction_rows)
    if predictions_df.empty:
        predictions_df = pd.DataFrame(
            columns=[
                "split",
                "image_stem",
                "image_path",
                "person_index",
                "weapon_index",
                "person_confidence",
                "hold_probability",
                "carry_probability",
                "weapon_confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
        )
    image_summary_df = pd.DataFrame(image_summary_rows)
    pipeline_summary_df = pd.DataFrame(
        [
            {
                "split": split_name,
                "images_processed": aggregate["images_processed"],
                "persons_detected": aggregate["persons_detected"],
                "persons_rejected_by_hold_gate": aggregate["persons_filtered_out"],
                "persons_filtered_out": aggregate["persons_filtered_out"],
                "persons_passed_stage2": aggregate["persons_passed_stage2"],
                "final_weapon_detections": aggregate["final_weapon_detections"],
                "stage0_miss_images": aggregate["stage0_miss_images"],
                "hold_threshold": hold_threshold,
                "carry_threshold": hold_threshold,
                "threshold_source": threshold_source,
            }
        ]
    )

    predictions_path = predictions_dir / f"{output_prefix}_predictions.csv"
    image_summary_path = predictions_dir / f"{output_prefix}_image_summary.csv"
    pipeline_summary_path = predictions_dir / f"{output_prefix}_pipeline_summary.csv"

    predictions_df.to_csv(predictions_path, index=False)
    image_summary_df.to_csv(image_summary_path, index=False)
    pipeline_summary_df.to_csv(pipeline_summary_path, index=False)

    print(f"[OK] Saved predictions: {predictions_path}")
    print(f"[OK] Saved image summary: {image_summary_path}")
    print(f"[OK] Saved pipeline summary: {pipeline_summary_path}")
    print(f"[OK] Hold threshold used: {hold_threshold:.2f} ({threshold_source})")


if __name__ == "__main__":
    main()
