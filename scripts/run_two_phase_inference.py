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
    crop_matches_weapon,
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
    point_in_box,
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
    parser.add_argument("--weapon-model", type=Path, default=None, help="Path to the crop-stage YOLO weapon checkpoint.")
    parser.add_argument(
        "--hold-checkpoint",
        "--carry-checkpoint",
        dest="hold_checkpoint",
        type=Path,
        default=None,
        help="Optional path to the hold/no_hold classifier checkpoint.",
    )
    parser.add_argument(
        "--hold-threshold",
        type=float,
        default=None,
        help="Override the hold/no_hold gate threshold saved in the checkpoint.",
    )
    parser.add_argument(
        "--enable-hold-gate",
        action="store_true",
        help="Enable the historical hold/no_hold gate before running the crop weapon detector.",
    )
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
    enable_hold_gate = bool(args.enable_hold_gate or config["inference"].get("enable_hold_gate", False))

    if not Path(weapon_model_path).exists():
        raise FileNotFoundError(
            f"Missing weapon detector checkpoint: {weapon_model_path}. "
            "Pass --weapon-model with the crop-trained YOLO best.pt path."
        )

    person_model = YOLO(person_model_name)
    weapon_model = YOLO(str(weapon_model_path))

    hold_model: CarryClassifierNet | None = None
    hold_transform = None
    hold_threshold: float | None = None
    threshold_source = "disabled"

    if enable_hold_gate:
        if not Path(hold_checkpoint_path).exists():
            raise FileNotFoundError(
                f"Missing hold/no_hold classifier checkpoint: {hold_checkpoint_path}. "
                "Run train_carry_classifier.py first or disable --enable-hold-gate."
            )
        checkpoint = torch.load(hold_checkpoint_path, map_location=torch_device)
        image_size = int(checkpoint.get("image_size", config["dataset"]["classifier_image_size"]))
        hold_threshold, threshold_source = resolve_hold_threshold(checkpoint, manual_override=args.hold_threshold)
        hold_model = CarryClassifierNet(image_size=image_size).to(torch_device)
        hold_model.load_state_dict(checkpoint["model_state_dict"])
        hold_model.eval()
        hold_transform = build_classifier_transform(image_size)

    output_prefix = args.output_prefix or split_name
    prediction_rows: list[dict[str, object]] = []
    pre_nms_prediction_rows: list[dict[str, object]] = []
    person_candidate_rows: list[dict[str, object]] = []
    image_summary_rows: list[dict[str, object]] = []
    aggregate = defaultdict(int)

    dataset_config = config["dataset"]
    inference_config = config["inference"]
    threshold_config = config["thresholds"]
    padding_x_fraction = float(dataset_config.get("crop_padding_x", dataset_config.get("crop_padding", 0.20)))
    padding_y_fraction = float(dataset_config.get("crop_padding_y", dataset_config.get("crop_padding", 0.20)))
    min_crop_side = int(dataset_config.get("min_crop_side", 1))
    match_ioa_threshold = float(dataset_config.get("match_ioa_threshold", 0.60))
    person_imgsz = int(inference_config.get("person_imgsz", 960))
    person_max_det = int(inference_config.get("person_max_det", 50))
    weapon_crop_imgsz = int(inference_config.get("weapon_crop_imgsz", 640))

    for _, row in manifest_df.iterrows():
        image_path = root / Path(row["image_path"])
        annotation_path = root / Path(row["annotation_path"])
        image = load_image(image_path)
        width, height = image.size

        person_result = person_model.predict(
            source=str(image_path),
            conf=float(threshold_config["person_conf"]),
            iou=float(threshold_config["person_iou"]),
            classes=[0],
            verbose=False,
            device=yolo_device,
            imgsz=person_imgsz,
            max_det=person_max_det,
        )[0]
        person_boxes = extract_yolo_boxes(person_result, allowed_class_ids={0})

        _, _, objects = parse_voc_xml(annotation_path)
        gt_weapon_boxes = filter_weapon_boxes(objects)

        raw_covered_weapon_indices: set[int] = set()
        expanded_covered_weapon_indices: set[int] = set()
        per_image_predictions_pre_nms: list[dict[str, object]] = []
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
            crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

            raw_match_count = 0
            expanded_match_count = 0
            for weapon_box in gt_weapon_boxes:
                center_x, center_y = center_of_box(weapon_box)
                if point_in_box(center_x, center_y, person_box):
                    raw_match_count += 1
                expanded_match, _, _ = crop_matches_weapon(
                    crop_box,
                    weapon_box,
                    match_ioa_threshold=match_ioa_threshold,
                )
                if expanded_match:
                    expanded_match_count += 1

            if raw_match_count > 0:
                raw_covered_weapon_indices.update(
                    weapon_idx
                    for weapon_idx, weapon_box in enumerate(gt_weapon_boxes)
                    if point_in_box(*center_of_box(weapon_box), person_box)
                )
            if expanded_match_count > 0:
                expanded_covered_weapon_indices.update(
                    weapon_idx
                    for weapon_idx, weapon_box in enumerate(gt_weapon_boxes)
                    if crop_matches_weapon(crop_box, weapon_box, match_ioa_threshold=match_ioa_threshold)[0]
                )

            hold_prob: float | None = None
            hold_passed = True
            if enable_hold_gate:
                if hold_model is None or hold_transform is None or hold_threshold is None:
                    raise RuntimeError("Hold gate was enabled but the classifier state was not initialized.")
                hold_prob = predict_hold_probability(hold_model, hold_transform, crop, torch_device)
                hold_passed = hold_prob >= hold_threshold
                if not hold_passed:
                    stage1_rejected += 1

            person_candidate_rows.append(
                {
                    "split": split_name,
                    "image_stem": row["image_stem"],
                    "image_path": row["image_path"],
                    "person_index": person_idx,
                    "person_box_id": person_idx,
                    "person_confidence": round(float(person_box["confidence"]), 6),
                    "person_xmin": round(float(person_box["xmin"]), 3),
                    "person_ymin": round(float(person_box["ymin"]), 3),
                    "person_xmax": round(float(person_box["xmax"]), 3),
                    "person_ymax": round(float(person_box["ymax"]), 3),
                    "crop_xmin": crop_xmin,
                    "crop_ymin": crop_ymin,
                    "crop_xmax": crop_xmax,
                    "crop_ymax": crop_ymax,
                    "raw_match_count": raw_match_count,
                    "expanded_match_count": expanded_match_count,
                    "crop_was_expanded_match": int(expanded_match_count > 0),
                    "hold_probability": round(hold_prob, 6) if hold_prob is not None else None,
                    "hold_gate_used": int(enable_hold_gate),
                    "hold_passed": int(hold_passed),
                }
            )

            if not hold_passed:
                continue

            stage2_passed += 1
            weapon_result = weapon_model.predict(
                source=crop,
                conf=float(threshold_config["weapon_conf"]),
                iou=float(threshold_config["weapon_iou"]),
                verbose=False,
                device=yolo_device,
                imgsz=weapon_crop_imgsz,
            )[0]
            weapon_boxes = extract_yolo_boxes(weapon_result)
            for weapon_idx, weapon_box in enumerate(weapon_boxes):
                global_xmin = crop_xmin + float(weapon_box["xmin"])
                global_ymin = crop_ymin + float(weapon_box["ymin"])
                global_xmax = crop_xmin + float(weapon_box["xmax"])
                global_ymax = crop_ymin + float(weapon_box["ymax"])
                per_image_predictions_pre_nms.append(
                    {
                        "split": split_name,
                        "image_stem": row["image_stem"],
                        "image_path": row["image_path"],
                        "person_index": person_idx,
                        "person_box_id": person_idx,
                        "weapon_index": weapon_idx,
                        "person_confidence": round(float(person_box["confidence"]), 6),
                        "hold_probability": round(hold_prob, 6) if hold_prob is not None else None,
                        "carry_probability": round(hold_prob, 6) if hold_prob is not None else None,
                        "weapon_confidence": round(float(weapon_box["confidence"]), 6),
                        "xmin": round(global_xmin, 3),
                        "ymin": round(global_ymin, 3),
                        "xmax": round(global_xmax, 3),
                        "ymax": round(global_ymax, 3),
                        "crop_xmin": crop_xmin,
                        "crop_ymin": crop_ymin,
                        "crop_xmax": crop_xmax,
                        "crop_ymax": crop_ymax,
                        "crop_was_expanded_match": int(expanded_match_count > 0),
                        "hold_gate_used": int(enable_hold_gate),
                    }
                )

        raw_person_cover = len(raw_covered_weapon_indices)
        expanded_crop_cover = len(expanded_covered_weapon_indices)
        stage0_raw_person_miss = 1 if gt_weapon_boxes and raw_person_cover == 0 else 0
        stage0_expanded_crop_miss = 1 if gt_weapon_boxes and expanded_crop_cover == 0 else 0
        stage0_image_miss = stage0_expanded_crop_miss

        final_predictions: list[dict[str, object]] = []
        if per_image_predictions_pre_nms:
            boxes = [
                {"xmin": pred["xmin"], "ymin": pred["ymin"], "xmax": pred["xmax"], "ymax": pred["ymax"]}
                for pred in per_image_predictions_pre_nms
            ]
            scores = [float(pred["weapon_confidence"]) for pred in per_image_predictions_pre_nms]
            keep_indices = nms_indices(boxes, scores, iou_threshold=float(threshold_config["image_level_nms_iou"]))
            keep_index_set = set(keep_indices)
            for pred_idx, pred_row in enumerate(per_image_predictions_pre_nms):
                debug_row = dict(pred_row)
                debug_row["kept_after_image_nms"] = int(pred_idx in keep_index_set)
                pre_nms_prediction_rows.append(debug_row)
            final_predictions = [per_image_predictions_pre_nms[idx] for idx in keep_indices]

        prediction_rows.extend(final_predictions)

        detections_before_nms = len(per_image_predictions_pre_nms)
        detections_after_nms = len(final_predictions)
        detections_removed_by_nms = detections_before_nms - detections_after_nms

        aggregate["images_processed"] += 1
        aggregate["gt_weapon_boxes"] += len(gt_weapon_boxes)
        aggregate["persons_detected"] += len(person_boxes)
        aggregate["persons_filtered_out"] += stage1_rejected
        aggregate["persons_passed_stage2"] += stage2_passed
        aggregate["final_weapon_detections_before_nms"] += detections_before_nms
        aggregate["final_weapon_detections"] += detections_after_nms
        aggregate["final_weapon_detections_removed_by_nms"] += detections_removed_by_nms
        aggregate["stage0_raw_person_miss_images"] += stage0_raw_person_miss
        aggregate["stage0_miss_images"] += stage0_image_miss
        aggregate["raw_person_cover"] += raw_person_cover
        aggregate["expanded_crop_cover"] += expanded_crop_cover

        image_summary_rows.append(
            {
                "split": split_name,
                "image_stem": row["image_stem"],
                "image_path": row["image_path"],
                "gt_weapon_boxes": len(gt_weapon_boxes),
                "stage0_image_miss": stage0_image_miss,
                "stage0_raw_person_miss": stage0_raw_person_miss,
                "stage0_expanded_crop_miss": stage0_expanded_crop_miss,
                "raw_person_cover": raw_person_cover,
                "expanded_crop_cover": expanded_crop_cover,
                "persons_detected": len(person_boxes),
                "persons_rejected_by_hold_gate": stage1_rejected,
                "persons_filtered_out": stage1_rejected,
                "persons_passed_stage2": stage2_passed,
                "final_weapon_detections_before_nms": detections_before_nms,
                "final_weapon_detections_removed_by_nms": detections_removed_by_nms,
                "final_weapon_detections": detections_after_nms,
                "hold_gate_used": int(enable_hold_gate),
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
                "person_box_id",
                "weapon_index",
                "person_confidence",
                "hold_probability",
                "carry_probability",
                "weapon_confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "crop_xmin",
                "crop_ymin",
                "crop_xmax",
                "crop_ymax",
                "crop_was_expanded_match",
                "hold_gate_used",
            ]
        )
    pre_nms_predictions_df = pd.DataFrame(pre_nms_prediction_rows)
    if pre_nms_predictions_df.empty:
        pre_nms_predictions_df = pd.DataFrame(columns=list(predictions_df.columns) + ["kept_after_image_nms"])
    person_candidates_df = pd.DataFrame(person_candidate_rows)
    if person_candidates_df.empty:
        person_candidates_df = pd.DataFrame(
            columns=[
                "split",
                "image_stem",
                "image_path",
                "person_index",
                "person_box_id",
                "person_confidence",
                "person_xmin",
                "person_ymin",
                "person_xmax",
                "person_ymax",
                "crop_xmin",
                "crop_ymin",
                "crop_xmax",
                "crop_ymax",
                "raw_match_count",
                "expanded_match_count",
                "crop_was_expanded_match",
                "hold_probability",
                "hold_gate_used",
                "hold_passed",
            ]
        )
    image_summary_df = pd.DataFrame(image_summary_rows)
    pipeline_summary_df = pd.DataFrame(
        [
            {
                "split": split_name,
                "images_processed": aggregate["images_processed"],
                "gt_weapon_boxes": aggregate["gt_weapon_boxes"],
                "raw_person_cover": aggregate["raw_person_cover"],
                "expanded_crop_cover": aggregate["expanded_crop_cover"],
                "persons_detected": aggregate["persons_detected"],
                "persons_rejected_by_hold_gate": aggregate["persons_filtered_out"],
                "persons_filtered_out": aggregate["persons_filtered_out"],
                "persons_passed_stage2": aggregate["persons_passed_stage2"],
                "final_weapon_detections_before_nms": aggregate["final_weapon_detections_before_nms"],
                "final_weapon_detections_removed_by_nms": aggregate["final_weapon_detections_removed_by_nms"],
                "final_weapon_detections": aggregate["final_weapon_detections"],
                "stage0_raw_person_miss_images": aggregate["stage0_raw_person_miss_images"],
                "stage0_miss_images": aggregate["stage0_miss_images"],
                "hold_gate_used": int(enable_hold_gate),
                "hold_threshold": hold_threshold if hold_threshold is not None else "",
                "carry_threshold": hold_threshold if hold_threshold is not None else "",
                "threshold_source": threshold_source,
            }
        ]
    )

    predictions_path = predictions_dir / f"{output_prefix}_predictions.csv"
    predictions_pre_nms_path = predictions_dir / f"{output_prefix}_predictions_pre_nms.csv"
    person_candidates_path = predictions_dir / f"{output_prefix}_person_candidates.csv"
    image_summary_path = predictions_dir / f"{output_prefix}_image_summary.csv"
    pipeline_summary_path = predictions_dir / f"{output_prefix}_pipeline_summary.csv"

    predictions_df.to_csv(predictions_path, index=False)
    pre_nms_predictions_df.to_csv(predictions_pre_nms_path, index=False)
    person_candidates_df.to_csv(person_candidates_path, index=False)
    image_summary_df.to_csv(image_summary_path, index=False)
    pipeline_summary_df.to_csv(pipeline_summary_path, index=False)

    print(f"[OK] Saved predictions: {predictions_path}")
    print(f"[OK] Saved pre-NMS predictions: {predictions_pre_nms_path}")
    print(f"[OK] Saved person candidates: {person_candidates_path}")
    print(f"[OK] Saved image summary: {image_summary_path}")
    print(f"[OK] Saved pipeline summary: {pipeline_summary_path}")
    if hold_threshold is not None:
        print(f"[OK] Hold threshold used: {hold_threshold:.2f} ({threshold_source})")
    else:
        print("[OK] Hold gate disabled for this run.")


if __name__ == "__main__":
    main()
