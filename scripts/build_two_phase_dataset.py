from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from two_phase_utils import (
    center_of_box,
    crop_matches_weapon,
    ensure_dir,
    expand_box,
    extract_yolo_boxes,
    filter_weapon_boxes,
    load_image,
    load_split_manifest,
    load_two_phase_config,
    path_for_csv,
    point_in_box,
    project_box_into_crop,
    project_root,
    remove_dir_if_exists,
    resolve_path,
    parse_voc_xml,
    yolo_label_line_for_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build person crops for the Sprint 4 hold/no_hold screening stage.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/two_phase.yaml.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--person-model", default=None, help="Override person detector model or weights path.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--max-images-per-split", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing Sprint 4 crop outputs for the selected split(s) before rebuilding.",
    )
    return parser.parse_args()


def write_text_lines(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")


def write_yolo_dataset_yaml(dataset_root: Path) -> None:
    dataset_yaml_path = dataset_root / "dataset.yaml"
    dataset_yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: weapon",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_two_phase_config(args.config)
    device = args.device or config["inference"]["default_device"]

    two_phase_root = resolve_path(root, config["paths"]["two_phase_root"])
    crops_root = two_phase_root / "crops"
    metadata_root = two_phase_root / "metadata"
    yolo_crops_root = two_phase_root / "yolo_crops"
    yolo_images_root = yolo_crops_root / "images"
    yolo_labels_root = yolo_crops_root / "labels"

    if args.overwrite:
        for split_name in args.splits:
            remove_dir_if_exists(crops_root / split_name)
            remove_dir_if_exists(yolo_images_root / split_name)
            remove_dir_if_exists(yolo_labels_root / split_name)
            for metadata_name in [
                f"{split_name}_person_crops.csv",
                f"{split_name}_stage0_misses.csv",
                f"{split_name}_yolo_crops.csv",
            ]:
                metadata_path = metadata_root / metadata_name
                if metadata_path.exists():
                    metadata_path.unlink()

    for split_name in args.splits:
        split_dir = crops_root / split_name
        yolo_split_dir = yolo_images_root / split_name
        if (split_dir.exists() or yolo_split_dir.exists()) and not args.overwrite:
            raise FileExistsError(
                f"Output already exists for split '{split_name}' at {split_dir}. "
                "Use --overwrite to rebuild and avoid stale crops."
            )

    ensure_dir(crops_root)
    ensure_dir(metadata_root)
    ensure_dir(yolo_images_root)
    ensure_dir(yolo_labels_root)

    person_model_name = args.person_model or config["models"]["person_detector"]
    person_model = YOLO(person_model_name)

    dataset_config = config["dataset"]
    inference_config = config["inference"]
    threshold_config = config["thresholds"]

    padding_x_fraction = float(dataset_config.get("crop_padding_x", dataset_config.get("crop_padding", 0.20)))
    padding_y_fraction = float(dataset_config.get("crop_padding_y", dataset_config.get("crop_padding", 0.20)))
    min_crop_side = int(dataset_config.get("min_crop_side", 1))
    max_negatives_per_image = int(dataset_config["max_negatives_per_image"])
    match_ioa_threshold = float(dataset_config.get("match_ioa_threshold", 0.60))
    person_imgsz = int(inference_config.get("person_imgsz", 960))
    person_max_det = int(inference_config.get("person_max_det", 50))

    summary_rows: list[dict[str, object]] = []

    for split_name in args.splits:
        split_df = load_split_manifest(root, split_name)
        if args.max_images_per_split is not None:
            split_df = split_df.head(args.max_images_per_split).copy()

        ensure_dir(crops_root / split_name / "hold")
        ensure_dir(crops_root / split_name / "no_hold")
        ensure_dir(yolo_images_root / split_name)
        ensure_dir(yolo_labels_root / split_name)

        crop_rows: list[dict[str, object]] = []
        stage0_miss_rows: list[dict[str, object]] = []
        yolo_rows: list[dict[str, object]] = []
        counters = defaultdict(int)

        for _, row in split_df.iterrows():
            image_path = root / Path(row["image_path"])
            annotation_path = root / Path(row["annotation_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file: {image_path}")
            if not annotation_path.exists():
                raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

            width, height, objects = parse_voc_xml(annotation_path)
            if width is None or height is None:
                raise ValueError(f"Missing size information in XML: {annotation_path}")

            image = load_image(image_path)
            weapon_boxes = filter_weapon_boxes(objects)
            counters["images_processed"] += 1
            counters["gt_weapon_boxes"] += len(weapon_boxes)
            if weapon_boxes:
                counters["images_with_weapon"] += 1

            result = person_model.predict(
                source=str(image_path),
                conf=float(threshold_config["person_conf"]),
                iou=float(threshold_config["person_iou"]),
                classes=[0],
                verbose=False,
                device=device,
                imgsz=person_imgsz,
                max_det=person_max_det,
            )[0]
            detections = extract_yolo_boxes(result, allowed_class_ids={0})
            counters["person_detections"] += len(detections)

            raw_covered_weapon_indices: set[int] = set()
            expanded_covered_weapon_indices: set[int] = set()
            detection_records: list[dict[str, object]] = []

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

                raw_match_indices: list[int] = []
                expanded_matches: list[dict[str, object]] = []
                for weapon_idx, weapon_box in enumerate(weapon_boxes):
                    center_x, center_y = center_of_box(weapon_box)
                    raw_center_match = point_in_box(center_x, center_y, det_box)
                    if raw_center_match:
                        raw_match_indices.append(weapon_idx)
                        raw_covered_weapon_indices.add(weapon_idx)

                    expanded_match, expanded_center_match, expanded_ioa = crop_matches_weapon(
                        crop_box,
                        weapon_box,
                        match_ioa_threshold=match_ioa_threshold,
                    )
                    if expanded_match:
                        expanded_matches.append(
                            {
                                "weapon_index": weapon_idx,
                                "center_match": expanded_center_match,
                                "ioa": expanded_ioa,
                            }
                        )
                        expanded_covered_weapon_indices.add(weapon_idx)

                detection_records.append(
                    {
                        "det_idx": det_idx,
                        "det_box": det_box,
                        "crop_box": crop_box,
                        "crop_coords": (crop_xmin, crop_ymin, crop_xmax, crop_ymax),
                        "raw_match_indices": raw_match_indices,
                        "expanded_matches": expanded_matches,
                        "raw_match_count": len(raw_match_indices),
                        "expanded_match_count": len(expanded_matches),
                    }
                )

            counters["raw_person_cover"] += len(raw_covered_weapon_indices)
            counters["expanded_crop_cover"] += len(expanded_covered_weapon_indices)
            counters["stage0_missed_weapon_boxes"] += max(0, len(weapon_boxes) - len(expanded_covered_weapon_indices))

            if weapon_boxes and not expanded_covered_weapon_indices:
                counters["stage0_miss_images"] += 1
                stage0_miss_rows.append(
                    {
                        "split": split_name,
                        "image_stem": row["image_stem"],
                        "image_path": row["image_path"],
                        "annotation_path": row["annotation_path"],
                        "num_gt_weapon_boxes": len(weapon_boxes),
                        "num_person_detections": len(detections),
                        "raw_person_cover": len(raw_covered_weapon_indices),
                        "expanded_crop_cover": len(expanded_covered_weapon_indices),
                    }
                )

            positive_indices = {int(record["det_idx"]) for record in detection_records if int(record["expanded_match_count"]) > 0}
            negative_indices = [idx for idx in range(len(detections)) if idx not in positive_indices]
            negative_indices.sort(key=lambda idx: detections[idx]["confidence"], reverse=True)
            kept_negative_indices = negative_indices[:max_negatives_per_image]
            classifier_kept_indices = positive_indices | set(kept_negative_indices)

            for record in detection_records:
                det_idx = int(record["det_idx"])
                det_box = record["det_box"]
                crop_box = record["crop_box"]
                crop_xmin, crop_ymin, crop_xmax, crop_ymax = record["crop_coords"]
                crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
                crop_width = crop_xmax - crop_xmin
                crop_height = crop_ymax - crop_ymin

                if det_idx in classifier_kept_indices:
                    label = "hold" if det_idx in positive_indices else "no_hold"
                    legacy_label = "carry" if label == "hold" else "no_carry"
                    crop_filename = f"{row['image_stem']}_person_{det_idx:02d}_{label}.jpg"
                    crop_output_path = crops_root / split_name / label / crop_filename
                    crop.save(crop_output_path, quality=95)

                    counters[f"{label}_crops"] += 1
                    crop_rows.append(
                        {
                            "split": split_name,
                            "image_stem": row["image_stem"],
                            "image_path": row["image_path"],
                            "annotation_path": row["annotation_path"],
                            "crop_filename": crop_filename,
                            "crop_path": path_for_csv(crop_output_path),
                            "label": label,
                            "label_id": 1 if label == "hold" else 0,
                            "legacy_label": legacy_label,
                            "person_confidence": round(float(det_box["confidence"]), 6),
                            "person_xmin": round(float(det_box["xmin"]), 3),
                            "person_ymin": round(float(det_box["ymin"]), 3),
                            "person_xmax": round(float(det_box["xmax"]), 3),
                            "person_ymax": round(float(det_box["ymax"]), 3),
                            "crop_xmin": crop_xmin,
                            "crop_ymin": crop_ymin,
                            "crop_xmax": crop_xmax,
                            "crop_ymax": crop_ymax,
                            "raw_match_count": int(record["raw_match_count"]),
                            "expanded_match_count": int(record["expanded_match_count"]),
                            "matched_weapon_indices": ";".join(
                                str(match["weapon_index"]) for match in record["expanded_matches"]
                            ),
                            "num_gt_weapon_boxes_in_image": len(weapon_boxes),
                        }
                    )

                yolo_crop_filename = f"{row['image_stem']}_person_{det_idx:02d}.jpg"
                yolo_crop_output_path = yolo_images_root / split_name / yolo_crop_filename
                crop.save(yolo_crop_output_path, quality=95)

                yolo_label_lines: list[str] = []
                for match in record["expanded_matches"]:
                    weapon_box = weapon_boxes[int(match["weapon_index"])]
                    projected_box = project_box_into_crop(weapon_box, crop_box)
                    if projected_box is None:
                        continue
                    yolo_label_lines.append(
                        yolo_label_line_for_box(
                            projected_box,
                            image_width=crop_width,
                            image_height=crop_height,
                            class_id=0,
                        )
                    )

                yolo_label_output_path = yolo_labels_root / split_name / f"{Path(yolo_crop_filename).stem}.txt"
                write_text_lines(yolo_label_output_path, yolo_label_lines)
                counters["yolo_crops"] += 1
                if yolo_label_lines:
                    counters["yolo_positive_crops"] += 1
                else:
                    counters["yolo_negative_crops"] += 1

                yolo_rows.append(
                    {
                        "split": split_name,
                        "image_stem": row["image_stem"],
                        "image_path": row["image_path"],
                        "annotation_path": row["annotation_path"],
                        "crop_filename": yolo_crop_filename,
                        "crop_path": path_for_csv(yolo_crop_output_path),
                        "label_path": path_for_csv(yolo_label_output_path),
                        "person_index": det_idx,
                        "person_confidence": round(float(det_box["confidence"]), 6),
                        "person_xmin": round(float(det_box["xmin"]), 3),
                        "person_ymin": round(float(det_box["ymin"]), 3),
                        "person_xmax": round(float(det_box["xmax"]), 3),
                        "person_ymax": round(float(det_box["ymax"]), 3),
                        "crop_xmin": crop_xmin,
                        "crop_ymin": crop_ymin,
                        "crop_xmax": crop_xmax,
                        "crop_ymax": crop_ymax,
                        "raw_match_count": int(record["raw_match_count"]),
                        "expanded_match_count": int(record["expanded_match_count"]),
                        "yolo_label_count": len(yolo_label_lines),
                    }
                )

        metadata_path = metadata_root / f"{split_name}_person_crops.csv"
        misses_path = metadata_root / f"{split_name}_stage0_misses.csv"
        yolo_metadata_path = metadata_root / f"{split_name}_yolo_crops.csv"

        crop_df = pd.DataFrame(
            crop_rows,
            columns=[
                "split",
                "image_stem",
                "image_path",
                "annotation_path",
                "crop_filename",
                "crop_path",
                "label",
                "label_id",
                "legacy_label",
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
                "matched_weapon_indices",
                "num_gt_weapon_boxes_in_image",
            ],
        )
        miss_df = pd.DataFrame(
            stage0_miss_rows,
            columns=[
                "split",
                "image_stem",
                "image_path",
                "annotation_path",
                "num_gt_weapon_boxes",
                "num_person_detections",
                "raw_person_cover",
                "expanded_crop_cover",
            ],
        )
        yolo_df = pd.DataFrame(
            yolo_rows,
            columns=[
                "split",
                "image_stem",
                "image_path",
                "annotation_path",
                "crop_filename",
                "crop_path",
                "label_path",
                "person_index",
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
                "yolo_label_count",
            ],
        )
        crop_df.to_csv(metadata_path, index=False)
        miss_df.to_csv(misses_path, index=False)
        yolo_df.to_csv(yolo_metadata_path, index=False)

        summary_rows.append(
            {
                "split": split_name,
                "images_processed": counters["images_processed"],
                "images_with_weapon": counters["images_with_weapon"],
                "gt_weapon_boxes": counters["gt_weapon_boxes"],
                "person_detections": counters["person_detections"],
                "raw_person_cover": counters["raw_person_cover"],
                "expanded_crop_cover": counters["expanded_crop_cover"],
                "hold_crops": counters["hold_crops"],
                "no_hold_crops": counters["no_hold_crops"],
                "carry_crops": counters["hold_crops"],
                "no_carry_crops": counters["no_hold_crops"],
                "yolo_crops": counters["yolo_crops"],
                "yolo_positive_crops": counters["yolo_positive_crops"],
                "yolo_negative_crops": counters["yolo_negative_crops"],
                "stage0_miss_images": counters["stage0_miss_images"],
                "stage0_missed_weapon_boxes": counters["stage0_missed_weapon_boxes"],
            }
        )

        print(
            f"[OK] {split_name}: "
            f"images={counters['images_processed']} "
            f"hold_crops={counters['hold_crops']} "
            f"no_hold_crops={counters['no_hold_crops']} "
            f"yolo_crops={counters['yolo_crops']} "
            f"stage0_miss_images={counters['stage0_miss_images']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = metadata_root / "two_phase_dataset_summary.csv"
    summary_md_path = root / "docs" / "sprint4_two_phase_dataset_summary.md"
    summary_df.to_csv(summary_csv_path, index=False)
    write_yolo_dataset_yaml(yolo_crops_root)

    with summary_md_path.open("w", encoding="utf-8") as f:
        f.write("# Sprint 4 - Two-Phase Dataset Summary\n\n")
        f.write("## Split summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Labeling rule\n\n")
        f.write("- `hold`: the expanded crop contains the weapon center or reaches the configured weapon IOA threshold.\n")
        f.write("- `no_hold`: detected person crop with no weapon match after crop expansion.\n")
        f.write("- `knife` is ignored exactly as in Sprint 3.\n")
        f.write("- `yolo_crops/` stores one padded person crop per detected person, with clipped YOLO labels for matched weapons.\n")

    print(f"[OK] Saved: {summary_csv_path}")
    print(f"[OK] Saved: {summary_md_path}")
    print(f"[OK] Saved: {yolo_crops_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
