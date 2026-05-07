from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from two_phase_utils import (
    center_of_box,
    ensure_dir,
    expand_box,
    extract_yolo_boxes,
    filter_weapon_boxes,
    load_image,
    load_split_manifest,
    load_two_phase_config,
    path_for_csv,
    project_root,
    remove_dir_if_exists,
    resolve_path,
    parse_voc_xml,
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


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_two_phase_config(args.config)
    device = args.device or config["inference"]["default_device"]

    two_phase_root = resolve_path(root, config["paths"]["two_phase_root"])
    crops_root = two_phase_root / "crops"
    metadata_root = two_phase_root / "metadata"

    if args.overwrite:
        for split_name in args.splits:
            remove_dir_if_exists(crops_root / split_name)
            metadata_path = metadata_root / f"{split_name}_person_crops.csv"
            misses_path = metadata_root / f"{split_name}_stage0_misses.csv"
            if metadata_path.exists():
                metadata_path.unlink()
            if misses_path.exists():
                misses_path.unlink()

    for split_name in args.splits:
        split_dir = crops_root / split_name
        if split_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists for split '{split_name}' at {split_dir}. "
                "Use --overwrite to rebuild and avoid stale crops."
            )

    ensure_dir(crops_root)
    ensure_dir(metadata_root)

    person_model_name = args.person_model or config["models"]["person_detector"]
    person_model = YOLO(person_model_name)

    padding_fraction = float(config["dataset"]["crop_padding"])
    max_negatives_per_image = int(config["dataset"]["max_negatives_per_image"])

    summary_rows: list[dict[str, object]] = []

    for split_name in args.splits:
        split_df = load_split_manifest(root, split_name)
        if args.max_images_per_split is not None:
            split_df = split_df.head(args.max_images_per_split).copy()

        ensure_dir(crops_root / split_name / "hold")
        ensure_dir(crops_root / split_name / "no_hold")

        crop_rows: list[dict[str, object]] = []
        stage0_miss_rows: list[dict[str, object]] = []
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
                conf=float(config["thresholds"]["person_conf"]),
                iou=float(config["thresholds"]["person_iou"]),
                classes=[0],
                verbose=False,
                device=device,
            )[0]
            detections = extract_yolo_boxes(result, allowed_class_ids={0})
            counters["person_detections"] += len(detections)

            positive_indices: set[int] = set()
            matches_by_index: dict[int, int] = defaultdict(int)
            for weapon_box in weapon_boxes:
                center_x, center_y = center_of_box(weapon_box)
                matched = False
                for det_idx, det_box in enumerate(detections):
                    if det_box["xmin"] <= center_x <= det_box["xmax"] and det_box["ymin"] <= center_y <= det_box["ymax"]:
                        positive_indices.add(det_idx)
                        matches_by_index[det_idx] += 1
                        matched = True
                if not matched:
                    counters["stage0_missed_weapon_boxes"] += 1

            if weapon_boxes and not positive_indices:
                counters["stage0_miss_images"] += 1
                stage0_miss_rows.append(
                    {
                        "split": split_name,
                        "image_stem": row["image_stem"],
                        "image_path": row["image_path"],
                        "annotation_path": row["annotation_path"],
                        "num_gt_weapon_boxes": len(weapon_boxes),
                        "num_person_detections": len(detections),
                    }
                )

            negative_indices = [idx for idx in range(len(detections)) if idx not in positive_indices]
            negative_indices.sort(key=lambda idx: detections[idx]["confidence"], reverse=True)
            kept_negative_indices = negative_indices[:max_negatives_per_image]
            kept_indices = sorted(positive_indices | set(kept_negative_indices))

            for det_idx in kept_indices:
                det_box = detections[det_idx]
                label = "hold" if det_idx in positive_indices else "no_hold"
                legacy_label = "carry" if label == "hold" else "no_carry"
                crop_xmin, crop_ymin, crop_xmax, crop_ymax = expand_box(
                    det_box["xmin"],
                    det_box["ymin"],
                    det_box["xmax"],
                    det_box["ymax"],
                    width=width,
                    height=height,
                    padding_fraction=padding_fraction,
                )
                crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
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
                        "matched_weapon_count": matches_by_index.get(det_idx, 0),
                        "num_gt_weapon_boxes_in_image": len(weapon_boxes),
                    }
                )

        metadata_path = metadata_root / f"{split_name}_person_crops.csv"
        misses_path = metadata_root / f"{split_name}_stage0_misses.csv"
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
                "matched_weapon_count",
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
            ],
        )
        crop_df.to_csv(metadata_path, index=False)
        miss_df.to_csv(misses_path, index=False)

        summary_rows.append(
            {
                "split": split_name,
                "images_processed": counters["images_processed"],
                "images_with_weapon": counters["images_with_weapon"],
                "gt_weapon_boxes": counters["gt_weapon_boxes"],
                "person_detections": counters["person_detections"],
                "hold_crops": counters["hold_crops"],
                "no_hold_crops": counters["no_hold_crops"],
                "carry_crops": counters["hold_crops"],
                "no_carry_crops": counters["no_hold_crops"],
                "stage0_miss_images": counters["stage0_miss_images"],
                "stage0_missed_weapon_boxes": counters["stage0_missed_weapon_boxes"],
            }
        )

        print(
            f"[OK] {split_name}: "
            f"images={counters['images_processed']} "
            f"hold_crops={counters['hold_crops']} "
            f"no_hold_crops={counters['no_hold_crops']} "
            f"stage0_miss_images={counters['stage0_miss_images']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = metadata_root / "two_phase_dataset_summary.csv"
    summary_md_path = root / "docs" / "sprint4_two_phase_dataset_summary.md"
    summary_df.to_csv(summary_csv_path, index=False)

    with summary_md_path.open("w", encoding="utf-8") as f:
        f.write("# Sprint 4 - Two-Phase Dataset Summary\n\n")
        f.write("## Split summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Labeling rule\n\n")
        f.write("- `hold`: the center of at least one ground-truth `weapon` box falls inside a detected person box.\n")
        f.write("- `no_hold`: detected person box with no ground-truth `weapon` center inside it.\n")
        f.write("- `knife` is ignored exactly as in Sprint 3.\n")

    print(f"[OK] Saved: {summary_csv_path}")
    print(f"[OK] Saved: {summary_md_path}")


if __name__ == "__main__":
    main()
