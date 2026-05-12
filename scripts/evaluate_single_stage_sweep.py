from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_detection_pipeline import evaluate_predictions, generate_single_stage_predictions
from two_phase_utils import ensure_dir, filter_weapon_boxes, load_split_manifest, parse_voc_xml, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for a single-stage YOLO checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="YOLO checkpoint to evaluate.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Split to evaluate.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0, cuda:0, or cpu.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/single_stage/evaluation"), help="Output directory.")
    parser.add_argument("--output-prefix", default=None, help="Output filename prefix.")
    parser.add_argument("--predict-conf", type=float, default=0.01, help="Low confidence used to collect candidates.")
    parser.add_argument("--predict-iou", type=float, default=0.45, help="YOLO prediction NMS IoU.")
    parser.add_argument("--eval-iou", type=float, default=0.50, help="Evaluation IoU threshold.")
    parser.add_argument("--thresholds", default="0.01,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50", help="Comma-separated confidence thresholds.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--reuse-predictions", type=Path, default=None, help="Reuse a previously generated low-conf predictions CSV.")
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(round(float(item), 4))
    if not values:
        raise ValueError("At least one threshold is required.")
    return values


def display_path(path: Path) -> str:
    root = project_root()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def add_box_size_counts(manifest_df: pd.DataFrame, per_image_df: pd.DataFrame) -> dict[str, int]:
    root = project_root()
    size_totals = {"small_gt": 0, "medium_gt": 0, "large_gt": 0}
    image_to_sizes: dict[str, dict[str, int]] = {}

    for _, row in manifest_df.iterrows():
        _, _, objects = parse_voc_xml(root / Path(row["annotation_path"]))
        counts = {"small_gt": 0, "medium_gt": 0, "large_gt": 0}
        for box in filter_weapon_boxes(objects):
            area = max(0.0, float(box["xmax"]) - float(box["xmin"])) * max(0.0, float(box["ymax"]) - float(box["ymin"]))
            if area < 32 * 32:
                counts["small_gt"] += 1
            elif area < 96 * 96:
                counts["medium_gt"] += 1
            else:
                counts["large_gt"] += 1
        image_to_sizes[str(row["image_path"])] = counts
        for key, value in counts.items():
            size_totals[key] += value

    for key in ["small_gt", "medium_gt", "large_gt"]:
        per_image_df[key] = per_image_df["image_path"].map(lambda path: image_to_sizes.get(str(path), {}).get(key, 0))
    return size_totals


def main() -> None:
    args = parse_args()
    root = project_root()

    model_path = args.model if args.model.is_absolute() else root / args.model
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    ensure_dir(output_dir)

    manifest_df = load_split_manifest(root, args.split)
    if args.max_images is not None:
        manifest_df = manifest_df.head(args.max_images).copy()

    output_prefix = args.output_prefix or f"{model_path.parent.parent.name}_{args.split}"
    low_conf_predictions_path = output_dir / f"{output_prefix}_lowconf_predictions.csv"

    if args.reuse_predictions is not None:
        reuse_path = args.reuse_predictions if args.reuse_predictions.is_absolute() else root / args.reuse_predictions
        predictions_df = pd.read_csv(reuse_path)
    else:
        predictions_df = generate_single_stage_predictions(
            manifest_df=manifest_df,
            model_path=model_path,
            output_path=low_conf_predictions_path,
            device=args.device,
            conf_threshold=args.predict_conf,
            iou_threshold=args.predict_iou,
        )

    thresholds = parse_thresholds(args.thresholds)
    rows = []
    per_image_outputs = {}
    best_f1 = None
    best_recall = None

    for threshold in thresholds:
        filtered = predictions_df[predictions_df["weapon_confidence"] >= threshold].copy()
        metrics, per_image = evaluate_predictions(manifest_df, filtered, iou_threshold=args.eval_iou)
        metrics_row = {
            "threshold": threshold,
            **metrics,
            "prediction_candidates": len(filtered),
        }
        rows.append(metrics_row)
        per_image_outputs[f"{threshold:.2f}"] = per_image

        if best_f1 is None or metrics["f1"] > best_f1["f1"]:
            best_f1 = metrics_row
        if best_recall is None or metrics["recall"] > best_recall["recall"]:
            best_recall = metrics_row

    sweep_df = pd.DataFrame(rows)
    sweep_path = output_dir / f"{output_prefix}_threshold_sweep.csv"
    summary_path = output_dir / f"{output_prefix}_threshold_sweep.md"
    sweep_df.to_csv(sweep_path, index=False)

    best_f1_threshold = f"{best_f1['threshold']:.2f}"
    best_per_image = per_image_outputs[best_f1_threshold].copy()
    size_totals = add_box_size_counts(manifest_df, best_per_image)
    best_per_image_path = output_dir / f"{output_prefix}_best_f1_per_image.csv"
    best_per_image.to_csv(best_per_image_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Single-Stage Threshold Sweep\n\n")
        f.write(f"- Model: `{display_path(model_path)}`\n")
        f.write(f"- Split: `{args.split}`\n")
        f.write(f"- Prediction confidence floor: `{args.predict_conf}`\n")
        f.write(f"- Prediction NMS IoU: `{args.predict_iou}`\n")
        f.write(f"- Evaluation IoU: `{args.eval_iou}`\n\n")
        f.write("## Best by F1\n\n")
        f.write(pd.DataFrame([best_f1]).to_markdown(index=False))
        f.write("\n\n## Best by Recall\n\n")
        f.write(pd.DataFrame([best_recall]).to_markdown(index=False))
        f.write("\n\n## Ground-Truth Box Sizes\n\n")
        f.write(pd.DataFrame([size_totals]).to_markdown(index=False))
        f.write("\n\n## Full Sweep\n\n")
        f.write(sweep_df.to_markdown(index=False))
        f.write("\n")

    print(f"[OK] Saved low-conf predictions: {low_conf_predictions_path}")
    print(f"[OK] Saved threshold sweep: {sweep_path}")
    print(f"[OK] Saved best-F1 per-image breakdown: {best_per_image_path}")
    print(f"[OK] Saved summary: {summary_path}")
    print("\nBest by F1:")
    print(pd.DataFrame([best_f1]).to_string(index=False))
    print("\nBest by recall:")
    print(pd.DataFrame([best_recall]).to_string(index=False))


if __name__ == "__main__":
    main()
