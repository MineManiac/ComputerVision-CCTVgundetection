"""
threshold_sweep.py — Varre diferentes valores de weapon_conf nas predições existentes.

Uso:
    python scripts/threshold_sweep.py \
        --predictions runs/two_phase/predictions/test_sweep_predictions_pre_nms.csv \
        --thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50

O arquivo de predições deve ter sido gerado com um weapon_conf baixo (ex: 0.05)
para capturar todas as detecções candidatas. O script então filtra cada threshold
e avalia sem precisar re-rodar o pipeline.

Output:
    runs/two_phase/evaluation/threshold_sweep.csv
    runs/two_phase/evaluation/threshold_sweep.md
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_detection_pipeline import (
    evaluate_predictions,
    resolve_manifest,
    group_prediction_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold sweep over existing weapon predictions.")
    p.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "runs/two_phase/predictions/test_sweep_predictions_pre_nms.csv",
        help="CSV com todas as detecções candidatas (gerado com weapon_conf baixo, ex 0.05).",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split a avaliar.",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
        help="Lista de thresholds a testar.",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold para matching GT (default 0.50).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/two_phase.yaml",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/two_phase/evaluation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Load manifest ──────────────────────────────────────────────────────────
    manifest_path = PROJECT_ROOT / config["paths"]["splits_dir"] / f"{args.split}_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    print(f"[INFO] Manifest: {len(manifest_df)} images ({args.split})")

    # ── Load all candidate predictions ────────────────────────────────────────
    if not args.predictions.exists():
        print(f"[ERROR] Predictions file not found: {args.predictions}")
        print()
        print("  Execute primeiro a inferência com weapon_conf=0.05 no configs/two_phase.yaml:")
        print("    python scripts/run_two_phase_inference.py --split test --output-prefix test_sweep_")
        print()
        sys.exit(1)

    df_all = pd.read_csv(args.predictions)
    # Support both pre_nms CSV (has kept_after_image_nms) and final predictions CSV
    if "kept_after_image_nms" in df_all.columns:
        # Use only kept predictions for fair comparison at each threshold
        # OR use all (pre-NMS) for a true threshold sweep
        # We use all pre-NMS predictions then apply per-image NMS ourselves — but
        # for simplicity and comparability, use kept_after_image_nms=1 detections
        # filtered by confidence threshold (same as production pipeline minus NMS).
        df_all = df_all[df_all["kept_after_image_nms"] == 1].copy()
        print(f"[INFO] Using post-NMS subset: {len(df_all)} detections")
    else:
        print(f"[INFO] Loaded {len(df_all)} candidate detections")

    print(f"[INFO] Confidence range: [{df_all['weapon_confidence'].min():.3f}, {df_all['weapon_confidence'].max():.3f}]")
    print(f"[INFO] Sweeping {len(args.thresholds)} thresholds: {args.thresholds}")
    print()

    # ── Baseline: single-stage ─────────────────────────────────────────────────
    ss_pred_path = PROJECT_ROOT / "runs/two_phase/evaluation/test_thr050_single_stage_predictions.csv"
    if ss_pred_path.exists():
        df_ss = pd.read_csv(ss_pred_path)
        ss_metrics, _ = evaluate_predictions(manifest_df, df_ss, iou_threshold=args.iou_threshold)
        print(f"[Baseline] Single-Stage — P={ss_metrics['precision']:.3f} "
              f"R={ss_metrics['recall']:.3f} F1={ss_metrics['f1']:.3f} "
              f"TP={ss_metrics['tp']} FP={ss_metrics['fp']} FN={ss_metrics['fn']}")
    else:
        ss_metrics = None
        print("[Baseline] Single-stage predictions not found, skipping baseline.")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    rows = []
    best_f1 = -1
    best_row = None

    for thr in sorted(args.thresholds):
        df_thr = df_all[df_all["weapon_confidence"] >= thr].copy()
        if len(df_thr) == 0:
            print(f"  thr={thr:.2f} — no detections, skipping")
            continue

        metrics, _ = evaluate_predictions(manifest_df, df_thr, iou_threshold=args.iou_threshold)
        row = {
            "threshold": thr,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "detections": len(df_thr),
            "det_per_image": round(metrics["detections_per_image"], 3),
        }
        rows.append(row)
        star = " ← best" if metrics["f1"] > best_f1 else ""
        print(f"  thr={thr:.2f} | P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} | TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
              f" | det={len(df_thr)}{star}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_row = row

    if not rows:
        print("[ERROR] No results — check predictions file.")
        sys.exit(1)

    df_results = pd.DataFrame(rows)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_out = args.output_dir / "threshold_sweep.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved: {csv_out}")

    # ── Save Markdown ──────────────────────────────────────────────────────────
    md_out = args.output_dir / "threshold_sweep.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Threshold Sweep — weapon_conf\n\n")
        f.write(f"**Split:** {args.split} | **IoU threshold:** {args.iou_threshold:.2f}\n\n")
        f.write(f"Predições candidatas geradas com `weapon_conf=0.05` (captura todos os candidatos).\n\n")

        if ss_metrics:
            f.write("## Baseline Single-Stage\n\n")
            f.write(f"| Precision | Recall | F1 | TP | FP | FN |\n")
            f.write(f"|---|---|---|---|---|---|\n")
            f.write(f"| {ss_metrics['precision']:.3f} | {ss_metrics['recall']:.3f} | "
                    f"{ss_metrics['f1']:.3f} | {ss_metrics['tp']} | {ss_metrics['fp']} | {ss_metrics['fn']} |\n\n")

        f.write("## Two-Phase — Sweep de Threshold\n\n")
        f.write("| Threshold | TP | FP | FN | Precision | Recall | F1 | Det/img |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            star = " ⭐" if row["threshold"] == best_row["threshold"] else ""
            f.write(f"| **{row['threshold']:.2f}**{star} | {row['tp']} | {row['fp']} | {row['fn']} | "
                    f"{row['precision']:.3f} | {row['recall']:.3f} | **{row['f1']:.4f}** | "
                    f"{row['det_per_image']:.2f} |\n")

        f.write(f"\n**Melhor threshold:** {best_row['threshold']:.2f} → F1={best_row['f1']:.4f} "
                f"(P={best_row['precision']:.3f}, R={best_row['recall']:.3f})\n\n")

        if ss_metrics:
            delta = (best_row["f1"] - ss_metrics["f1"]) / ss_metrics["f1"] * 100
            f.write(f"**Delta vs single-stage baseline (F1):** {delta:+.1f}%\n")

    print(f"[OK] Saved: {md_out}")
    print(f"\n=== MELHOR THRESHOLD: {best_row['threshold']:.2f} ===")
    print(f"    F1={best_row['f1']:.4f} | P={best_row['precision']:.3f} | R={best_row['recall']:.3f}")
    print(f"    TP={best_row['tp']} | FP={best_row['fp']} | FN={best_row['fn']}")


if __name__ == "__main__":
    main()
