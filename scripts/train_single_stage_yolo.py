from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-stage YOLO weapon detector.")
    parser.add_argument("--model", default="yolo26n.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("configs/yolo_data.yaml"), help="YOLO dataset YAML.")
    parser.add_argument("--name", default=None, help="Run name under the project directory.")
    parser.add_argument("--project", type=Path, default=Path("runs/single_stage"), help="Ultralytics project directory.")
    parser.add_argument("--device", default="0", help="Training device, for example 0, cuda:0, or cpu.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size. Lower this for larger image sizes.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer setting.")
    parser.add_argument("--lr0", type=float, default=None, help="Optional initial learning rate override.")
    parser.add_argument("--lrf", type=float, default=None, help="Optional final learning-rate fraction override.")
    parser.add_argument("--close-mosaic", type=int, default=15, help="Disable mosaic in the last N epochs.")
    parser.add_argument("--cache", action="store_true", help="Cache images if memory allows.")
    parser.add_argument("--rect", action="store_true", help="Use rectangular training batches.")
    parser.add_argument("--multi-scale", action="store_true", help="Enable YOLO multi-scale training.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the provided model/checkpoint.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse the requested run directory name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT

    data_yaml = args.data if args.data.is_absolute() else root / args.data
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {data_yaml}")

    model = YOLO(args.model)
    run_name = args.name or f"{Path(args.model).stem}_img{args.imgsz}"

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(root / args.project if not args.project.is_absolute() else args.project),
        "name": run_name,
        "pretrained": not args.resume,
        "workers": args.workers,
        "device": args.device,
        "cache": args.cache,
        "verbose": True,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "close_mosaic": args.close_mosaic,
        "rect": args.rect,
        "multi_scale": args.multi_scale,
        "cos_lr": args.cos_lr,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "plots": True,
        "val": True,
    }
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
