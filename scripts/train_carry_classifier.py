from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from two_phase_utils import (
    build_carry_classifier,
    build_classifier_transform,
    ensure_dir,
    load_two_phase_config,
    normalize_torch_device,
    project_root,
    resolve_path,
    set_random_seed,
    threshold_range,
    zoom_lower_fraction,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_train_transform(image_size: int) -> transforms.Compose:
    """
    Augmented transform for training.

    Extra augmentations beyond the baseline (resize + normalize):
      - RandomHorizontalFlip: weapons can be carried on either side.
      - ColorJitter: compensates for the variable lighting across cameras
        (Cam5 has irregular lighting per paper Section 3.1.1).
      - RandomRotation(10): slight camera angle variation in CCTV frames.
      - RandomAffine: small translation/scale to improve generalisation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class HoldCropDataset(Dataset):
    """
    Dataset that loads hold / no_hold person-crop images.

    Supports:
      - Legacy label dirs: hold / no_hold  or  carry / no_carry
      - Optional zoom-crop: keeps only the lower `zoom_lower_fraction`
        of each image before the transform (increases weapon pixel size by ~80%).
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Any,
        zoom_lower_fraction: float | None = None,
        max_items: int | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.zoom_lower_fraction = zoom_lower_fraction
        self.samples: list[tuple[Path, int]] = []

        if not root_dir.exists():
            raise FileNotFoundError(f"Missing crop directory: {root_dir}")

        if (root_dir / "hold").exists() or (root_dir / "no_hold").exists():
            label_dir_candidates = [("hold", 1), ("no_hold", 0)]
        else:
            label_dir_candidates = [("carry", 1), ("no_carry", 0)]

        for label_name, label_id in label_dir_candidates:
            label_dir = root_dir / label_name
            if not label_dir.exists():
                continue
            for image_path in sorted(label_dir.glob("*.jpg")):
                self.samples.append((image_path, label_id))

        if max_items is not None:
            self.samples = self.samples[:max_items]

        if not self.samples:
            raise ValueError(f"No hold/no_hold crop images found under: {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as img:
            img = img.convert("RGB")

        if self.zoom_lower_fraction is not None:
            img = zoom_lower_fraction(img, self.zoom_lower_fraction)

        tensor = self.transform(img)
        return tensor, torch.tensor(float(label), dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def build_weighted_sampler(dataset: HoldCropDataset) -> WeightedRandomSampler:
    """
    Oversample the minority class so each epoch sees a balanced distribution.

    This prevents the classifier from learning "predict no_hold always" when
    hold samples are rare (which they often are in CCTV datasets).
    """
    labels = [label for _, label in dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        # Cannot balance a single-class dataset; fall back to uniform sampling
        weights = [1.0] * len(labels)
    else:
        weight_pos = 1.0 / n_pos
        weight_neg = 1.0 / n_neg
        weights = [weight_pos if lbl == 1 else weight_neg for lbl in labels]
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
    return sampler


def build_loader(
    dataset: HoldCropDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    sampler = build_weighted_sampler(dataset) if use_weighted_sampler else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu()
        all_targets.append(targets.detach().cpu())
        all_probs.append(probs)
        total_loss += float(loss.item()) * len(images)

    all_targets_t = torch.cat(all_targets)
    all_probs_t = torch.cat(all_probs)
    avg_loss = total_loss / max(1, len(dataloader.dataset))
    return avg_loss, all_targets_t, all_probs_t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_at_threshold(
    targets: torch.Tensor, probs: torch.Tensor, threshold: float
) -> dict[str, float]:
    preds = (probs >= threshold).int()
    targets_int = targets.int()
    tp = int(((preds == 1) & (targets_int == 1)).sum())
    fp = int(((preds == 1) & (targets_int == 0)).sum())
    fn = int(((preds == 0) & (targets_int == 1)).sum())
    tn = int(((preds == 0) & (targets_int == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def sweep_thresholds(
    targets: torch.Tensor, probs: torch.Tensor
) -> list[dict[str, float]]:
    return [metrics_at_threshold(targets, probs, t) for t in threshold_range()]


def select_best_f1_threshold(sweep_rows: list[dict[str, float]]) -> dict[str, float]:
    best: dict[str, float] | None = None
    for row in sweep_rows:
        if best is None or row["f1"] > best["f1"] or (
            row["f1"] == best["f1"] and row["recall"] > best["recall"]
        ):
            best = row
    if best is None:
        raise RuntimeError("Empty threshold sweep.")
    return best


def select_stage1_gate_threshold(
    sweep_rows: list[dict[str, float]], recall_floor: float
) -> dict[str, float]:
    eligible = [r for r in sweep_rows if r["recall"] >= recall_floor]
    if eligible:
        return max(eligible, key=lambda r: (r["f1"], r["threshold"]))
    return max(sweep_rows, key=lambda r: (r["recall"], r["f1"], r["threshold"]))


def make_threshold_sweep_rows(
    split_name: str,
    epoch: int,
    sweep_rows: list[dict[str, float]],
    best_f1_threshold: float,
    gate_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in sweep_rows:
        thr = float(row["threshold"])
        role = ""
        if abs(thr - best_f1_threshold) < 1e-9 and abs(thr - gate_threshold) < 1e-9:
            role = "best_f1_and_gate"
        elif abs(thr - best_f1_threshold) < 1e-9:
            role = "best_f1"
        elif abs(thr - gate_threshold) < 1e-9:
            role = "stage1_gate"
        rows.append({
            "kind": "threshold_sweep",
            "epoch": epoch,
            "split": split_name,
            "threshold": thr,
            "threshold_role": role,
            "precision": round(float(row["precision"]), 6),
            "recall": round(float(row["recall"]), 6),
            "f1": round(float(row["f1"]), 6),
            "accuracy": round(float(row["accuracy"]), 6),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "tn": int(row["tn"]),
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the carry/no-carry classifier.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--backbone", default=None,
                        help="Classifier backbone: 'mobilenet_v3_small' or 'custom_cnn'.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Disable ImageNet pre-trained weights.")
    parser.add_argument("--no-zoom", action="store_true",
                        help="Disable zoom-crop even if set in config.")
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_two_phase_config(args.config)
    seed = int(config["training"]["seed"])
    recall_floor = float(config["training"]["threshold_recall_floor"])
    set_random_seed(seed)

    data_root = args.data_root or resolve_path(root, config["paths"]["two_phase_root"]) / "crops"
    output_dir = args.output_dir or resolve_path(root, config["paths"]["carry_runs_dir"])
    ensure_dir(output_dir)

    device = normalize_torch_device(args.device or config["inference"]["default_device"])
    epochs = args.epochs or int(config["training"]["epochs"])
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    num_workers = (
        args.num_workers if args.num_workers is not None
        else int(config["training"]["num_workers"])
    )
    image_size = int(config["dataset"]["classifier_image_size"])
    backbone = args.backbone or config["training"].get("classifier_backbone", "mobilenet_v3_small")
    pretrained = not args.no_pretrained
    backbone_freeze_epochs = int(config["training"].get("backbone_freeze_epochs", 5))
    backbone_lr_factor = float(config["training"].get("backbone_lr_factor", 0.1))
    base_lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    # Zoom-crop: use lower fraction of person crop for classifier input
    zoom_fraction: float | None = None
    if not args.no_zoom:
        raw_zoom = config["dataset"].get("classifier_zoom_lower_fraction", None)
        if raw_zoom is not None:
            zoom_fraction = float(raw_zoom)

    print(f"[Config] backbone={backbone} pretrained={pretrained} epochs={epochs}")
    print(f"[Config] image_size={image_size} batch_size={batch_size}")
    print(f"[Config] zoom_lower_fraction={zoom_fraction} recall_floor={recall_floor}")
    print(f"[Config] backbone_freeze_epochs={backbone_freeze_epochs} backbone_lr_factor={backbone_lr_factor}")

    train_transform = _build_train_transform(image_size)
    eval_transform = build_classifier_transform(image_size)

    train_dataset = HoldCropDataset(
        data_root / "train", transform=train_transform,
        zoom_lower_fraction=zoom_fraction, max_items=args.max_train_items,
    )
    val_dataset = HoldCropDataset(
        data_root / "val", transform=eval_transform,
        zoom_lower_fraction=zoom_fraction, max_items=args.max_val_items,
    )
    test_dataset = HoldCropDataset(
        data_root / "test", transform=eval_transform,
        zoom_lower_fraction=zoom_fraction, max_items=args.max_test_items,
    )

    train_positive = sum(lbl for _, lbl in train_dataset.samples)
    train_negative = len(train_dataset.samples) - train_positive
    if train_positive == 0:
        raise ValueError("Training set has no positive hold samples.")

    print(f"[Data] train pos={train_positive} neg={train_negative} | "
          f"val={len(val_dataset)} | test={len(test_dataset)}")

    # Use WeightedRandomSampler to balance classes during training
    train_loader = build_loader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, use_weighted_sampler=True,
    )
    val_loader = build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # pos_weight for BCEWithLogitsLoss as secondary class-balancing signal
    pos_weight_value = max(1.0, float(train_negative) / float(train_positive))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )

    model = build_carry_classifier(backbone=backbone, image_size=image_size, pretrained=pretrained)
    model = model.to(device)

    # Two-phase optimizer: head-only first, then backbone + head
    is_mobilenet = hasattr(model, "freeze_backbone")
    if is_mobilenet and backbone_freeze_epochs > 0:
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            model.head_parameters(), lr=base_lr, weight_decay=weight_decay
        )
        print(f"[Train] Phase 1: backbone frozen for {backbone_freeze_epochs} epochs.")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

    # Cosine LR scheduler over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_state: dict[str, Any] | None = None
    best_gate_f1 = -1.0
    best_gate_recall = -1.0
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after backbone_freeze_epochs → switch to full optimizer
        if is_mobilenet and epoch == backbone_freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone_parameters(), "lr": base_lr * backbone_lr_factor},
                {"params": model.head_parameters(), "lr": base_lr},
            ], weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - backbone_freeze_epochs, eta_min=1e-6
            )
            print(f"[Train] Phase 2 (epoch {epoch}): backbone unfrozen at lr×{backbone_lr_factor}.")

        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_targets, val_probs = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        sweep_rows = sweep_thresholds(val_targets, val_probs)
        best_f1_metrics = select_best_f1_threshold(sweep_rows)
        gate_metrics = select_stage1_gate_threshold(sweep_rows, recall_floor=recall_floor)

        history_rows.append({
            "kind": "epoch_summary",
            "epoch": epoch,
            "split": "val",
            "loss": round(val_loss, 6),
            "train_loss": round(train_loss, 6),
            "best_f1_threshold": round(float(best_f1_metrics["threshold"]), 6),
            "best_f1_precision": round(float(best_f1_metrics["precision"]), 6),
            "best_f1_recall": round(float(best_f1_metrics["recall"]), 6),
            "best_f1_f1": round(float(best_f1_metrics["f1"]), 6),
            "stage1_gate_threshold": round(float(gate_metrics["threshold"]), 6),
            "stage1_gate_precision": round(float(gate_metrics["precision"]), 6),
            "stage1_gate_recall": round(float(gate_metrics["recall"]), 6),
            "stage1_gate_f1": round(float(gate_metrics["f1"]), 6),
            "threshold_policy": "recall_floor",
            "threshold_recall_floor": recall_floor,
        })

        gate_f1 = float(gate_metrics["f1"])
        gate_recall = float(gate_metrics["recall"])
        if gate_f1 > best_gate_f1 or (gate_f1 == best_gate_f1 and gate_recall > best_gate_recall):
            best_gate_f1 = gate_f1
            best_gate_recall = gate_recall
            best_state = {
                "model_state_dict": model.state_dict(),
                "image_size": image_size,
                "backbone": backbone,
                "zoom_lower_fraction": zoom_fraction,
                "best_threshold": float(gate_metrics["threshold"]),
                "best_f1_threshold": float(best_f1_metrics["threshold"]),
                "stage1_gate_threshold": float(gate_metrics["threshold"]),
                "best_f1_metrics": {k: float(v) for k, v in best_f1_metrics.items()},
                "stage1_gate_metrics": {k: float(v) for k, v in gate_metrics.items()},
                "epoch": epoch,
                "pos_weight": pos_weight_value,
                "threshold_policy": "recall_floor",
                "threshold_recall_floor": recall_floor,
            }

        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"gate_thr={gate_metrics['threshold']:.2f} "
            f"gate_recall={gate_metrics['recall']:.3f} gate_f1={gate_metrics['f1']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    checkpoint_path = output_dir / "best.pt"
    torch.save(best_state, checkpoint_path)

    # Final evaluation on the best checkpoint
    model.load_state_dict(best_state["model_state_dict"])
    val_loss, val_targets, val_probs = run_epoch(model, val_loader, criterion, None, device)
    val_sweep = sweep_thresholds(val_targets, val_probs)
    best_f1_metrics = select_best_f1_threshold(val_sweep)
    gate_metrics = select_stage1_gate_threshold(val_sweep, recall_floor=recall_floor)
    history_rows.extend(
        make_threshold_sweep_rows(
            split_name="val",
            epoch=int(best_state["epoch"]),
            sweep_rows=val_sweep,
            best_f1_threshold=float(best_f1_metrics["threshold"]),
            gate_threshold=float(gate_metrics["threshold"]),
        )
    )

    test_loss, test_targets, test_probs = run_epoch(model, test_loader, criterion, None, device)
    test_metrics = metrics_at_threshold(test_targets, test_probs, float(gate_metrics["threshold"]))
    history_rows.append({
        "kind": "final_test",
        "epoch": int(best_state["epoch"]),
        "split": "test",
        "loss": round(test_loss, 6),
        "train_loss": "",
        "best_f1_threshold": round(float(best_f1_metrics["threshold"]), 6),
        "stage1_gate_threshold": round(float(gate_metrics["threshold"]), 6),
        "threshold_policy": "recall_floor",
        "threshold_recall_floor": recall_floor,
        "precision": round(float(test_metrics["precision"]), 6),
        "recall": round(float(test_metrics["recall"]), 6),
        "f1": round(float(test_metrics["f1"]), 6),
        "accuracy": round(float(test_metrics["accuracy"]), 6),
        "tp": int(test_metrics["tp"]),
        "fp": int(test_metrics["fp"]),
        "fn": int(test_metrics["fn"]),
        "tn": int(test_metrics["tn"]),
    })

    metrics_df = pd.DataFrame(history_rows)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.md"
    metrics_df.to_csv(metrics_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Carry Classifier Training Summary\n\n")
        f.write(f"- Backbone: `{backbone}`\n")
        f.write(f"- Pretrained: `{pretrained}`\n")
        f.write(f"- Zoom lower fraction: `{zoom_fraction}`\n")
        f.write(f"- Train samples: `{len(train_dataset)}`  (pos={train_positive} neg={train_negative})\n")
        f.write(f"- Validation samples: `{len(val_dataset)}`\n")
        f.write(f"- Test samples: `{len(test_dataset)}`\n")
        f.write(f"- Positive weight: `{pos_weight_value:.4f}`\n")
        f.write(f"- Best checkpoint epoch: `{int(best_state['epoch'])}`\n")
        f.write(f"- Stage 1 gate threshold: `{float(gate_metrics['threshold']):.2f}`\n")
        f.write(f"- Recall floor: `{recall_floor:.2f}`\n\n")
        f.write("## Final test metrics (gate threshold)\n\n")
        f.write(f"- Precision: `{test_metrics['precision']:.4f}`\n")
        f.write(f"- Recall: `{test_metrics['recall']:.4f}`\n")
        f.write(f"- F1: `{test_metrics['f1']:.4f}`\n")
        f.write(f"- TP: `{test_metrics['tp']}` FP: `{test_metrics['fp']}` FN: `{test_metrics['fn']}`\n\n")
        sweep_df = metrics_df[metrics_df["kind"] == "threshold_sweep"].copy()
        if not sweep_df.empty:
            f.write("## Validation threshold sweep\n\n")
            f.write(sweep_df.to_markdown(index=False))
            f.write("\n")

    print(f"[OK] Checkpoint: {checkpoint_path}")
    print(f"[OK] Metrics:    {metrics_path}")
    print(f"[OK] Summary:    {summary_path}")
    print(f"[OK] Test → precision={test_metrics['precision']:.4f} "
          f"recall={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
