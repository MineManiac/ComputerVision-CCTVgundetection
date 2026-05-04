from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from two_phase_utils import (
    CarryClassifierNet,
    build_classifier_transform,
    ensure_dir,
    load_two_phase_config,
    normalize_torch_device,
    project_root,
    resolve_path,
    set_random_seed,
)


class CarryCropDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Any, max_items: int | None = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not root_dir.exists():
            raise FileNotFoundError(f"Missing crop directory: {root_dir}")

        label_dirs = [("carry", 1), ("no_carry", 0)]
        for label_name, label_id in label_dirs:
            label_dir = root_dir / label_name
            if not label_dir.exists():
                continue
            for image_path in sorted(label_dir.glob("*.jpg")):
                self.samples.append((image_path, label_id))

        if max_items is not None:
            self.samples = self.samples[:max_items]

        if not self.samples:
            raise ValueError(f"No crop images found under: {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.tensor(float(label), dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sprint 4 carry/no_carry classifier.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/two_phase.yaml.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override crop root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output run directory.")
    parser.add_argument("--device", default=None, help="Training device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--max-train-items", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-val-items", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-test-items", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)


def run_epoch(
    model: CarryClassifierNet,
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

    all_targets_tensor = torch.cat(all_targets)
    all_probs_tensor = torch.cat(all_probs)
    average_loss = total_loss / len(dataloader.dataset)
    return average_loss, all_targets_tensor, all_probs_tensor


def metrics_at_threshold(targets: torch.Tensor, probs: torch.Tensor, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).int()
    targets_int = targets.int()
    tp = int(((preds == 1) & (targets_int == 1)).sum().item())
    fp = int(((preds == 1) & (targets_int == 0)).sum().item())
    fn = int(((preds == 0) & (targets_int == 1)).sum().item())
    tn = int(((preds == 0) & (targets_int == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def find_best_threshold(targets: torch.Tensor, probs: torch.Tensor) -> dict[str, float]:
    candidate_thresholds = [round(step / 100.0, 2) for step in range(5, 96, 5)]
    best_metrics: dict[str, float] | None = None
    for threshold in candidate_thresholds:
        metrics = metrics_at_threshold(targets, probs, threshold)
        if best_metrics is None:
            best_metrics = metrics
            continue
        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            continue
        if metrics["f1"] == best_metrics["f1"] and metrics["recall"] > best_metrics["recall"]:
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("Threshold search produced no metrics.")
    return best_metrics


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_two_phase_config(args.config)
    seed = int(config["training"]["seed"])
    set_random_seed(seed)

    data_root = args.data_root or resolve_path(root, config["paths"]["two_phase_root"]) / "crops"
    output_dir = args.output_dir or resolve_path(root, config["paths"]["carry_runs_dir"])
    ensure_dir(output_dir)

    device = normalize_torch_device(args.device or config["inference"]["default_device"])
    epochs = args.epochs or int(config["training"]["epochs"])
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(config["training"]["num_workers"])
    image_size = int(config["dataset"]["classifier_image_size"])
    transform = build_classifier_transform(image_size)

    train_dataset = CarryCropDataset(data_root / "train", transform=transform, max_items=args.max_train_items)
    val_dataset = CarryCropDataset(data_root / "val", transform=transform, max_items=args.max_val_items)
    test_dataset = CarryCropDataset(data_root / "test", transform=transform, max_items=args.max_test_items)

    train_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_positive = sum(label for _, label in train_dataset.samples)
    train_negative = len(train_dataset.samples) - train_positive
    if train_positive == 0:
        raise ValueError("Training set has no positive carry samples.")

    pos_weight_value = max(1.0, float(train_negative) / float(train_positive))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    model = CarryClassifierNet(image_size=image_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_state: dict[str, Any] | None = None
    best_val_f1 = -1.0
    best_threshold = 0.5
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_targets, val_probs = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        best_val_metrics = find_best_threshold(val_targets, val_probs)

        history_rows.append(
            {
                "kind": "epoch",
                "epoch": epoch,
                "split": "val",
                "loss": round(val_loss, 6),
                "train_loss": round(train_loss, 6),
                "threshold": best_val_metrics["threshold"],
                "precision": round(best_val_metrics["precision"], 6),
                "recall": round(best_val_metrics["recall"], 6),
                "f1": round(best_val_metrics["f1"], 6),
                "accuracy": round(best_val_metrics["accuracy"], 6),
                "tp": best_val_metrics["tp"],
                "fp": best_val_metrics["fp"],
                "fn": best_val_metrics["fn"],
                "tn": best_val_metrics["tn"],
            }
        )

        if best_val_metrics["f1"] > best_val_f1:
            best_val_f1 = best_val_metrics["f1"]
            best_threshold = float(best_val_metrics["threshold"])
            best_state = {
                "model_state_dict": model.state_dict(),
                "image_size": image_size,
                "best_threshold": best_threshold,
                "epoch": epoch,
                "pos_weight": pos_weight_value,
            }

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={best_val_metrics['f1']:.4f} threshold={best_val_metrics['threshold']:.2f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished without selecting a best checkpoint.")

    checkpoint_path = output_dir / "best.pt"
    torch.save(best_state, checkpoint_path)

    model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_targets, test_probs = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
    test_metrics = metrics_at_threshold(test_targets, test_probs, best_threshold)
    history_rows.append(
        {
            "kind": "final",
            "epoch": int(best_state["epoch"]),
            "split": "test",
            "loss": round(test_loss, 6),
            "train_loss": "",
            "threshold": best_threshold,
            "precision": round(test_metrics["precision"], 6),
            "recall": round(test_metrics["recall"], 6),
            "f1": round(test_metrics["f1"], 6),
            "accuracy": round(test_metrics["accuracy"], 6),
            "tp": test_metrics["tp"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
            "tn": test_metrics["tn"],
        }
    )

    metrics_df = pd.DataFrame(history_rows)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.md"
    metrics_df.to_csv(metrics_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Carry Classifier Training Summary\n\n")
        f.write(f"- Train samples: `{len(train_dataset)}`\n")
        f.write(f"- Validation samples: `{len(val_dataset)}`\n")
        f.write(f"- Test samples: `{len(test_dataset)}`\n")
        f.write(f"- Positive weight: `{pos_weight_value:.4f}`\n")
        f.write(f"- Best validation F1: `{best_val_f1:.4f}`\n")
        f.write(f"- Frozen inference threshold: `{best_threshold:.2f}`\n\n")
        f.write("## Metrics\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n")

    print(f"[OK] Saved checkpoint: {checkpoint_path}")
    print(f"[OK] Saved metrics: {metrics_path}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
