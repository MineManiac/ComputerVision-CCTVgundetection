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
    threshold_range,
)


class HoldCropDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Any, max_items: int | None = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
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
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.tensor(float(label), dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sprint 4 hold/no_hold classifier.")
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


def sweep_thresholds(targets: torch.Tensor, probs: torch.Tensor) -> list[dict[str, float]]:
    return [metrics_at_threshold(targets, probs, threshold) for threshold in threshold_range()]


def select_best_f1_threshold(sweep_rows: list[dict[str, float]]) -> dict[str, float]:
    best_row: dict[str, float] | None = None
    for row in sweep_rows:
        if best_row is None:
            best_row = row
            continue
        if row["f1"] > best_row["f1"]:
            best_row = row
            continue
        if row["f1"] == best_row["f1"] and row["recall"] > best_row["recall"]:
            best_row = row
    if best_row is None:
        raise RuntimeError("Threshold sweep produced no rows for best-F1 selection.")
    return best_row


def select_stage1_gate_threshold(
    sweep_rows: list[dict[str, float]],
    recall_floor: float,
) -> dict[str, float]:
    eligible = [row for row in sweep_rows if row["recall"] >= recall_floor]
    if eligible:
        return max(eligible, key=lambda row: (row["f1"], row["threshold"]))

    fallback = max(sweep_rows, key=lambda row: (row["recall"], row["f1"], row["threshold"]))
    return fallback


def make_threshold_sweep_rows(
    split_name: str,
    epoch: int,
    sweep_rows: list[dict[str, float]],
    best_f1_threshold: float,
    gate_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in sweep_rows:
        threshold = float(row["threshold"])
        threshold_role = ""
        if abs(threshold - best_f1_threshold) < 1e-9 and abs(threshold - gate_threshold) < 1e-9:
            threshold_role = "best_f1_and_gate"
        elif abs(threshold - best_f1_threshold) < 1e-9:
            threshold_role = "best_f1"
        elif abs(threshold - gate_threshold) < 1e-9:
            threshold_role = "stage1_gate"

        rows.append(
            {
                "kind": "threshold_sweep",
                "epoch": epoch,
                "split": split_name,
                "threshold": threshold,
                "threshold_role": threshold_role,
                "precision": round(float(row["precision"]), 6),
                "recall": round(float(row["recall"]), 6),
                "f1": round(float(row["f1"]), 6),
                "accuracy": round(float(row["accuracy"]), 6),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tn": int(row["tn"]),
            }
        )
    return rows


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
    num_workers = args.num_workers if args.num_workers is not None else int(config["training"]["num_workers"])
    image_size = int(config["dataset"]["classifier_image_size"])
    transform = build_classifier_transform(image_size)

    train_dataset = HoldCropDataset(data_root / "train", transform=transform, max_items=args.max_train_items)
    val_dataset = HoldCropDataset(data_root / "val", transform=transform, max_items=args.max_val_items)
    test_dataset = HoldCropDataset(data_root / "test", transform=transform, max_items=args.max_test_items)

    train_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_positive = sum(label for _, label in train_dataset.samples)
    train_negative = len(train_dataset.samples) - train_positive
    if train_positive == 0:
        raise ValueError("Training set has no positive hold samples.")

    pos_weight_value = max(1.0, float(train_negative) / float(train_positive))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    model = CarryClassifierNet(image_size=image_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_state: dict[str, Any] | None = None
    best_gate_f1 = -1.0
    best_gate_recall = -1.0
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_targets, val_probs = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        sweep_rows = sweep_thresholds(val_targets, val_probs)
        best_f1_metrics = select_best_f1_threshold(sweep_rows)
        gate_metrics = select_stage1_gate_threshold(sweep_rows, recall_floor=recall_floor)

        history_rows.append(
            {
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
            }
        )

        gate_f1 = float(gate_metrics["f1"])
        gate_recall = float(gate_metrics["recall"])
        if gate_f1 > best_gate_f1 or (gate_f1 == best_gate_f1 and gate_recall > best_gate_recall):
            best_gate_f1 = gate_f1
            best_gate_recall = gate_recall
            best_state = {
                "model_state_dict": model.state_dict(),
                "image_size": image_size,
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
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best_f1_thr={best_f1_metrics['threshold']:.2f} best_f1={best_f1_metrics['f1']:.4f} "
            f"gate_thr={gate_metrics['threshold']:.2f} gate_recall={gate_metrics['recall']:.4f} gate_f1={gate_metrics['f1']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished without selecting a best checkpoint.")

    checkpoint_path = output_dir / "best.pt"
    torch.save(best_state, checkpoint_path)

    model.load_state_dict(best_state["model_state_dict"])
    val_loss, val_targets, val_probs = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
    val_sweep_rows = sweep_thresholds(val_targets, val_probs)
    best_f1_metrics = select_best_f1_threshold(val_sweep_rows)
    gate_metrics = select_stage1_gate_threshold(val_sweep_rows, recall_floor=recall_floor)
    history_rows.extend(
        make_threshold_sweep_rows(
            split_name="val",
            epoch=int(best_state["epoch"]),
            sweep_rows=val_sweep_rows,
            best_f1_threshold=float(best_f1_metrics["threshold"]),
            gate_threshold=float(gate_metrics["threshold"]),
        )
    )

    test_loss, test_targets, test_probs = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
    test_metrics = metrics_at_threshold(test_targets, test_probs, float(gate_metrics["threshold"]))
    history_rows.append(
        {
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
        }
    )

    metrics_df = pd.DataFrame(history_rows)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.md"
    metrics_df.to_csv(metrics_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Hold/No-Hold Classifier Training Summary\n\n")
        f.write(f"- Train samples: `{len(train_dataset)}`\n")
        f.write(f"- Validation samples: `{len(val_dataset)}`\n")
        f.write(f"- Test samples: `{len(test_dataset)}`\n")
        f.write(f"- Positive weight: `{pos_weight_value:.4f}`\n")
        f.write(f"- Best checkpoint epoch: `{int(best_state['epoch'])}`\n")
        f.write(f"- Best-F1 validation threshold: `{float(best_f1_metrics['threshold']):.2f}`\n")
        f.write(f"- Stage 1 gate threshold: `{float(gate_metrics['threshold']):.2f}`\n")
        f.write(f"- Threshold policy: `recall_floor`\n")
        f.write(f"- Recall floor for gate selection: `{recall_floor:.2f}`\n\n")
        f.write("## Why this threshold was chosen\n\n")
        f.write(
            "The real pipeline uses a permissive `hold/no_hold` gate. "
            "The selected Stage 1 threshold is the highest validation threshold that still preserves the configured recall floor, "
            "with F1 used as a tie-breaker among eligible thresholds.\n\n"
        )
        f.write("Expected tradeoff: higher candidate flow to Stage 2, fewer collapsed-recall runs, and more false positives handled later by YOLO26n.\n\n")
        f.write("## Validation sweep summary\n\n")
        f.write(
            f"- Best-F1 validation metrics: precision `{float(best_f1_metrics['precision']):.4f}`, "
            f"recall `{float(best_f1_metrics['recall']):.4f}`, F1 `{float(best_f1_metrics['f1']):.4f}`\n"
        )
        f.write(
            f"- Gate validation metrics: precision `{float(gate_metrics['precision']):.4f}`, "
            f"recall `{float(gate_metrics['recall']):.4f}`, F1 `{float(gate_metrics['f1']):.4f}`\n\n"
        )
        sweep_df = metrics_df[metrics_df["kind"] == "threshold_sweep"].copy()
        if not sweep_df.empty:
            f.write(sweep_df.to_markdown(index=False))
            f.write("\n\n")
        final_test_df = metrics_df[metrics_df["kind"] == "final_test"].copy()
        if not final_test_df.empty:
            f.write("## Final test metrics with gate threshold\n\n")
            f.write(final_test_df.to_markdown(index=False))
            f.write("\n")

    print(f"[OK] Saved checkpoint: {checkpoint_path}")
    print(f"[OK] Saved metrics: {metrics_path}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
