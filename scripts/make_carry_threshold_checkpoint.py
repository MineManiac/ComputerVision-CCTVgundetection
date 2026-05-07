from pathlib import Path
import argparse
import torch


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description="Write a new hold/no_hold threshold into an existing checkpoint.")
    parser.add_argument("--base", default="runs/two_phase/carry_classifier/best.pt")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--field",
        default="stage1_gate_threshold",
        choices=["best_f1_threshold", "stage1_gate_threshold"],
        help="Which threshold field to overwrite in the checkpoint.",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {base_path}")

    checkpoint = load_checkpoint(base_path)
    threshold_value = float(args.threshold)
    checkpoint[args.field] = threshold_value
    if args.field == "stage1_gate_threshold":
        checkpoint["best_threshold"] = threshold_value
        checkpoint["threshold_policy"] = "manual_override"

    out_path = base_path.parent / f"best_{args.field}_thr{int(args.threshold * 100):03d}.pt"
    torch.save(checkpoint, out_path)

    print(f"Saved: {out_path}")
    print(f"Updated {args.field}: {checkpoint[args.field]}")
    if "best_threshold" in checkpoint:
        print(f"Compatibility best_threshold: {checkpoint['best_threshold']}")


if __name__ == "__main__":
    main()
