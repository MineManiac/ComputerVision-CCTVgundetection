from pathlib import Path
import argparse
import torch


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="runs/two_phase/carry_classifier/best.pt")
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {base_path}")

    checkpoint = load_checkpoint(base_path)
    checkpoint["best_threshold"] = float(args.threshold)

    out_path = base_path.parent / f"best_thr{int(args.threshold * 100):03d}.pt"
    torch.save(checkpoint, out_path)

    print(f"Saved: {out_path}")
    print(f"New best_threshold: {checkpoint['best_threshold']}")


if __name__ == "__main__":
    main()