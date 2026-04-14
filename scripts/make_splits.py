from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

TRAIN_CAMERAS = {"cam1", "cam7"}
TEST_CAMERAS = {"cam5"}

VAL_FRACTION = 0.10
CHUNK_SIZE = 15
SEED = 42


def extract_frame_number(filename_stem: str) -> int | None:
    numbers = re.findall(r"\d+", filename_stem)
    if not numbers:
        return None
    return int(numbers[-1])


def build_chunks(camera_df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    df = camera_df.copy()
    df["frame_number"] = df["image_stem"].apply(extract_frame_number)
    df["frame_sort_key"] = df["frame_number"].fillna(-1)
    df = df.sort_values(["frame_sort_key", "image_filename"]).reset_index(drop=True)

    group_ids = []
    for idx in range(len(df)):
        group_idx = idx // chunk_size
        group_ids.append(f"{df.iloc[idx]['camera_id']}_group_{group_idx:04d}")

    df["group_id"] = group_ids
    return df


def stratified_group_split(
    camera_df: pd.DataFrame,
    val_fraction: float,
    chunk_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_chunks(camera_df, chunk_size=chunk_size)

    group_stats = (
        df.groupby("group_id")
        .agg(
            camera_id=("camera_id", "first"),
            images=("image_filename", "count"),
            positive_images=("has_weapon", "sum"),
            total_boxes=("num_boxes", "sum"),
        )
        .reset_index()
    )

    group_stats["has_positive"] = group_stats["positive_images"] > 0

    positive_groups = group_stats[group_stats["has_positive"]].copy()
    negative_groups = group_stats[~group_stats["has_positive"]].copy()

    rng = random.Random(seed)

    def pick_groups(groups_df: pd.DataFrame, target_fraction: float) -> set[str]:
        if groups_df.empty:
            return set()

        group_ids = groups_df["group_id"].tolist()
        rng.shuffle(group_ids)

        target_images = max(1, int(round(groups_df["images"].sum() * target_fraction)))

        selected = []
        selected_images = 0
        for gid in group_ids:
            selected.append(gid)
            selected_images += int(groups_df.loc[groups_df["group_id"] == gid, "images"].iloc[0])
            if selected_images >= target_images:
                break

        return set(selected)

    val_positive_groups = pick_groups(positive_groups, val_fraction)
    val_negative_groups = pick_groups(negative_groups, val_fraction)
    val_group_ids = val_positive_groups | val_negative_groups

    val_df = df[df["group_id"].isin(val_group_ids)].copy()
    train_df = df[~df["group_id"].isin(val_group_ids)].copy()

    return train_df, val_df


def save_image_list(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for image_path in df["image_path"].tolist():
            f.write(f"{image_path}\n")


def summarize_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    summary = (
        df.groupby("camera_id")
        .agg(
            images=("image_filename", "count"),
            positive_images=("has_weapon", "sum"),
            total_boxes=("num_boxes", "sum"),
        )
        .reset_index()
    )
    summary["split"] = split_name
    return summary[["split", "camera_id", "images", "positive_images", "total_boxes"]]


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    manifest_path = project_root / "data" / "interim" / "manifest.csv"
    splits_dir = project_root / "data" / "splits"
    stats_dir = project_root / "results" / "split_stats"
    docs_dir = project_root / "docs"

    splits_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df = df[df["pair_status"] == "ok"].copy()
    df["has_weapon"] = df["has_weapon"].fillna(0).astype(int)
    df["num_boxes"] = df["num_boxes"].fillna(0).astype(int)

    test_df = df[df["camera_id"].isin(TEST_CAMERAS)].copy()
    trainval_df = df[df["camera_id"].isin(TRAIN_CAMERAS)].copy()

    train_parts = []
    val_parts = []

    for camera_id in sorted(TRAIN_CAMERAS):
        camera_df = trainval_df[trainval_df["camera_id"] == camera_id].copy()
        train_cam_df, val_cam_df = stratified_group_split(
            camera_df=camera_df,
            val_fraction=VAL_FRACTION,
            chunk_size=CHUNK_SIZE,
            seed=SEED,
        )
        train_parts.append(train_cam_df)
        val_parts.append(val_cam_df)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    save_image_list(train_df, splits_dir / "train.txt")
    save_image_list(val_df, splits_dir / "val.txt")
    save_image_list(test_df, splits_dir / "test.txt")

    train_df.to_csv(splits_dir / "train_manifest.csv", index=False)
    val_df.to_csv(splits_dir / "val_manifest.csv", index=False)
    test_df.to_csv(splits_dir / "test_manifest.csv", index=False)

    train_summary = summarize_split(train_df, "train")
    val_summary = summarize_split(val_df, "val")
    test_summary = summarize_split(test_df, "test")

    split_summary_df = pd.concat([train_summary, val_summary, test_summary], ignore_index=True)
    split_summary_df.to_csv(stats_dir / "split_summary.csv", index=False)

    total_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "images": len(train_df),
                "positive_images": int(train_df["has_weapon"].sum()),
                "total_boxes": int(train_df["num_boxes"].sum()),
            },
            {
                "split": "val",
                "images": len(val_df),
                "positive_images": int(val_df["has_weapon"].sum()),
                "total_boxes": int(val_df["num_boxes"].sum()),
            },
            {
                "split": "test",
                "images": len(test_df),
                "positive_images": int(test_df["has_weapon"].sum()),
                "total_boxes": int(test_df["num_boxes"].sum()),
            },
        ]
    )
    total_summary.to_csv(stats_dir / "split_totals.csv", index=False)

    summary_md = docs_dir / "sprint2_split_summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Sprint 2 - Split Summary\n\n")
        f.write("## Strategy\n\n")
        f.write("- Test split fixed as all images from `cam5`\n")
        f.write("- Train/val pool built from `cam1` and `cam7`\n")
        f.write("- Validation built with grouped chunks and positive/negative stratification\n")
        f.write(f"- Validation fraction: `{VAL_FRACTION:.2f}`\n")
        f.write(f"- Group chunk size: `{CHUNK_SIZE}` images\n")
        f.write(f"- Random seed: `{SEED}`\n\n")

        f.write("## Totals by split\n\n")
        f.write(total_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Per-camera summary\n\n")
        f.write(split_summary_df.to_markdown(index=False))
        f.write("\n")

    print(f"[OK] Saved: {splits_dir / 'train.txt'}")
    print(f"[OK] Saved: {splits_dir / 'val.txt'}")
    print(f"[OK] Saved: {splits_dir / 'test.txt'}")
    print(f"[OK] Saved: {splits_dir / 'train_manifest.csv'}")
    print(f"[OK] Saved: {splits_dir / 'val_manifest.csv'}")
    print(f"[OK] Saved: {splits_dir / 'test_manifest.csv'}")
    print(f"[OK] Saved: {stats_dir / 'split_summary.csv'}")
    print(f"[OK] Saved: {stats_dir / 'split_totals.csv'}")
    print(f"[OK] Saved: {summary_md}")

    print("\n=== SPLIT TOTALS ===")
    for _, row in total_summary.iterrows():
        print(
            f"{row['split']}: "
            f"images={row['images']}, "
            f"positive_images={row['positive_images']}, "
            f"total_boxes={row['total_boxes']}"
        )

    print("\n=== PER CAMERA ===")
    for _, row in split_summary_df.iterrows():
        print(
            f"{row['split']} | {row['camera_id']}: "
            f"images={row['images']}, "
            f"positive_images={row['positive_images']}, "
            f"total_boxes={row['total_boxes']}"
        )


if __name__ == "__main__":
    main()