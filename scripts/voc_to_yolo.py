from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# Mapeamento para o baseline fiel ao paper:
# handgun + short_rifle => weapon (classe 0)
# knife => excluir
CLASS_TO_YOLO_ID = {
    "handgun": 0,
    "short_rifle": 0,
    "short-rifle": 0,
    "rifle": 0,
    "gun": 0,
    "pistol": 0,
    "revolver": 0,
    "firearm": 0,
}

EXCLUDED_CLASSES = {"knife"}


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def parse_voc_xml(xml_path: Path) -> tuple[int | None, int | None, list[dict]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = None
    height = None

    if size is not None:
        width_text = size.findtext("width")
        height_text = size.findtext("height")
        if width_text is not None and height_text is not None:
            width = int(float(width_text))
            height = int(float(height_text))

    objects = []
    for obj in root.findall("object"):
        class_name = normalize_class_name(obj.findtext("name") or "")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = int(float(bndbox.findtext("xmin")))
            ymin = int(float(bndbox.findtext("ymin")))
            xmax = int(float(bndbox.findtext("xmax")))
            ymax = int(float(bndbox.findtext("ymax")))
        except (TypeError, ValueError):
            continue

        objects.append(
            {
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return width, height, objects


def voc_box_to_yolo(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    box_width = xmax - xmin
    box_height = ymax - ymin

    x_center = xmin + box_width / 2.0
    y_center = ymin + box_height / 2.0

    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_split(
    project_root: Path,
    split_name: str,
    split_manifest_path: Path,
    yolo_root: Path,
) -> dict:
    df = pd.read_csv(split_manifest_path)

    images_out_dir = yolo_root / "images" / split_name
    labels_out_dir = yolo_root / "labels" / split_name

    ensure_dir(images_out_dir)
    ensure_dir(labels_out_dir)

    images_processed = 0
    images_with_weapon = 0
    negative_images = 0
    excluded_boxes = 0
    converted_boxes = 0
    skipped_invalid_boxes = 0

    for _, row in df.iterrows():
        image_rel = row["image_path"]
        xml_rel = row["annotation_path"]

        image_src = project_root / image_rel
        xml_src = project_root / xml_rel

        image_dst = images_out_dir / image_src.name
        label_dst = labels_out_dir / f"{image_src.stem}.txt"

        if not image_src.exists():
            raise FileNotFoundError(f"Image not found: {image_src}")
        if not xml_src.exists():
            raise FileNotFoundError(f"XML not found: {xml_src}")

        width, height, objects = parse_voc_xml(xml_src)

        if width is None or height is None:
            raise ValueError(f"Missing size info in XML: {xml_src}")

        yolo_lines = []

        for obj in objects:
            class_name = obj["class_name"]

            if class_name in EXCLUDED_CLASSES:
                excluded_boxes += 1
                continue

            if class_name not in CLASS_TO_YOLO_ID:
                # classe não coberta pelo baseline atual
                continue

            xmin = obj["xmin"]
            ymin = obj["ymin"]
            xmax = obj["xmax"]
            ymax = obj["ymax"]

            if xmin >= xmax or ymin >= ymax:
                skipped_invalid_boxes += 1
                continue

            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                skipped_invalid_boxes += 1
                continue

            x_center, y_center, box_width, box_height = voc_box_to_yolo(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                image_width=width,
                image_height=height,
            )

            class_id = CLASS_TO_YOLO_ID[class_name]
            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )
            converted_boxes += 1

        shutil.copy2(image_src, image_dst)

        with label_dst.open("w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        images_processed += 1
        if yolo_lines:
            images_with_weapon += 1
        else:
            negative_images += 1

    return {
        "split": split_name,
        "images_processed": images_processed,
        "images_with_weapon": images_with_weapon,
        "negative_images": negative_images,
        "converted_boxes": converted_boxes,
        "excluded_boxes": excluded_boxes,
        "skipped_invalid_boxes": skipped_invalid_boxes,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    splits_dir = project_root / "data" / "splits"
    yolo_root = project_root / "data" / "interim" / "yolo"
    docs_dir = project_root / "docs"

    ensure_dir(yolo_root / "images")
    ensure_dir(yolo_root / "labels")
    ensure_dir(docs_dir)

    split_files = {
        "train": splits_dir / "train_manifest.csv",
        "val": splits_dir / "val_manifest.csv",
        "test": splits_dir / "test_manifest.csv",
    }

    for split_name, split_file in split_files.items():
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split manifest: {split_file}")

    summaries = []
    for split_name, split_file in split_files.items():
        summary = process_split(
            project_root=project_root,
            split_name=split_name,
            split_manifest_path=split_file,
            yolo_root=yolo_root,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_csv_path = project_root / "results" / "split_stats" / "yolo_conversion_summary.csv"
    ensure_dir(summary_csv_path.parent)
    summary_df.to_csv(summary_csv_path, index=False)

    summary_md_path = docs_dir / "sprint2_yolo_conversion_summary.md"
    with summary_md_path.open("w", encoding="utf-8") as f:
        f.write("# Sprint 2 - VOC to YOLO Conversion Summary\n\n")
        f.write("## Mapping used\n\n")
        f.write("- `handgun` -> `weapon` (class `0`)\n")
        f.write("- `short_rifle` -> `weapon` (class `0`)\n")
        f.write("- `knife` -> excluded from YOLO baseline labels\n\n")

        f.write("## Conversion summary by split\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")

    print(f"[OK] Saved: {summary_csv_path}")
    print(f"[OK] Saved: {summary_md_path}")

    print("\n=== YOLO CONVERSION SUMMARY ===")
    for _, row in summary_df.iterrows():
        print(
            f"{row['split']}: "
            f"images_processed={row['images_processed']}, "
            f"images_with_weapon={row['images_with_weapon']}, "
            f"negative_images={row['negative_images']}, "
            f"converted_boxes={row['converted_boxes']}, "
            f"excluded_boxes={row['excluded_boxes']}, "
            f"skipped_invalid_boxes={row['skipped_invalid_boxes']}"
        )

    print("\nYOLO dataset created at:")
    print(yolo_root)


if __name__ == "__main__":
    main()