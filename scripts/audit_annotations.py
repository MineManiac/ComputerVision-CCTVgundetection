from __future__ import annotations

import ast
import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Critério COCO citado no paper
SMALL_MAX_AREA = 32 * 32
MEDIUM_MAX_AREA = 96 * 96

# baseline fiel ao paper:
# handgun + short_rifle -> weapon
# knife fica fora
CLASS_MAP = {
    "handgun": "weapon",
    "gun": "weapon",
    "pistol": "weapon",
    "revolver": "weapon",
    "rifle": "weapon",
    "short_rifle": "weapon",
    "short-rifle": "weapon",
    "firearm": "weapon",
    "knife": "exclude",
}


def parse_boxes_from_xml(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip().lower().replace(" ", "_")
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

        w = xmax - xmin
        h = ymax - ymin
        area = w * h

        if area < SMALL_MAX_AREA:
            size_bucket = "small"
        elif area < MEDIUM_MAX_AREA:
            size_bucket = "medium"
        else:
            size_bucket = "large"

        mapped_class = CLASS_MAP.get(class_name, "unknown")

        boxes.append(
            {
                "original_class": class_name,
                "mapped_class": mapped_class,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": w,
                "height": h,
                "area": area,
                "size_bucket": size_bucket,
            }
        )

    return boxes


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    manifest_path = project_root / "data" / "interim" / "manifest.csv"
    audit_dir = project_root / "results" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    # filtrar apenas pares válidos
    valid_df = df[df["pair_status"] == "ok"].copy()

    all_boxes = []
    image_level_rows = []

    for _, row in valid_df.iterrows():
        xml_rel = row["annotation_path"]
        xml_path = project_root / xml_rel
        camera_id = row["camera_id"]
        image_filename = row["image_filename"]

        boxes = parse_boxes_from_xml(xml_path)

        # estatísticas por imagem
        mapped_counts = Counter(b["mapped_class"] for b in boxes)
        original_counts = Counter(b["original_class"] for b in boxes)

        image_level_rows.append(
            {
                "image_filename": image_filename,
                "camera_id": camera_id,
                "num_boxes_total": len(boxes),
                "num_weapon_boxes": mapped_counts.get("weapon", 0),
                "num_excluded_boxes": mapped_counts.get("exclude", 0),
                "num_unknown_boxes": mapped_counts.get("unknown", 0),
                "has_weapon_after_mapping": int(mapped_counts.get("weapon", 0) > 0),
            }
        )

        for b in boxes:
            all_boxes.append(
                {
                    "image_filename": image_filename,
                    "camera_id": camera_id,
                    **b,
                }
            )

    boxes_df = pd.DataFrame(all_boxes)
    image_stats_df = pd.DataFrame(image_level_rows)

    # Se não houver boxes, evita quebrar
    if boxes_df.empty:
        raise RuntimeError("Nenhum bounding box foi encontrado no dataset auditado.")

    # distribuição de classes originais
    class_distribution = (
        boxes_df.groupby("original_class")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # distribuição após mapeamento
    mapped_distribution = (
        boxes_df.groupby("mapped_class")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # distribuição por câmera
    camera_distribution = (
        image_stats_df.groupby("camera_id")
        .agg(
            total_images=("image_filename", "count"),
            positive_images_after_mapping=("has_weapon_after_mapping", "sum"),
            total_boxes=("num_boxes_total", "sum"),
            total_weapon_boxes=("num_weapon_boxes", "sum"),
            total_excluded_boxes=("num_excluded_boxes", "sum"),
            total_unknown_boxes=("num_unknown_boxes", "sum"),
        )
        .reset_index()
        .sort_values("camera_id")
    )

    # tamanho dos boxes por câmera e classe mapeada
    box_size_distribution = (
        boxes_df.groupby(["camera_id", "mapped_class", "size_bucket"])
        .size()
        .reset_index(name="count")
        .sort_values(["camera_id", "mapped_class", "size_bucket"])
    )

    # distribuição global de tamanhos apenas para weapon
    weapon_boxes_df = boxes_df[boxes_df["mapped_class"] == "weapon"].copy()
    weapon_size_distribution = (
        weapon_boxes_df.groupby("size_bucket")
        .size()
        .reset_index(name="count")
        .sort_values("size_bucket")
    )

    # boxes desconhecidos
    unknown_boxes_df = boxes_df[boxes_df["mapped_class"] == "unknown"].copy()

    # salvar CSVs
    class_distribution.to_csv(audit_dir / "class_distribution_original.csv", index=False)
    mapped_distribution.to_csv(audit_dir / "class_distribution_mapped.csv", index=False)
    camera_distribution.to_csv(audit_dir / "camera_distribution.csv", index=False)
    box_size_distribution.to_csv(audit_dir / "box_size_distribution_by_camera.csv", index=False)
    weapon_size_distribution.to_csv(audit_dir / "weapon_size_distribution.csv", index=False)
    image_stats_df.to_csv(audit_dir / "image_level_stats.csv", index=False)
    unknown_boxes_df.to_csv(audit_dir / "unknown_boxes.csv", index=False)

    # resumo markdown
    total_images = len(valid_df)
    total_boxes = len(boxes_df)
    total_weapon_boxes = len(weapon_boxes_df)
    total_excluded_boxes = len(boxes_df[boxes_df["mapped_class"] == "exclude"])
    total_unknown_boxes = len(unknown_boxes_df)
    total_positive_images_after_mapping = int(image_stats_df["has_weapon_after_mapping"].sum())

    size_counts = dict(zip(weapon_size_distribution["size_bucket"], weapon_size_distribution["count"]))
    small_count = size_counts.get("small", 0)
    medium_count = size_counts.get("medium", 0)
    large_count = size_counts.get("large", 0)

    summary_md = audit_dir / "audit_summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Sprint 2 - Annotation Audit Summary\n\n")
        f.write(f"- Total valid images: **{total_images}**\n")
        f.write(f"- Total boxes (all original classes): **{total_boxes}**\n")
        f.write(f"- Total boxes mapped to `weapon`: **{total_weapon_boxes}**\n")
        f.write(f"- Total excluded boxes (`knife`): **{total_excluded_boxes}**\n")
        f.write(f"- Total unknown boxes: **{total_unknown_boxes}**\n")
        f.write(f"- Positive images after mapping to `weapon`: **{total_positive_images_after_mapping}**\n\n")

        f.write("## Weapon box sizes (COCO-style buckets)\n\n")
        f.write(f"- small: **{small_count}**\n")
        f.write(f"- medium: **{medium_count}**\n")
        f.write(f"- large: **{large_count}**\n\n")

        f.write("## Camera distribution\n\n")
        f.write(camera_distribution.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Original class distribution\n\n")
        f.write(class_distribution.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Mapped class distribution\n\n")
        f.write(mapped_distribution.to_markdown(index=False))
        f.write("\n\n")

        if total_unknown_boxes > 0:
            f.write("## Warning\n\n")
            f.write("- There are unknown classes not covered by CLASS_MAP. Check `unknown_boxes.csv`.\n")

    print(f"[OK] Saved: {audit_dir / 'class_distribution_original.csv'}")
    print(f"[OK] Saved: {audit_dir / 'class_distribution_mapped.csv'}")
    print(f"[OK] Saved: {audit_dir / 'camera_distribution.csv'}")
    print(f"[OK] Saved: {audit_dir / 'box_size_distribution_by_camera.csv'}")
    print(f"[OK] Saved: {audit_dir / 'weapon_size_distribution.csv'}")
    print(f"[OK] Saved: {audit_dir / 'image_level_stats.csv'}")
    print(f"[OK] Saved: {audit_dir / 'unknown_boxes.csv'}")
    print(f"[OK] Saved: {summary_md}")

    print("\n=== AUDIT SUMMARY ===")
    print(f"Total valid images: {total_images}")
    print(f"Total boxes: {total_boxes}")
    print(f"Total weapon boxes: {total_weapon_boxes}")
    print(f"Total excluded boxes (knife): {total_excluded_boxes}")
    print(f"Total unknown boxes: {total_unknown_boxes}")
    print(f"Positive images after mapping: {total_positive_images_after_mapping}")
    print(f"Weapon sizes -> small: {small_count}, medium: {medium_count}, large: {large_count}")


if __name__ == "__main__":
    main()