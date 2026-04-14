from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VALID_WEAPON_CLASSES = {
    "weapon",
    "handgun",
    "gun",
    "pistol",
    "revolver",
    "rifle",
    "firearm",
    "short_rifle",
    "short-rifle",
}


def infer_camera_id(filename: str) -> str:
    """
    Tenta inferir camera_id a partir do nome do arquivo.
    Exemplos aceitos:
      cam1_001.jpg
      Cam5-frame-002.xml
      camera7_xyz.jpg
      C1_foo.jpg
    """
    name = filename.lower()

    patterns = [
        r"(?:cam|camera)[_\-\s]?(\d+)",
        r"\bc(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return f"cam{match.group(1)}"

    return "unknown"


def parse_xml(annotation_path: Path) -> dict:
    """
    Faz parsing básico do XML Pascal VOC.
    Retorna metadados e lista de objetos.
    """
    result = {
        "xml_ok": False,
        "width": None,
        "height": None,
        "objects": [],
        "error": "",
    }

    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        size = root.find("size")
        if size is not None:
            width_text = size.findtext("width")
            height_text = size.findtext("height")
            if width_text is not None and height_text is not None:
                result["width"] = int(float(width_text))
                result["height"] = int(float(height_text))

        objects = []
        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            bndbox = obj.find("bndbox")

            xmin = ymin = xmax = ymax = None
            if bndbox is not None:
                xmin_text = bndbox.findtext("xmin")
                ymin_text = bndbox.findtext("ymin")
                xmax_text = bndbox.findtext("xmax")
                ymax_text = bndbox.findtext("ymax")

                if None not in (xmin_text, ymin_text, xmax_text, ymax_text):
                    xmin = int(float(xmin_text))
                    ymin = int(float(ymin_text))
                    xmax = int(float(xmax_text))
                    ymax = int(float(ymax_text))

            objects.append(
                {
                    "class_name": class_name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )

        result["objects"] = objects
        result["xml_ok"] = True
        return result

    except Exception as exc:
        result["error"] = str(exc)
        return result


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_")


def is_weapon_class(class_name: str) -> bool:
    return normalize_class_name(class_name) in VALID_WEAPON_CLASSES


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    images_dir = project_root / "data" / "raw" / "images"
    annotations_dir = project_root / "data" / "raw" / "annotations"
    interim_dir = project_root / "data" / "interim"
    results_dir = project_root / "results" / "audit"

    interim_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = interim_dir / "manifest.csv"
    summary_path = results_dir / "manifest_summary.txt"

    image_files = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )
    xml_files = sorted([p for p in annotations_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xml"])

    xml_by_stem = {p.stem: p for p in xml_files}
    image_by_stem = {p.stem: p for p in image_files}

    rows = []
    class_counter = Counter()
    camera_counter = Counter()
    status_counter = Counter()

    for image_path in image_files:
        stem = image_path.stem
        annotation_path = xml_by_stem.get(stem)

        camera_id = infer_camera_id(image_path.name)
        camera_counter[camera_id] += 1

        row = {
            "image_stem": stem,
            "image_filename": image_path.name,
            "image_path": str(image_path.relative_to(project_root)),
            "annotation_filename": annotation_path.name if annotation_path else "",
            "annotation_path": str(annotation_path.relative_to(project_root)) if annotation_path else "",
            "pair_status": "ok" if annotation_path else "missing_xml",
            "xml_ok": False,
            "camera_id": camera_id,
            "width": "",
            "height": "",
            "num_boxes": 0,
            "classes_present": "",
            "has_weapon": 0,
            "invalid_boxes": 0,
            "xml_error": "",
        }

        if annotation_path is not None:
            parsed = parse_xml(annotation_path)
            row["xml_ok"] = parsed["xml_ok"]
            row["xml_error"] = parsed["error"]

            if not parsed["xml_ok"]:
                row["pair_status"] = "xml_parse_error"
            else:
                width = parsed["width"]
                height = parsed["height"]
                row["width"] = width if width is not None else ""
                row["height"] = height if height is not None else ""

                objects = parsed["objects"]
                row["num_boxes"] = len(objects)

                classes = []
                invalid_boxes = 0
                has_weapon = 0

                for obj in objects:
                    class_name = normalize_class_name(obj["class_name"])
                    if class_name:
                        classes.append(class_name)
                        class_counter[class_name] += 1
                        if is_weapon_class(class_name):
                            has_weapon = 1

                    xmin = obj["xmin"]
                    ymin = obj["ymin"]
                    xmax = obj["xmax"]
                    ymax = obj["ymax"]

                    if None in (xmin, ymin, xmax, ymax):
                        invalid_boxes += 1
                        continue

                    if xmin >= xmax or ymin >= ymax:
                        invalid_boxes += 1
                        continue

                    if width not in ("", None) and height not in ("", None):
                        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                            invalid_boxes += 1

                row["classes_present"] = ";".join(sorted(set(classes)))
                row["has_weapon"] = has_weapon
                row["invalid_boxes"] = invalid_boxes

                if invalid_boxes > 0 and row["pair_status"] == "ok":
                    row["pair_status"] = "invalid_boxes"

        rows.append(row)
        status_counter[row["pair_status"]] += 1

    # XMLs órfãos: têm anotação mas não têm imagem correspondente
    orphan_xml_count = 0
    for xml_path in xml_files:
        if xml_path.stem not in image_by_stem:
            orphan_xml_count += 1
            rows.append(
                {
                    "image_stem": xml_path.stem,
                    "image_filename": "",
                    "image_path": "",
                    "annotation_filename": xml_path.name,
                    "annotation_path": str(xml_path.relative_to(project_root)),
                    "pair_status": "missing_image",
                    "xml_ok": False,
                    "camera_id": infer_camera_id(xml_path.name),
                    "width": "",
                    "height": "",
                    "num_boxes": 0,
                    "classes_present": "",
                    "has_weapon": 0,
                    "invalid_boxes": 0,
                    "xml_error": "",
                }
            )
            status_counter["missing_image"] += 1

    fieldnames = [
        "image_stem",
        "image_filename",
        "image_path",
        "annotation_filename",
        "annotation_path",
        "pair_status",
        "xml_ok",
        "camera_id",
        "width",
        "height",
        "num_boxes",
        "classes_present",
        "has_weapon",
        "invalid_boxes",
        "xml_error",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_images = len(image_files)
    total_xmls = len(xml_files)
    total_rows = len(rows)
    total_boxes = sum(class_counter.values())
    total_positive_images = sum(1 for r in rows if str(r["has_weapon"]) == "1")
    total_invalid_box_rows = sum(1 for r in rows if str(r["invalid_boxes"]) not in ("", "0"))

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== MANIFEST SUMMARY ===\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Total XML annotations: {total_xmls}\n")
        f.write(f"Total manifest rows: {total_rows}\n")
        f.write(f"Total boxes: {total_boxes}\n")
        f.write(f"Positive images (has_weapon=1): {total_positive_images}\n")
        f.write(f"Rows with invalid boxes: {total_invalid_box_rows}\n")
        f.write(f"Orphan XMLs (missing image): {orphan_xml_count}\n\n")

        f.write("Pair status counts:\n")
        for status, count in sorted(status_counter.items()):
            f.write(f"  {status}: {count}\n")

        f.write("\nCamera counts:\n")
        for camera, count in sorted(camera_counter.items()):
            f.write(f"  {camera}: {count}\n")

        f.write("\nClass counts:\n")
        for class_name, count in sorted(class_counter.items()):
            f.write(f"  {class_name}: {count}\n")

    print(f"[OK] Manifest saved to: {manifest_path}")
    print(f"[OK] Summary saved to: {summary_path}")
    print("\n=== QUICK SUMMARY ===")
    print(f"Total images: {total_images}")
    print(f"Total XMLs: {total_xmls}")
    print(f"Total manifest rows: {total_rows}")
    print(f"Total boxes: {total_boxes}")
    print(f"Positive images: {total_positive_images}")
    print(f"Rows with invalid boxes: {total_invalid_box_rows}")
    print(f"Orphan XMLs: {orphan_xml_count}")

    print("\nPair status counts:")
    for status, count in sorted(status_counter.items()):
        print(f"  {status}: {count}")

    print("\nTop classes:")
    for class_name, count in class_counter.most_common(10):
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()