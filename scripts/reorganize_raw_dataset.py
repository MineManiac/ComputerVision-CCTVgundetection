from pathlib import Path
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANNOT_EXTS = {".xml"}

def move_file(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists():
        print(f"[SKIP] {dst} already exists")
        return

    shutil.move(str(src), str(dst))
    print(f"[MOVE] {src} -> {dst}")

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    # possíveis locais onde seus arquivos podem estar hoje
    candidate_dirs = [
        project_root / "data",
        project_root / "data" / "cctv_mock_attack",
        project_root / "data" / "cctv_mock_attack" / "Images",
        project_root / "data" / "Images",
    ]

    raw_images_dir = project_root / "data" / "raw" / "images"
    raw_annotations_dir = project_root / "data" / "raw" / "annotations"

    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_annotations_dir.mkdir(parents=True, exist_ok=True)

    moved_images = 0
    moved_annotations = 0

    for base_dir in candidate_dirs:
        if not base_dir.exists():
            continue

        for file_path in base_dir.rglob("*"):
            if not file_path.is_file():
                continue

            # não mexer no que já está organizado
            if raw_images_dir in file_path.parents or raw_annotations_dir in file_path.parents:
                continue

            suffix = file_path.suffix.lower()

            if suffix in IMAGE_EXTS:
                move_file(file_path, raw_images_dir)
                moved_images += 1
            elif suffix in ANNOT_EXTS:
                move_file(file_path, raw_annotations_dir)
                moved_annotations += 1

    print("\n=== SUMMARY ===")
    print(f"Images moved: {moved_images}")
    print(f"Annotations moved: {moved_annotations}")
    print(f"Images dir: {raw_images_dir}")
    print(f"Annotations dir: {raw_annotations_dir}")

if __name__ == "__main__":
    main()