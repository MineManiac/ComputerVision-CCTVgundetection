from pathlib import Path
from ultralytics import YOLO

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "configs" / "yolo_data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {data_yaml}")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_yaml),
        epochs=5,
        imgsz=640,
        batch=8,
        project=str(project_root / "runs"),
        name="yolo11n_smoke",
        pretrained=True,
        workers=0,
        device=0,
        cache=False,
        verbose=True,
        seed=42,
    )

if __name__ == "__main__":
    main()