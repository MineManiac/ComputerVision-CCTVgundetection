# CCTV Gun Detection - Reproduction Study

**Course:** Computer Vision - Insper  
**Team:** SafeSight Research  
**Members:** Matheus Ribeiro Barros, Fernanda de Oliveira Pereira

> Reproducing and evaluating CCTV weapon detection pipelines using the public mock attack dataset from Salazar-Gonzalez et al. (2020), with a reproducible data pipeline, modern YOLO baselines, and a Sprint 4 two-phase extension.

---

## Table of Contents

- [CCTV Gun Detection - Reproduction Study](#cctv-gun-detection---reproduction-study)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset Setup](#dataset-setup)
    - [Dataset - Mock Attack Dataset (Salazar-Gonzalez et al., 2020)](#dataset---mock-attack-dataset-salazar-gonzalez-et-al-2020)
    - [Option 1 - Direct download](#option-1---direct-download)
    - [Option 2 - Hugging Face dataset page](#option-2---hugging-face-dataset-page)
    - [Raw extracted structure](#raw-extracted-structure)
    - [Local organized structure used by this project](#local-organized-structure-used-by-this-project)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Pipeline Overview](#pipeline-overview)
  - [Sprint 4 Workflow](#sprint-4-workflow)
  - [References](#references)

---

## Project Overview

This project is a structured reproduction and evaluation study on CCTV weapon detection. It does **not** claim algorithmic novelty. The repository currently contains:

- the Sprint 2 data-preparation pipeline
- the Sprint 3 single-stage YOLO comparison artifacts
- the Sprint 4 two-phase pipeline scripts for person screening plus final weapon detection

The current goals are:

1. Organize and audit the real CCTV dataset in a reproducible way.
2. Build train/validation/test splits aligned with the camera setup of the dataset.
3. Prepare a YOLO-format version of the dataset using a single `weapon` class.
4. Compare modern YOLO baselines under a fixed local protocol.
5. Evaluate a two-phase pipeline that filters person crops before final weapon detection.

---

## Dataset Setup

> The dataset is **not included** in this repository because of its size.  
> Download it locally before running any experiments.

After downloading, place the dataset inside a `data/` folder at the root of this repository.

### Dataset - Mock Attack Dataset (Salazar-Gonzalez et al., 2020)

This is the real CCTV dataset used in this project. It was collected during a mock attack scenario in a university, captured by three surveillance cameras at 2 FPS, and manually annotated with weapon bounding boxes.

| Camera | Location | Frames |
|--------|----------|--------|
| Cam1 | Corridor 1 | 607 |
| Cam7 | Corridor 2 | 3,511 |
| Cam5 | University entrance | 1,031 |
| **Total** | | **5,149** |

**License:** CC BY-NC 4.0 - free for academic use with citation.

### Option 1 - Direct download

```bash
mkdir -p data
curl -L -o data/weapons_images_2fps.zip \
  "https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/resolve/main/weapons_images_2fps.zip"
cd data && unzip weapons_images_2fps.zip -d cctv_mock_attack && cd ..
```

On **Windows PowerShell**:

```powershell
New-Item -ItemType Directory -Force data
curl.exe -L --retry 5 --retry-delay 5 `
  "https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/resolve/main/weapons_images_2fps.zip" `
  -o "data\weapons_images_2fps.zip"
Expand-Archive -Path "data\weapons_images_2fps.zip" -DestinationPath "data\cctv_mock_attack"
```

### Option 2 - Hugging Face dataset page

Download manually from the official dataset page:  
https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset

### Raw extracted structure

The extracted archive stores the files in a shared `Images/` folder. Each `.jpg` frame has a matching `.xml` annotation file in the same directory, and the camera is identified by the filename prefix (`Cam1-`, `Cam5-`, `Cam7-`).

```text
data/
└── cctv_mock_attack/
    └── Images/
        ├── Cam1-...jpg
        ├── Cam1-...xml
        ├── Cam5-...jpg
        ├── Cam5-...xml
        ├── Cam7-...jpg
        └── Cam7-...xml
```

### Local organized structure used by this project

After extraction, the dataset is reorganized locally into separate image and annotation folders:

```text
data/
├── raw/
│   ├── images/
│   └── annotations/
├── interim/
│   ├── manifest.csv
│   ├── yolo/
│   │   ├── images/
│   │   └── labels/
│   └── two_phase/
│       ├── crops/
│       └── metadata/
└── splits/
    ├── train_manifest.csv
    ├── val_manifest.csv
    └── test_manifest.csv
```

> Tip: keep `data/` in `.gitignore` to avoid committing large local files.

---

## Installation

```bash
git clone https://github.com/<your-org>/ComputerVision-CCTVgundetection.git
cd ComputerVision-CCTVgundetection
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Current key dependencies:

| Package | Purpose |
|--------|---------|
| `pandas` | Manifests, split summaries, evaluation tables |
| `matplotlib` | Analysis plots |
| `pillow` | Image loading and crop generation |
| `pyyaml` | Config files |
| `torch` | Sprint 4 hold/no_hold classifier |
| `torchvision` | Transforms and image utilities |
| `ultralytics` | YOLO training and inference |

---

## Project Structure

```text
ComputerVision-CCTVgundetection/
├── configs/
│   ├── yolo_data.yaml
│   └── two_phase.yaml
├── data/                        # Local dataset only, not committed
│   ├── raw/
│   ├── interim/
│   └── splits/
├── docs/
│   ├── sprint3_results_summary.md
│   └── sprint4_two_phase_protocol.md
├── results/
│   ├── audit/
│   └── split_stats/
├── scripts/
│   ├── build_manifest.py
│   ├── audit_annotations.py
│   ├── make_splits.py
│   ├── voc_to_yolo.py
│   ├── train_yolo_smoke.py
│   ├── two_phase_utils.py
│   ├── build_two_phase_dataset.py
│   ├── train_carry_classifier.py
│   ├── run_two_phase_inference.py
│   └── evaluate_detection_pipeline.py
└── README.md
```

---

## Pipeline Overview

```text
[Downloaded CCTV zip]
    -> [Extract raw Images/ folder]
    -> [Reorganize into raw/images + raw/annotations]
    -> [Build manifest]
    -> [Annotation audit + class remapping]
    -> [Grouped camera-aware train/val/test split]
    -> [VOC to YOLO conversion]
    -> [Sprint 3 single-stage YOLO comparison]
    -> [Sprint 4 person detection]
    -> [Sprint 4 hold/no_hold screening]
    -> [Sprint 4 weapon detection on approved crops]
    -> [Single-stage vs two-phase evaluation]
```

---

## Sprint 4 Workflow

The Sprint 4 pipeline is prepared to run once the required checkpoints are available.

### 1. Build the person-crop dataset

```bash
python scripts/build_two_phase_dataset.py --device 0
```

### 2. Train the Stage 1 hold/no_hold classifier

```bash
python scripts/train_carry_classifier.py --device 0
```

### 3. Run two-phase inference

```bash
python scripts/run_two_phase_inference.py \
  --split test \
  --device 0 \
  --weapon-model runs/<your-yolo26n-checkpoint>/weights/best.pt
```

### 4. Compare single-stage vs two-phase

```bash
python scripts/evaluate_detection_pipeline.py \
  --split test \
  --device 0 \
  --single-stage-model runs/<your-yolo26n-checkpoint>/weights/best.pt \
  --two-phase-predictions runs/two_phase/predictions/test_predictions.csv \
  --two-phase-image-summary runs/two_phase/predictions/test_image_summary.csv
```

Important notes:

- `yolo11n.pt` is the default person detector for Stage 0.
- The Sprint 3 `YOLO26n` checkpoint is **not** versioned in the repository, so pass it explicitly with `--weapon-model` or update `configs/two_phase.yaml`.
- The detailed protocol is documented in `docs/sprint4_two_phase_protocol.md`.

---

## References

1. Salazar Gonzalez, J. L., Zaccaro, C., Alvarez-Garcia, J. A., Soria-Morillo, L. M., and Sancho Caparrini, F. (2020). *Real-time gun detection in CCTV: An open problem.* Neural Networks. https://doi.org/10.1016/j.neunet.2020.09.013
2. Ultralytics. *YOLO Documentation.* https://docs.ultralytics.com
3. Wang, G. et al. (2023). *Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLO v4.* Digital Signal Processing.
4. Bochkovskiy, A. et al. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection.* arXiv:2004.10934.

---

*SafeSight Research - Insper Computer Vision - 2026*
