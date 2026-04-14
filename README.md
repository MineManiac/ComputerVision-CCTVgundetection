# CCTV Gun Detection - Reproduction Study

**Course:** Computer Vision - Insper  
**Team:** SafeSight Research  
**Members:** Matheus Ribeiro Barros, Fernanda de Oliveira Pereira

> Reproducing and evaluating CCTV weapon detection pipelines using the public mock attack dataset from Salazar-Gonzalez et al. (2020), with a paper-inspired setup and a modern YOLO11n baseline prepared in Sprint 2.

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
  - [References](#references)

---

## Project Overview

This project is a structured reproduction and evaluation study on CCTV weapon detection. It does **not** claim algorithmic novelty. The current workflow is based on the public CCTV mock attack dataset from Salazar-Gonzalez et al. (2020), while Sprint 2 focuses on preparing a reproducible data pipeline and a modern YOLO11n baseline.

The current goals are:

1. Organize and audit the real CCTV dataset in a reproducible way.
2. Build train/validation/test splits aligned with the camera setup of the dataset.
3. Prepare a YOLO-format version of the dataset using a single `weapon` class.
4. Run a first smoke test with YOLO11n before moving to longer experiments and model comparisons.

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
# Create the local data directory
mkdir -p data

# Download the zip (~2 GB) directly from Hugging Face
curl -L -o data/weapons_images_2fps.zip \
  "https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/resolve/main/weapons_images_2fps.zip"

# Extract into data/
cd data && unzip weapons_images_2fps.zip -d cctv_mock_attack && cd ..
```

On **Windows PowerShell**:

```powershell
# Create the local data directory
New-Item -ItemType Directory -Force data

# Download
curl.exe -L --retry 5 --retry-delay 5 `
  "https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/resolve/main/weapons_images_2fps.zip" `
  -o "data\weapons_images_2fps.zip"

# Extract
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
│   └── yolo/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
└── splits/
```

> Tip: keep `data/` in `.gitignore` to avoid committing large local files.

---

## Installation

```bash
# Clone this repository
git clone https://github.com/<your-org>/ComputerVision-CCTVgundetection.git
cd ComputerVision-CCTVgundetection

# Create a virtual environment
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Current key dependencies:

| Package | Purpose |
|--------|---------|
| `pandas` | Manifest, split summaries, audit tables |
| `matplotlib` | Plots and analysis |
| `pillow` | Image utilities |
| `pyyaml` | Dataset configuration files |
| `opencv-python` | Image processing utilities |
| `ultralytics` | YOLO11n training and inference |

---


## Project Structure

```text
ComputerVision-CCTVgundetection/
├── configs/
│   └── yolo_data.yaml
├── data/                        # Local dataset only, not committed
│   ├── raw/
│   │   ├── images/
│   │   └── annotations/
│   ├── interim/
│   │   ├── manifest.csv
│   │   └── yolo/
│   │       ├── images/
│   │       └── labels/
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── train_manifest.csv
│       ├── val_manifest.csv
│       └── test_manifest.csv
├── docs/                        # Sprint notes and summaries
├── results/
│   ├── audit/
│   └── split_stats/
├── runs/                        # YOLO training outputs
├── scripts/
│   ├── reorganize_raw_dataset.py
│   ├── build_manifest.py
│   ├── audit_annotations.py
│   ├── make_splits.py
│   ├── voc_to_yolo.py
│   └── train_yolo_smoke.py
├── requirements.txt
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
    -> [YOLO11n smoke test]
    -> [Longer training, comparisons, and robustness analysis]
```

---

## References

1. Salazar Gonzalez, J. L., Zaccaro, C., Alvarez-Garcia, J. A., Soria-Morillo, L. M., and Sancho Caparrini, F. (2020). *Real-time gun detection in CCTV: An open problem.* Neural Networks. https://doi.org/10.1016/j.neunet.2020.09.013
2. Ultralytics. *YOLO Documentation.* https://docs.ultralytics.com
3. Wang, G. et al. (2023). *Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLO v4.* Digital Signal Processing.
4. Bochkovskiy, A. et al. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection.* arXiv:2004.10934.

---

*SafeSight Research - Insper Computer Vision - 2026*
