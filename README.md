# CCTV Gun Detection — YOLOv4 Reproduction Study

**Course:** Computer Vision · Insper  
**Team:** SafeSight Research  
**Members:** Matheus Ribeiro Barros, Fernanda de Oliveira Pereira

> Reproducing and evaluating *"Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLOv4"* (Wang et al., DSP 2023) using fully public datasets, with a modern YOLO comparison and adversarial-condition robustness analysis.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Setup](#-dataset-setup)
  - [Expected Directory Structure](#expected-directory-structure)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [References](#-references)

---

## 🔍 Project Overview

This project is a **structured reproduction and stress-test study** of a 2023 paper on CCTV weapon detection. It does **not** claim algorithmic novelty — the goal is to:

1. Reproduce the improved YOLOv4 pipeline described in the paper using public data.
2. Compare it against a standard YOLOv4 baseline and a modern YOLO-family detector (e.g., YOLO11).
3. Evaluate robustness under realistic CCTV conditions: motion blur, low light, and small object scale.
4. Explore a **two-phase detection strategy** (carry-screening → weapon classification) to reduce false positives.

---

## 📦 Dataset Setup

> **⚠️ The dataset is NOT included in this repository** due to its size.  
> Follow the steps below to download it locally before running any experiments.

After downloading, place the dataset inside a `data/` folder at the root of this repository (see [Expected Directory Structure](#expected-directory-structure)).

### Dataset — Mock Attack Dataset (Salazar-González et al., 2020)

This is the **real CCTV dataset** used in this project. It was collected during a mock attack scenario in a university, captured by three surveillance cameras at 2 FPS, and manually annotated with weapon bounding boxes.

| Camera | Location | Frames |
|--------|----------|--------|
| Cam1 | Corridor 1 | 607 |
| Cam7 | Corridor 2 | 3,511 |
| Cam5 | University entrance | 1,031 |
| **Total** | | **5,149** |

**License:** CC BY-NC 4.0 — free for academic use with citation (see [References](#-references))

#### Option 1 — Direct download (recommended)

```bash
# Download the zip (~2 GB) directly from Hugging Face
curl -L -o data/weapons_images_2fps.zip \
  "https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/resolve/main/weapons_images_2fps.zip"

# Extract into data/
cd data && unzip weapons_images_2fps.zip -d cctv_mock_attack && cd ..
```

On **Windows PowerShell**:

```powershell
# Download
Invoke-WebRequest -Uri "https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/resolve/main/weapons_images_2fps.zip" `
  -OutFile "data\weapons_images_2fps.zip"

# Extract
Expand-Archive -Path "data\weapons_images_2fps.zip" -DestinationPath "data\cctv_mock_attack"
```

#### Option 2 — Hugging Face dataset page

Browse and download manually from the official dataset page:  
👉 **https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset**

---

### Expected Directory Structure

```
ComputerVision-CCTVgundetection/
├── data/
│   └── cctv_mock_attack/
│       ├── Cam1/
│       │   ├── images/
│       │   └── annotations/
│       ├── Cam5/
│       │   ├── images/
│       │   └── annotations/
│       └── Cam7/
│           ├── images/
│           └── annotations/
│
├── notebooks/
├── scripts/
├── models/
├── results/
├── cctv_gun_detection_research_proposal.md
└── README.md
```

> **Tip:** Add `data/` to your `.gitignore` to avoid accidentally committing large files.

```gitignore
# .gitignore
data/
*.pt
*.weights
*.onnx
__pycache__/
.env
```

---

## 🛠 Installation

```bash
# Clone this repository
git clone https://github.com/<your-org>/ComputerVision-CCTVgundetection.git
cd ComputerVision-CCTVgundetection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.\.venv\Scripts\activate         # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|--------|---------|
| `torch` / `torchvision` | Deep learning backend |
| `ultralytics` | YOLO11 / modern YOLO baseline |
| `opencv-python` | Image preprocessing & analysis |
| `albumentations` | Data augmentation |
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |

---

## 📁 Project Structure

```
ComputerVision-CCTVgundetection/
├── data/                   # Dataset (not committed — see Dataset Setup above)
├── notebooks/              # Exploratory notebooks (EDA, failure analysis)
├── scripts/
│   ├── preprocess.py       # Blur/brightness/size analysis
│   ├── split_dataset.py    # Train/val/test split generation
│   └── evaluate.py         # mAP@50, mAP@50:95, FPS evaluation
├── models/
│   ├── yolov4_baseline/    # Standard YOLOv4 config & weights
│   ├── yolov4_improved/    # Reproduced improved YOLOv4 (Wang et al.)
│   └── yolo_modern/        # Modern YOLO family comparison
├── results/                # Outputs: metrics, plots, qualitative examples
├── cctv_gun_detection_research_proposal.md
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Overview

```
[CCTV Images / Frames]
    → [Data Cleaning + Annotation Verification]
    → [Classical Preprocessing: Blur / Brightness / Size Analysis]
    → [Phase 1: Carry-Object Screening]
    → [Phase 2: Weapon vs Non-Weapon Classification]
    → [YOLOv4 Baseline | Improved YOLOv4 | Modern YOLO]
    → [Quantitative Evaluation + Failure Analysis]
    → [Medium Article + YouTube Video + Final Report]
```

---

## 📚 References

1. Salazar González, J. L., Zaccaro, C., Álvarez-García, J. A., Soria-Morillo, L. M., & Sancho Caparrini, F. (2020). *Real-time gun detection in CCTV: An open problem.* Neural Networks. https://doi.org/10.1016/j.neunet.2020.09.013
2. Wang, G. et al. (2023). *Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLO v4.* Digital Signal Processing.
3. Bochkovskiy, A. et al. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection.* arXiv:2004.10934.
4. Ultralytics. *YOLO Documentation.* https://docs.ultralytics.com

---

*SafeSight Research · Insper Computer Vision · 2026*
