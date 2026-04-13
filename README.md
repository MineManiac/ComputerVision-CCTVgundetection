# CCTV Gun Detection - YOLOv4 Reproduction Study

**Course:** Computer Vision - Insper  
**Team:** SafeSight Research  
**Members:** Matheus Ribeiro Barros, Fernanda de Oliveira Pereira

> Reproducing and evaluating *Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLOv4* (Wang et al., DSP 2023) using public datasets, a modern YOLO comparison, and robustness analysis under adverse CCTV conditions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [References](#references)

---

## Project Overview

This project is a structured reproduction and stress-test study of a 2023 paper on CCTV weapon detection. It does **not** claim algorithmic novelty. The goals are:

1. Reproduce the improved YOLOv4 pipeline described in the paper using public data.
2. Compare it against a standard YOLOv4 baseline and a modern YOLO-family detector such as YOLO11.
3. Evaluate robustness under realistic CCTV conditions such as motion blur, low light, and small object scale.
4. Explore a two-phase detection strategy to reduce false positives.

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

If `Invoke-WebRequest` works well in your environment, you can use it instead of `curl.exe`, but `curl.exe` is often more stable for large downloads on Windows.

### Option 2 - Hugging Face dataset page

Download manually from the official dataset page:  
https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset

### Expected Directory Structure

The extracted archive stores the files in a shared `Images/` folder. Each `.jpg` frame has a matching `.xml` annotation file in the same directory, and the camera is identified by the filename prefix (`Cam1-`, `Cam5-`, `Cam7-`).

```text
ComputerVision-CCTVgundetection/
|-- data/
|   |-- weapons_images_2fps.zip
|   `-- cctv_mock_attack/
|       `-- Images/
|           |-- Cam1-...jpg
|           |-- Cam1-...xml
|           |-- Cam5-...jpg
|           |-- Cam5-...xml
|           |-- Cam7-...jpg
|           `-- Cam7-...xml
|
|-- notebooks/
|-- scripts/
|-- models/
|-- results/
|-- cctv_gun_detection_research_proposal.md
`-- README.md
```

You can verify the extraction with:

```powershell
Get-ChildItem data\cctv_mock_attack
Get-ChildItem data\cctv_mock_attack\Images | Select-Object -First 10
```

Expected result:

- an `Images/` directory inside `data\cctv_mock_attack`
- paired `.jpg` and `.xml` files in that folder
- filenames starting with `Cam1-`, `Cam5-`, or `Cam7-`

> Tip: add `data/` to your `.gitignore` to avoid accidentally committing large files.

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

## Installation

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
| `opencv-python` | Image preprocessing and analysis |
| `albumentations` | Data augmentation |
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |

---

## Project Structure

```text
ComputerVision-CCTVgundetection/
|-- data/                   # Dataset (not committed)
|-- docs/                   # Planning notes, reviews, EDA summaries
|-- notebooks/              # Exploratory notebooks
|-- scripts/                # Data prep, training, evaluation scripts
|-- models/                 # Trained models and configs
|-- results/                # Metrics, plots, examples
|-- cctv_gun_detection_research_proposal.md
|-- sprint1.md
|-- sprint_planning.md
`-- README.md
```

---

## Pipeline Overview

```text
[CCTV Images / Frames]
    -> [Data Cleaning + Annotation Verification]
    -> [Classical Preprocessing: Blur / Brightness / Size Analysis]
    -> [Phase 1: Carry-Object Screening]
    -> [Phase 2: Weapon vs Non-Weapon Classification]
    -> [YOLOv4 Baseline | Improved YOLOv4 | Modern YOLO]
    -> [Quantitative Evaluation + Failure Analysis]
    -> [Medium Article + YouTube Video + Final Report]
```

---

## References

1. Salazar Gonzalez, J. L., Zaccaro, C., Alvarez-Garcia, J. A., Soria-Morillo, L. M., and Sancho Caparrini, F. (2020). *Real-time gun detection in CCTV: An open problem.* Neural Networks. https://doi.org/10.1016/j.neunet.2020.09.013
2. Wang, G. et al. (2023). *Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLO v4.* Digital Signal Processing.
3. Bochkovskiy, A. et al. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection.* arXiv:2004.10934.
4. Ultralytics. *YOLO Documentation.* https://docs.ultralytics.com

---

*SafeSight Research - Insper Computer Vision - 2026*
