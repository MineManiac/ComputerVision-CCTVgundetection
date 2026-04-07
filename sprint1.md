# Sprint Planning — CCTV Gun Detection

**Course:** Computer Vision · Insper  
**Members:** Matheus Ribeiro Barros · Fernanda de Oliveira Pereira  
**Project:** Reproducing Improved YOLOv4 for Real-Time CCTV Gun Detection

---

## Sprint Track

| Sprint | Period | Focus | Owner |
|--------|--------|-------|-------|
| Sprint 1 | Apr 07 → Apr 14 | Foundation & Data Preparation | 🔵 M + 🟣 F |
| Sprint 2 | Apr 14 → Apr 28 | Classical CV & Data Pipeline | 🔵 M + 🟣 F |
| Sprint 3 | Apr 28 → May 05 | Baselines & Midterm Checkpoint | 🔵 M + 🟣 F |
| Sprint 4 | May 05 → May 12 | Paper Reproduction & Two-Phase Pipeline | 🔵 M + 🟣 F |
| Sprint 5 | May 12 → May 21 | Failure Analysis, Communication & Delivery | 🔵 M + 🟣 F |
| ⭐ Submission | **May 21, 2026** | Report, Medium, YouTube, Repository | 🤝 Both |
| 🎤 Presentation | **May 26, 2026** | Final talk + live demo + Q&A | 🤝 Both |

---

## Responsibility Legend

| Symbol | Member |
|--------|--------|
| 🔵 M | Matheus Ribeiro Barros |
| 🟣 F | Fernanda de Oliveira Pereira |
| 🤝 Both | Joint collaboration |

---

## Sprint 1 — Foundation & Data Preparation
**Period:** Apr 07 → Apr 14

| # | Task | Owner |
|---|------|-------|
| 1.1 | Full reading of Wang et al. (2023) — annotate key architectural decisions | 🤝 Both |
| 1.2 | GitHub repository setup (folder structure, `.gitignore`, `README.md`) | 🔵 M |
| 1.3 | Download and verify primary dataset (CCTV Mock Attack, 5,149 frames) | 🔵 M |
| 1.4 | Download and inspect secondary dataset ACF (8,319 images) | 🟣 F |
| 1.5 | Roboflow Universe dataset audit (class compatibility and annotation quality) | 🟣 F |
| 1.6 | Annotation verification: YOLO format, class consistency, corrupted bboxes | 🔵 M |
| 1.7 | Initial EDA: class distribution, object size histograms, frame quality | 🟣 F |
| 1.8 | Document reproduction assumptions and expected deviations from the paper | 🟣 F |

**Deliverables:** structured repository · EDA notebook · Roboflow audit notes · assumptions document

---

## Sprint 2 — Classical CV & Data Pipeline
**Period:** Apr 14 → Apr 28

| # | Task | Owner |
|---|------|-------|
| 2.1 | Blur analysis script (Variance of Laplacian per frame) | 🔵 M |
| 2.2 | Brightness and contrast analysis per frame | 🔵 M |
| 2.3 | Object size distribution analysis (bounding box histograms) | 🟣 F |
| 2.4 | CLAHE and denoising pipeline as a preprocessing ablation | 🔵 M |
| 2.5 | Train/val/test split script with fixed seed and stratified distribution | 🔵 M |
| 2.6 | Define and document evaluation protocol (splits, metrics, IoU thresholds) | 🟣 F |
| 2.7 | Update Related Work section of the report | 🟣 F |
| 2.8 | Set up training environment (Colab / local, dependencies, GPU check) | 🔵 M |
| 2.9 | Visualizations: high-blur, low-brightness, and small-object frame examples | 🟣 F |

**Deliverables:** `preprocess.py` · `split_dataset.py` · classical analysis notebook · evaluation protocol

---

## Sprint 3 — Baselines & Midterm Checkpoint
**Period:** Apr 28 → May 05

| # | Task | Owner |
|---|------|-------|
| 3.1 | Set up and train **Standard YOLOv4** baseline | 🔵 M |
| 3.2 | Set up and train **YOLO11** (modern YOLO baseline) | 🔵 M |
| 3.3 | Implement `evaluate.py` (mAP@50, mAP@50:95, Precision, Recall, FPS) | 🔵 M |
| 3.4 | Quantitative evaluation of both baselines | 🟣 F |
| 3.5 | Comparative metrics table | 🟣 F |
| 3.6 | Inference visualizations (TP, FP, FN examples) | 🟣 F |
| 3.7 | Midterm checkpoint presentation slides | 🤝 Both |
| 3.8 | Write preliminary results section of the report | 🟣 F |

**Deliverables:** trained baselines · metrics table · checkpoint presentation slides

---

## Sprint 4 — Paper Reproduction & Two-Phase Pipeline
**Period:** May 05 → May 12

| # | Task | Owner |
|---|------|-------|
| 4.1 | Implement **SCSP-ResNet backbone** (paper reproduction) | 🔵 M |
| 4.2 | Implement **Receptive Field Enhancement** module | 🔵 M |
| 4.3 | Implement **F-PaNet feature fusion** strategy | 🔵 M |
| 4.4 | Train Improved YOLOv4 with all reproduced modules | 🔵 M |
| 4.5 | Design and implement **Phase 1:** carry / no-carry screening | 🤝 Both |
| 4.6 | Implement **Phase 2:** weapon vs. non-weapon classification | 🔵 M |
| 4.7 | Evaluate two-phase pipeline (FP reduction vs. recall tradeoff) | 🟣 F |
| 4.8 | Compare Improved YOLOv4 against all baselines | 🟣 F |
| 4.9 | Document paper deviations and implementation decisions | 🟣 F |
| 4.10 | Ablation study: with / without each reproduced module | 🟣 F |

**Deliverables:** trained Improved YOLOv4 · two-phase pipeline evaluation · ablation table

---

## Sprint 5 — Failure Analysis, Communication & Delivery
**Period:** May 12 → May 21

| # | Task | Owner |
|---|------|-------|
| 5.1 | Failure analysis: error rate vs. blur, brightness, and object size | 🟣 F |
| 5.2 | Final figures and plots (PR curves, confusion matrix, qualitative examples) | 🟣 F |
| 5.3 | Robustness evaluation with adversarial augmentations (blur, low-light, scale) | 🔵 M |
| 5.4 | Comparison: real-only vs. synthetic-only vs. mixed-data training | 🔵 M |
| 5.5 | Critical review of Roboflow datasets (include or exclude with justification) | 🟣 F |
| 5.6 | Write and publish **Medium article** | 🟣 F |
| 5.7 | Record and edit **YouTube video** | 🤝 Both |
| 5.8 | Polish GitHub repository (documentation, notebooks, scripts) | 🔵 M |
| 5.9 | Write and review the **final report** | 🤝 Both |
| 5.10 | Prepare **final presentation** slides | 🤝 Both |

**Deliverables:** failure analysis · Medium article · YouTube video · final report · polished repository

---

## Submission & Presentation Checklists

### Submission — May 21, 2026

| # | Item | Owner | Done |
|---|------|-------|------|
| E.1 | GitHub repository organized and fully documented | 🔵 M | ☐ |
| E.2 | EDA and analysis notebooks reviewed and clean | 🟣 F | ☐ |
| E.3 | Training and evaluation scripts functional and commented | 🔵 M | ☐ |
| E.4 | Final results saved in `results/` (metrics, plots, examples) | 🤝 Both | ☐ |
| E.5 | Medium article published | 🟣 F | ☐ |
| E.6 | YouTube video published | 🤝 Both | ☐ |
| E.7 | Final report submitted (PDF) | 🤝 Both | ☐ |
| E.8 | Medium and YouTube links added to README | 🔵 M | ☐ |
| E.9 | `requirements.txt` updated and environment tested from scratch | 🔵 M | ☐ |
| E.10 | Cross-review of all written material | 🟣 F | ☐ |

### Presentation — May 26, 2026

| # | Item | Owner | Done |
|---|------|-------|------|
| A.1 | Final slides assembled (intro, method, results, conclusion) | 🤝 Both | ☐ |
| A.2 | Full rehearsal with split speaking roles | 🤝 Both | ☐ |
| A.3 | Live demo ready (inference on a sample CCTV video) | 🔵 M | ☐ |
| A.4 | Answers prepared for likely Q&A questions | 🟣 F | ☐ |
| A.5 | Slides backup as PDF with shared link | 🟣 F | ☐ |

---

*SafeSight Research · Insper Computer Vision · 2026*
