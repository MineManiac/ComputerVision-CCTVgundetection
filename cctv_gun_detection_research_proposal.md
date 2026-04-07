# Project Proposal Template

**Course:** Computer Vision  
**Team Name:** SafeSight Research  
**Team Members:** Matheus Ribeiro Barros, Fernanda  
**Date:** 12/03/2026

---

## 1. Project Title

*Reproducing Improved YOLOv4 for Real-Time CCTV Gun Detection Under Adverse Surveillance Conditions, with Modern YOLO Comparison*

---

## 2. Problem Statement (0.5 page)

### 2.1 What problem are you solving?

This project addresses the problem of **reproducing and critically evaluating a recent computer vision paper for weapon detection in CCTV imagery**. Specifically, the chosen paper is *Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLO v4* (Digital Signal Processing, 2023). The original work proposes an improved YOLOv4-based detector tailored to surveillance conditions, where weapons are often small, partially occluded, low-resolution, and embedded in cluttered backgrounds.

The goal of this project is not to build a commercial security product, but to **study, implement, and analyze** the paper’s main ideas under a public and reproducible setup. The project will focus on reproducing the paper’s key architectural and training ideas as closely as possible using open datasets, then measuring how well those ideas hold under difficult conditions such as blur, low light, and small object scale. In addition, following instructor feedback, the project will also test whether a **more recent YOLO detector** can serve as a stronger modern baseline than YOLOv4, while still preserving the paper reproduction as the central reference point.

### 2.2 Why is this important?

CCTV weapon detection is a challenging and socially relevant problem because surveillance footage typically contains poor lighting, low contrast, long viewing distance, motion blur, and very small target objects. These factors make standard object detectors less reliable. Reproducing a recent paper in this area is important for two reasons.

First, it helps verify whether the reported improvements are reproducible outside the authors’ exact training environment. Second, it creates a clearer and more accessible explanation of a difficult research topic for a broader audience through the required Medium article and YouTube video. This project therefore contributes both technically, through structured reproduction and ablation, and educationally, through research communication.

A second reason for extending the proposal is that the field has evolved since YOLOv4. A comparison against a more recent YOLO model can help answer whether the gains claimed by a CCTV-specific YOLOv4 variant still matter when compared to newer general-purpose detectors.

### 2.3 Who are the users/beneficiaries?

The main beneficiaries are:

- Computer vision students who want an accessible explanation of a recent surveillance-detection paper
- Researchers interested in reproducibility for small-object detection in CCTV
- Developers and practitioners working with safety or surveillance imagery
- Instructors or readers who want a clear public-facing summary of the method, assumptions, and limitations

---

## 3. Related Work (0.5 page)

### 3.1 Existing Solutions

| Approach | Key Features | Limitations |
|----------|-------------|-------------|
| Classical ROI + hand-crafted features | Simple pipeline, interpretable, low compute | Weak performance on small objects and complex backgrounds |
| Standard YOLOv4 / one-stage detectors | Real-time object detection, strong baseline | Struggles with very small and low-quality CCTV targets |
| More recent YOLO families (e.g., YOLO11 or similar) | Stronger modern baselines, easier training pipelines, better tooling | Not specialized for CCTV weapon-detection conditions by default |
| Improved CCTV-specific YOLO variants | Better feature extraction and multi-scale fusion for surveillance | Harder to reproduce; may depend on custom training details and dataset combinations |

### 3.2 Key Papers/Projects

1. **Real-time gun detection in CCTV: An open problem** - Salazar-González et al. - 2020
   - Key contribution: introduced an open CCTV gun-detection benchmark and highlighted the difficulty of weapon detection in surveillance conditions.
   - Limitations: the problem remains difficult because of occlusion, small object size, and domain shift.
   - Relevance to your project: provides the public dataset and baseline framing for the reproduction study.

2. **Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLO v4** - Wang et al. - 2023
   - Key contribution: proposed an improved YOLOv4-based detector for CCTV weapon detection using SCSP-ResNet, receptive-field enhancement, F-PaNet, and dataset-combination strategies.
   - Limitations: exact implementation details and training conditions may be difficult to reproduce perfectly from the paper alone.
   - Relevance to your project: this is the main paper being reproduced and analyzed.

3. **YOLOv4: Optimal Speed and Accuracy of Object Detection** - Bochkovskiy et al. - 2020
   - Key contribution: established YOLOv4 as a strong real-time object detection baseline with a practical bag of freebies and specials.
   - Limitations: not specifically designed for CCTV weapon detection or tiny-object robustness.
   - Relevance to your project: serves as the paper-aligned baseline architecture against which the paper’s modifications will be evaluated.

4. **Recent YOLO-family implementations (e.g., YOLO11 or similar)**
   - Key contribution: provide a modern and practical benchmark for current real-time detection pipelines.
   - Limitations: improvements may come from overall ecosystem progress rather than CCTV-specific design.
   - Relevance to your project: allows a fairer contemporary comparison, helping determine whether the paper’s YOLOv4-based improvements remain competitive.

### 3.3 How will your approach differ?

This project does not claim algorithmic novelty. Instead, its contribution is a **structured research reproduction and stress-test study**. The planned work differs from the original paper in four ways:

- it uses a **fully public dataset setup** so that experiments are easier to reproduce;
- it explicitly evaluates robustness under **blur, low-light, and small-object conditions**;
- it compares the paper’s YOLOv4-based setup against a **more recent YOLO-family baseline**;
- it explores a **two-phase detection formulation**, where the system first identifies whether a person appears to be carrying a relevant object and only then predicts whether the object is a weapon.

In addition, the project will perform a brief **critical inspection of Roboflow-hosted community datasets** to determine whether they are useful as auxiliary data or whether they introduce label inconsistency and evaluation bias.

---

## 4. Proposed Approach (1 page)

### 4.1 System Overview

The proposed study will follow the pipeline below:

```text
[CCTV Images / Frames]
    → [Data Cleaning + Annotation Verification + Roboflow Dataset Audit]
    → [Classical Preprocessing / Adverse-Condition Analysis]
    → [Person / Candidate Detection]
    → [Phase 1: Carrying-Object vs No-Carry Screening]
    → [Phase 2: Weapon vs Non-Weapon Decision]
    → [YOLOv4 Baseline + Improved YOLOv4 Reproduction + Modern YOLO Comparison]
    → [Quantitative Evaluation + Failure Analysis]
    → [Medium Article + YouTube Video + Final Report]
```

The project will first reproduce a strong YOLOv4 baseline. After that, the improvements described in the target paper will be added incrementally, as closely as possible given the public information available. In parallel, a more recent YOLO-family detector will be trained as a modern comparison point. The project will then compare baseline versus improved model performance and analyze behavior under adverse surveillance conditions.

Following instructor feedback, part of the study will also test a **two-phase detection strategy**. Instead of directly detecting weapons in a single step, one version of the system will first determine whether the detected person appears to be carrying a salient handheld object or suspicious item, and only then classify the candidate as weapon or non-weapon. This may reduce false positives in cluttered CCTV scenes.

### 4.2 Classical Computer Vision Components

**What classical CV techniques will you use and why?**

- **Image Quality Analysis:**
  - Variance of Laplacian to estimate blur
  - Brightness and contrast statistics to identify low-light frames
  - Object-size distribution analysis to characterize small-object difficulty

- **Preprocessing / Ablation Components:**
  - Histogram equalization or CLAHE for low-light normalization
  - Resizing and aspect-ratio-preserving padding for detector input consistency
  - Optional denoising to evaluate whether simple preprocessing helps detection

- **Failure Analysis Tools:**
  - Bounding-box size histograms
  - Error analysis by brightness, blur, and apparent weapon size
  - Visualization of false positives and false negatives by scene condition

These components are not the core detector, but they are important for understanding *why* the detector succeeds or fails in CCTV conditions.

### 4.3 Deep Learning Components

**What neural network architectures will you use?**

- **Paper-aligned baseline architecture:** Standard YOLOv4
- **Modern comparison architecture:** a more recent YOLO-family detector (e.g., YOLO11 or equivalent accessible implementation)
- **Target reproduction:** Improved YOLOv4 as described in the paper
- **Key components to reproduce:**
  - SCSP-ResNet backbone modification
  - receptive field enhancement module
  - F-PaNet feature fusion strategy
  - pruning / streamlined neck where feasible
- **Training setup:** transfer learning with real + synthetic CCTV data combinations
- **Implementation strategy:** reproduce the method as closely as possible in an accessible framework, while documenting any deviations from the paper
- **Augmentation:** scaling, flips, color jitter, blur, noise, and low-light style perturbations
- **Alternative detection formulation:** evaluate a two-phase pipeline in which a first detector or screening stage narrows the search to people carrying a relevant object, followed by a second weapon/non-weapon detection stage

### 4.4 Integration Strategy

**How will you combine classical and deep learning approaches?**

The deep learning detector is the main system being studied, while classical CV will support four functions:

1. **preprocessing ablations**, such as CLAHE and simple denoising before detection;
2. **data characterization**, such as blur and brightness analysis;
3. **failure interpretation**, helping explain in what visual conditions the reproduced model fails;
4. **two-phase screening support**, helping analyze whether the first-stage “carrying something” filter reduces false positives before the final weapon decision.

This integration keeps the project aligned with the course template while still maintaining a clear research focus.

### 4.5 Technical Challenges

**What technical challenges do you anticipate?**

1. **Incomplete implementation detail in the paper** - Carefully document every reproduction assumption and any deviations from the published description.
2. **Small-object detection difficulty** - Evaluate performance as a function of object size and use the paper’s multi-scale ideas as the main point of analysis.
3. **Dataset domain shift** - Compare real and synthetic splits separately and in combination.
4. **Limited compute for repeated experiments** - Prioritize the baseline, one recent YOLO comparison, and a subset of the paper’s modules if full reproduction becomes too expensive.
5. **Label or split inconsistencies in public data** - Verify annotations and create a transparent train/validation/test split protocol.
6. **Two-phase pipeline error propagation** - A missed detection in the first stage can prevent correct classification in the second stage, so this design must be evaluated carefully.
7. **Roboflow dataset variability** - Community datasets may require relabeling, class merging, or exclusion if definitions are inconsistent.

---

## 5. Dataset (0.5 page)

### 5.1 Data Source

**Where will you get your data?**

- **Primary Source:** US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset
- **Source Type:** Official public academic dataset
- **Size:** 5,149 real CCTV frames plus synthetic Unity-based splits with 500, 1,000, and 2,500 images
- **License:** CC BY-NC 4.0 / free for academic research with citation

- **Secondary Source (external validation):** ACF: An Armed CCTV Footage Dataset
- **Source Type:** Public academic CCTV weapon dataset released with paper/code
- **Size:** 8,319 CCTV images across pistol and knife scenarios
- **License / Access:** public academic release via authors' GitHub as reported in the paper

- **Auxiliary exploratory source:** selected Roboflow Universe CCTV/weapon datasets
- **Source Type:** Community-curated public datasets/models
- **Planned role:** critical inspection and possible auxiliary comparison only, not automatic inclusion in the main benchmark

The project will use the public open-problem dataset as the **main dataset** because it is directly aligned with CCTV gun detection, includes both real CCTV frames and synthetic Unity data, and matches the target paper's interest in dataset combination and training-scheme analysis. The ACF dataset will be used as a **secondary external evaluation dataset** when feasible, because it offers a larger CCTV-style pool for testing generalization beyond the main benchmark.

Roboflow-hosted datasets will be examined as an **auxiliary source of comparison**, especially to check whether newer community datasets provide cleaner annotations, more modern formatting, or more diverse scenes. However, they will only be incorporated if their class definitions, visual domain, and annotation quality are compatible with a fair evaluation protocol.

### 5.2 Data Statistics

| Split | # Images | # Classes | Image Size |
|-------|----------|-----------|------------|
| Training (planned, main dataset) | ~6,400 | 3 weapon classes or merged weapon setting | variable, resized for training |
| Validation (planned, main dataset) | ~1,350 | 3 weapon classes or merged weapon setting | variable, resized for training |
| Testing (planned, main dataset) | ~1,350 | 3 weapon classes or merged weapon setting | variable, resized for training |
| External Test (ACF, optional) | up to 8,319 | pistol / knife or mapped weapon setting | 1920×1080 originals, resized for testing |
| Roboflow Auxiliary Check (optional) | TBD after audit | TBD after audit / possible merged setting | variable |

**Note:** the exact split will be finalized after annotation verification and class cleaning. The main pool is expected to include the 5,149 real CCTV frames plus the 4,000 synthetic Unity images, while ACF will be used only if label mapping and evaluation protocol are feasible within the semester timeline. Roboflow data, if used at all, will first go through a compatibility audit focused on taxonomy consistency and CCTV realism.

### 5.3 Data Challenges

[Describe any issues with the data:]
- Severe small-object difficulty due to CCTV distance and resolution
- Potential class imbalance across weapon categories
- Domain shift between synthetic and real images
- Possible partial occlusion, motion blur, and low illumination
- Need to confirm whether binary or multi-class detection is more reliable under the available labels
- Need to verify whether Roboflow community datasets use compatible class taxonomies or require class merging/relabeling

### 5.4 Data Preparation Plan

- [x] Download/collect raw data
- [ ] Clean and filter low-quality images
- [ ] Split into train/val/test sets
- [ ] Create annotations if needed
- [ ] Verify class distribution
- [ ] Audit Roboflow candidates for compatibility
- [ ] Create data loaders

---

## 6. Evaluation Plan (0.5 page)

### 6.1 Metrics

**What metrics will you use to measure success?**

Primary Metrics:
- **mAP@50** - appropriate because it is the standard detection metric and easier to compare with prior work
- **mAP@50:95** - appropriate because it captures localization quality more strictly across IoU thresholds

Secondary Metrics:
- **Precision / Recall / F1-score**
- **Inference time / FPS**
- **Performance by object size and image condition**
- **False-positive reduction in the two-phase setup**

### 6.2 Baseline Comparisons

**What will you compare against?**

1. **Lower-bound heuristic:** No-preprocessing baseline and simple always-negative threshold checks for sanity
2. **Classical Baseline:** Hand-crafted ROI proposal with HOG/LBP + linear SVM on cropped candidates
3. **Deep Learning Baseline:** Standard YOLOv4
4. **Modern Deep Learning Baseline:** recent YOLO-family detector (e.g., YOLO11 or similar)
5. **Target Paper Reproduction:** Improved YOLOv4 with the paper’s main modules
6. **Two-Phase Variant:** carry/no-carry screening followed by weapon vs non-weapon detection
7. **Ablation Variants:** Improved YOLOv4 without one selected module, and with/without preprocessing

### 6.3 Success Criteria

**What results would you consider successful?**

- **Minimum viable:** a working YOLOv4 baseline plus at least one reproduced improvement, with measurable detection performance on public CCTV data
- **Expected:** the improved reproduction outperforms the standard YOLOv4 baseline on at least one main detection metric or shows a clear robustness advantage under adverse conditions
- **Additional expected insight:** the modern YOLO comparison clarifies whether the paper’s gains remain meaningful against newer detectors, and the two-phase pipeline shows whether false positives can be reduced without unacceptable recall loss
- **Stretch goal:** approximate the paper’s reported relative trend while also providing a strong failure analysis and public-facing explanation via Medium and YouTube

### 6.4 Failure Case Analysis

**How will you analyze failures?**

- Confusion matrix or class-wise AP where applicable
- Visualization of false positives and false negatives
- Error rate vs. blur, brightness, and bounding-box size
- Comparison of real-only, synthetic-only, and mixed-data training
- Comparison of single-stage vs two-phase detection behavior
- Discussion of which paper components seem most important in reproduction
- Discussion of whether Roboflow-style auxiliary datasets help or hurt consistency

---

## 7. Timeline (0.5 page)

| Week | Tasks | Deliverables | Team Member |
|------|-------|--------------|-------------|
| 5 | Proposal, paper reading, dataset download | This document | Matheus + Fernanda |
| 6 | Annotation verification, split design, EDA, Roboflow audit | Clean dataset statistics + auxiliary dataset notes | Matheus + Fernanda |
| 6-7 | Classical CV analysis tools + literature mapping | Blur/brightness/size analysis scripts + related-work notes | Matheus + Fernanda |
| 7-8 | Standard YOLOv4 baseline reproduction + modern YOLO setup | Baseline models and first results | Matheus + Fernanda |
| 8 | Midterm prep | Checkpoint presentation | Matheus + Fernanda |
| 9 | **Midterm Checkpoint** | Baseline + initial reproduction results | Matheus + Fernanda |
| 9-10 | Implement paper improvements + two-phase formulation | Reproduced modules + staged pipeline prototype | Matheus + Fernanda |
| 10-11 | Ablation and robustness analysis | Comparative results | Matheus + Fernanda |
| 11-12 | Failure analysis + figure generation | Analysis plots and qualitative examples | Matheus + Fernanda |
| 12-13 | Medium draft + YouTube script | Public-facing communication draft | Matheus + Fernanda |
| 13 | Report writing | Draft report | Matheus + Fernanda |
| 14 | **Final Presentation** | Slides, results, research summary | Matheus + Fernanda |
| 14 | Final polishing | Repo, article, video, final deliverables | Matheus + Fernanda |

### Contingency Plans

**What if things don't go as planned?**

- **If dataset is insufficient:** restrict the study to binary weapon detection and/or use only the most reliable public split.
- **If approach doesn't work:** reproduce the baseline and only a subset of the paper’s modules, explicitly framing the project as a partial reproduction.
- **If falling behind:** prioritize the standard YOLOv4 baseline, one recent YOLO comparison, and a strong analysis section over implementing every module.
- **If the two-phase setup is too unstable:** treat it as an ablation/extension rather than the main system.
- **If Roboflow data is inconsistent:** keep it only as a critical comparison in the discussion section and exclude it from training.

---

## 8. Team Responsibilities

| Team Member | Primary Responsibilities | Secondary Responsibilities |
|-------------|-------------------------|---------------------------|
| Matheus Ribeiro Barros | Model implementation, training pipeline, experiment execution, metric computation, qualitative result generation | Co-writing the proposal/report/article, video presentation, dataset verification |
| Fernanda | Paper dissection, related work synthesis, experiment logging, figure/table organization, first-pass writing of proposal/report/article | Co-implementation review, ablation design, video presentation, dataset verification |

**Equal Task Division Plan:**
- Both team members will read the paper and reproduce the baseline together before splitting into implementation and analysis workstreams.
- Both will participate in dataset verification, experiment design, ablations, and final interpretation of results.
- Matheus will lead engineering-heavy tasks, while Fernanda will lead communication-heavy tasks, but each member will also contribute to the other side so that the workload remains balanced.
- The final Medium article, YouTube video, and presentation will be co-authored and co-presented.

**Communication Plan:**
- Weekly meetings: Tuesday and Friday, 18:00-19:00
- Primary communication: WhatsApp
- Code repository: GitHub
- Document sharing: Google Drive / Overleaf

---

## 9. Required Resources

### Computational Resources
- [x] GPU access (specify: Colab/Kaggle/Local/Other) - Google Colab and/or local GPU
- [x] Storage space estimate: 15-25 GB
- [x] Expected training time: 10-20 hours across main experiments

### Software/Libraries
- PyTorch
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- Pandas
- YOLOv4-compatible training framework
- Ultralytics or equivalent modern YOLO framework

### External Resources
- [ ] API access (if any)
- [ ] Cloud credits (if needed)
- [ ] Specialized tools

---

## 10. Expected Contributions

### Technical Novelty

This project does not primarily aim to propose a new model. Its expected contribution is a **transparent reproduction and robustness analysis** of a recent CCTV weapon-detection paper using public data. The project will document which parts of the paper are straightforward to reproduce, which parts are ambiguous, and how much each reproduced module contributes in practice. This can provide useful insight into reproducibility for surveillance-oriented small-object detection.

In addition, the project will contribute a **contemporary comparison point** by checking whether a newer YOLO-family detector already closes part of the gap that the paper addresses, and whether a **two-phase detection design** can reduce false positives in cluttered surveillance scenes. A smaller but still useful contribution is the **critical inspection of Roboflow-based auxiliary datasets**, discussing whether recent community datasets are genuinely helpful or whether they introduce label inconsistency and benchmarking noise.

### Practical Impact

If successful, this project can serve as a publicly understandable case study on how recent detection papers work in practice, where they fail, and what assumptions they rely on. The final Medium article and YouTube video can make a specialized surveillance-detection topic accessible to a broader audience without requiring the reader to parse the full paper alone.

### Learning Goals

**What do you hope to learn from this project?**

- **Matheus Ribeiro Barros:** paper reproduction methodology, small-object detection in surveillance settings, robust experimental design, and public communication of technical vision research.
- **Fernanda:** critical reading of modern computer vision papers, experimental reproducibility, dataset analysis for surveillance imagery, and scientific communication for general audiences.

---

## 11. References

[List all papers, datasets, and resources mentioned above in proper academic format]

1. Wang, G., Ding, H., Duan, M., Pu, Y., Yang, Z., & Li, H., "Fighting against terrorism: A real-time CCTV autonomous weapons detection based on improved YOLO v4," *Digital Signal Processing*, vol. 132, 103790, 2023.
2. Salazar-González, J. L., Zaccaro, C., Álvarez-García, J. A., Soria-Morillo, L. M., & Sancho-Caparrini, F., "Real-time gun detection in CCTV: An open problem," *Neural Networks*, vol. 132, pp. 297-308, 2020.
3. Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M., "YOLOv4: Optimal Speed and Accuracy of Object Detection," *arXiv preprint arXiv:2004.10934*, 2020.
4. Deepknowledge-US, "US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset," official dataset repository and project page, accessed 2026.
5. Ultralytics, "Ultralytics YOLO Documentation," official documentation, accessed 2026.
6. Roboflow Universe, selected public CCTV / weapon detection datasets and models, accessed 2026.

---

## Appendix (Optional)

### A. Initial Experiments

No experiments have been completed yet. The current stage is paper selection, dataset validation, and proposal preparation.

### B. Visual Examples

Expected examples include CCTV frames containing small or partially occluded weapons, with qualitative comparisons between baseline YOLOv4, modern YOLO-family detectors, and reproduced improved YOLOv4 detections.

### C. Code Snippets

No final code snippets are included at the proposal stage.

---

**Instructor Feedback Section** (Do not fill)

| Criterion | Feedback |
|-----------|----------|
| Problem clarity | |
| Approach feasibility | |
| Scope appropriateness | |
| Timeline realism | |
| **Decision** | ☐ Approved ☐ Revisions needed ☐ Not approved |

**Comments:**

---

**Submission Checklist:**

- [x] All sections completed
- [ ] Figures/diagrams included
- [x] References properly cited
- [x] Timeline is realistic
- [x] Team responsibilities clearly defined
- [x] Backup plans identified
- [x] Dataset is accessible
- [x] Success criteria are measurable
- [ ] PDF format
- [ ] Filename: `project_proposal_teamname.pdf`

**Good luck with your project!** 🚀
