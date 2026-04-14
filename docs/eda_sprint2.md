# EDA Sprint 2 - Main Dataset

**Project:** Reproducing Improved YOLOv4 for Real-Time CCTV Gun Detection  
---

## Objective

This document summarizes a first exploratory data analysis of the main CCTV dataset used in the project. The goal of this EDA is to verify the scale of the dataset, identify the class distribution, inspect bounding-box size patterns, and register a few practical observations that may affect the experimental design.

---

## Dataset Structure

```text
data/cctv_mock_attack/Images/
```

Inside this directory, each sample appears as a `.jpg` image paired with an `.xml` annotation file. The camera source is encoded in the filename prefix, such as `Cam1-`, `Cam5-`, or `Cam7-`. This means that camera-specific grouping must be inferred from filenames rather than from the directory structure.

---

## 1. Number of Images and Annotations

The dataset contains **5,149 images** in total, which matches the count described in the project README. The number of XML files also matches the image count, indicating that every frame has a corresponding annotation file.

| Camera | Images | XML annotations | Share of total images |
|--------|--------|-----------------|-----------------------|
| Cam1 | 607 | 607 | 11.79% |
| Cam5 | 1,031 | 1,031 | 20.02% |
| Cam7 | 3,511 | 3,511 | 68.19% |
| **Total** | **5,149** | **5,149** | **100%** |

### Interpretation

- **Cam7 dominates the dataset**, contributing more than two-thirds of all frames.
- This creates an important imbalance at the camera level, which may bias the model toward the visual conditions of Cam7 unless splits are designed carefully.
- Because each image has a matching XML file, the main data integrity check at this stage looks good.

---

## 2. Classes and Distribution

The XML annotations expose **three weapon classes**:

| Class | Bounding boxes | Percentage |
|-------|----------------|------------|
| Handgun | 1,714 | 62.99% |
| Short_rifle | 797 | 29.29% |
| Knife | 210 | 7.72% |
| **Total** | **2,721** | **100%** |

### Interpretation

- The dataset is **strongly dominated by handguns**.
- `Short_rifle` appears often enough to support analysis, but much less than `Handgun`.
- `Knife` is clearly underrepresented and may be difficult to model reliably without class balancing, class merging, or careful evaluation.

---

## 3. Distribution by Camera and Class

The class distribution is not uniform across cameras.

| Camera | Class | Count |
|--------|-------|-------|
| Cam1 | Handgun | 260 |
| Cam1 | Short_rifle | 220 |
| Cam1 | Knife | 13 |
| Cam5 | Handgun | 1,104 |
| Cam5 | Short_rifle | 409 |
| Cam5 | Knife | 173 |
| Cam7 | Handgun | 350 |
| Cam7 | Short_rifle | 168 |
| Cam7 | Knife | 24 |

### Interpretation

- **Cam5 contains the largest number of weapon annotations**, especially for `Handgun`.
- **Knife examples are concentrated mostly in Cam5**, which may produce camera-specific bias for that class.
- Although Cam7 has the most frames overall, it does not contain the largest number of annotated objects. This suggests that many Cam7 frames may have no target object or fewer annotated instances.

---

## 4. Positive Frames vs Total Frames

The parsed annotations contain **2,721 bounding boxes** distributed across **1,534 unique annotated image files**.

This means:

- only about **29.79%** of all frames contain at least one annotated object;
- the remaining frames are likely negative examples or frames without visible labeled weapons;
- negative frames may be useful for learning background discrimination, but they also increase class sparsity.

---

## 5. Bounding-Box Size Analysis

Bounding-box statistics show that the targets are very small relative to the full image.

| Metric | Value |
|--------|-------|
| Total bounding boxes | 2,721 |
| Average box width | 45.25 px |
| Average box height | 66.24 px |
| Average box area | 0.161% of image area |
| Minimum box area | 0.0035% of image area |
| Maximum box area | 3.4261% of image area |

### Size bins

| Size bin | Count |
|----------|-------|
| tiny (<0.10%) | 1,299 |
| small (0.10%-0.50%) | 1,330 |
| medium (0.50%-1.00%) | 66 |
| large (>1.00%) | 26 |

### Interpretation

- The dataset is overwhelmingly composed of **tiny and small objects**.
- **96.99%** of all bounding boxes fall below **0.50% of the image area**.
- This confirms that **small-object detection is a central difficulty** in this project, exactly as expected for CCTV imagery.
- Any model evaluation should therefore pay attention not only to global mAP, but also to performance under small-scale targets.

---

## 6. Visual Examples

Example images and their paired XML files were copied to:

```text
results/eda/examples/
```

These examples include random samples from Cam1, Cam5, and Cam7 and can be used for quick qualitative inspection of:

- scene composition;
- annotation style;
- visible scale differences between cameras;
- potential issues such as occlusion, motion blur, or low visibility.

---

## 7. Main EDA Conclusions

This first EDA suggests five important conclusions for Sprint 2:

1. The dataset is structurally usable, but the actual organization differs from the structure assumed in the README. The pipeline should therefore parse filenames and XML files directly from a shared `Images/` folder.
2. The dataset is imbalanced at both the **camera level** and the **class level**. Cam7 dominates the number of frames, while `Handgun` dominates the number of labeled objects.
3. `Knife` is underrepresented, which may limit reliable multi-class performance unless the class setup is simplified or carefully balanced.
4. Most labeled objects are **extremely small**, confirming that this is a genuine small-object detection problem rather than a standard object-detection benchmark.
5. Because many frames appear to be negative, the split strategy should be designed carefully to preserve both camera diversity and class diversity while avoiding leakage across highly similar sequences.

---

## 8. Short Comparison: Main Dataset vs Auxiliary Roboflow Datasets

The main CCTV dataset is more suitable than the reviewed Roboflow alternatives for the core experiments.

| Criterion | Main CCTV dataset | Roboflow auxiliary datasets |
|----------|-------------------|-----------------------------|
| Visual domain | real surveillance footage | mixed and sometimes uncertain |
| Quantity | 5,149 images | 1,009 to 9,404 images depending on dataset |
| Class structure | explicit weapon classes from XML annotations | unstable or partially corrupted in some cases |
| Annotation quality | academic benchmark, more consistent | variable and often weakly documented |
| Risk of bias | camera imbalance and class imbalance | label inconsistency, domain mismatch, taxonomy noise |
| Best role in project | main benchmark | auxiliary comparison only |

### Interpretation

The main dataset has its own limitations, especially camera imbalance and very small targets, but it remains methodologically stronger than the reviewed Roboflow datasets. The Roboflow alternatives introduce extra risks such as unclear taxonomy, weak public documentation, and potentially mixed visual domains. For that reason, they should be used only for auxiliary comparison or qualitative discussion, not as the primary benchmark.

---

## Files Used

This note was prepared from the following generated CSV files:

- `results/eda/camera_summary.csv`
- `results/eda/class_distribution.csv`
- `results/eda/camera_class_distribution.csv`
- `results/eda/bbox_summary.csv`
- `results/eda/bbox_size_bins.csv`
- `results/eda/bbox_records.csv`
