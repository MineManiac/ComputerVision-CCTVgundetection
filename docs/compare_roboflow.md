# Roboflow Dataset Comparison

**Project:** Reproducing Improved YOLOv4 for Real-Time CCTV Gun Detection  
**Author:** Fernanda de Oliveira Pereira  
**Date:** April 13, 2026

---

## Objective

This document compares selected Roboflow Universe datasets that appear related to CCTV weapon detection. The goal is to decide whether any of them are suitable for training, auxiliary experiments, or only qualitative comparison.

---

## Datasets Reviewed

### 1. weapon detection cctv v3 dataset

- **Link:** https://universe.roboflow.com/weapon-detection-cctv/weapon-detection-cctv-v3-dataset/dataset/1
- **Total images:** 4,424
- **Dataset split:** 82% train, 12% validation, 6% test
- **Preprocessing reported:** auto-orient, resize to 416x416, grayscale, class remapping, and class dropping
- **Main concern:** the page reports that **8 classes were remapped and 3 were dropped**, which reduces label transparency

### 2. CCTV Gun Detector

- **Link:** https://universe.roboflow.com/sense-onkyy/cctv-gun-detector-1is1p
- **Total images:** 1,009
- **Classes:** `Guns`, `Guns perspective`, `Long guns`
- **Preprocessing reported:** not clearly described on the overview page
- **Main concern:** the taxonomy mixes weapon type and viewpoint, and the dataset has no public description explaining annotation criteria

### 3. FYP - Weapon Detection in CCTV

- **Link:** https://universe.roboflow.com/weapon-detection-uzwei/fyp-weapon-detection-in-cctv-kw2mb
- **Total images:** 9,404
- **Classes shown publicly:** `gun`, `armed man`, `man`, plus malformed labels that appear to be version artifacts
- **Main concern:** class naming is clearly corrupted, which makes the dataset unreliable for controlled experiments

---

## Comparison Table

| Dataset | Size | Classes | Annotation Quality | CCTV Realism | Best Use |
|--------|------|---------|--------------------|--------------|----------|
| weapon detection cctv v3 dataset | 4,424 | unclear after remapping | medium | partial | auxiliary comparison |
| CCTV Gun Detector | 1,009 | 3 classes | low to medium | uncertain | qualitative inspection |
| FYP - Weapon Detection in CCTV | 9,404 | 6 shown, but corrupted | low | uncertain | not recommended |

---

## Short Evaluation

### weapon detection cctv v3 dataset

This is the strongest Roboflow candidate among the reviewed options because it is moderately sized and directly framed around CCTV weapon detection. However, it has significant limitations for academic use. The preprocessing pipeline is aggressive and includes grayscale conversion and class remapping. Because of this, the dataset may be useful for a limited auxiliary experiment, but it is not ideal as a core benchmark.

### CCTV Gun Detector

This dataset is easier to inspect because its class list is visible and relatively small, but the class design is not very clean. `Guns perspective` is not a stable object category, and the dataset is small for robust detector training. It may still be useful for qualitative checks or to illustrate how community datasets often mix class identity with visual viewpoint.

### FYP - Weapon Detection in CCTV

This dataset is the least reliable option despite being the largest. Its public class list contains malformed labels such as version names and separator strings, which strongly suggests taxonomy corruption. For a reproducibility-focused project, this creates too much risk. It should not be used for training or formal evaluation.

---

## Final Recommendation

The Roboflow datasets reviewed here should **not** be used as the primary training and evaluation source for the project. The safest decision is:

- keep the public academic CCTV dataset as the main benchmark;
- use Roboflow only as an auxiliary source of comparison;
- consider `weapon detection cctv v3 dataset` only for a small side experiment if time allows;
- exclude `CCTV Gun Detector` and `FYP - Weapon Detection in CCTV` from the core pipeline.

This recommendation is based mainly on three problems:

- inconsistent or unstable class taxonomies;
- weak public documentation;
- unclear annotation quality and domain consistency.

---

## Source Notes

The assessment above is based on what is publicly visible on the Roboflow Universe project and dataset pages on **April 13, 2026**. In places where the pages did not provide enough detail, the judgment about CCTV realism was treated as a cautious inference rather than a confirmed fact.

### Sources

- Roboflow Universe, **weapon detection cctv v3 dataset**: https://universe.roboflow.com/weapon-detection-cctv/weapon-detection-cctv-v3-dataset/dataset/1
- Roboflow Universe, **CCTV Gun Detector**: https://universe.roboflow.com/sense-onkyy/cctv-gun-detector-1is1p
- Roboflow Universe, **FYP - Weapon Detection in CCTV**: https://universe.roboflow.com/weapon-detection-uzwei/fyp-weapon-detection-in-cctv-kw2mb
