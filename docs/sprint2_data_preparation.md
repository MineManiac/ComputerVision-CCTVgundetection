# Sprint 2 Data Preparation

The current Sprint 2 pipeline is:

1. Reorganize the raw extracted dataset
2. Build a dataset manifest
3. Audit annotations and class mapping
4. Create grouped train/val/test splits
5. Convert Pascal VOC XML annotations to YOLO format
6. Run a short YOLO11n smoke test

### 1) Reorganize extracted files

If your raw download still has `.jpg` and `.xml` mixed inside `Images/`, run:

```bash
python scripts/reorganize_raw_dataset.py
```

This separates the dataset into:

- `data/raw/images/`
- `data/raw/annotations/`

### 2) Build the manifest

```bash
python scripts/build_manifest.py
```

Outputs:
- `data/interim/manifest.csv`
- `results/audit/manifest_summary.txt`

### 3) Audit annotations

```bash
python scripts/audit_annotations.py
```

This stage:
- checks the parsed annotations
- maps `handgun` and `short_rifle` to `weapon`
- excludes `knife` from the YOLO baseline labels
- summarizes class and box-size distributions

Outputs include:
- `results/audit/audit_summary.md`
- `results/audit/class_distribution_original.csv`
- `results/audit/class_distribution_mapped.csv`

### 4) Create grouped train/val/test splits

```bash
python scripts/make_splits.py
```

Current split strategy:
- `cam1` + `cam7` -> train/val
- `cam5` -> test
- validation built with grouped chunks and positive/negative stratification

Current totals:

| split | images | positive_images | total_boxes |
|:------|------:|----------------:|------------:|
| train | 3683  | 653             | 937         |
| val   | 435   | 64              | 98          |
| test  | 1031  | 803             | 1686        |

> `total_boxes` above refers to the original Pascal VOC annotations before YOLO remapping.

### 5) Convert VOC to YOLO

```bash
python scripts/voc_to_yolo.py
```

Mapping used:
- `handgun` -> `weapon` (`class 0`)
- `short_rifle` -> `weapon` (`class 0`)
- `knife` -> excluded

Current conversion summary:

| split | images_processed | images_with_weapon | negative_images | converted_boxes | excluded_boxes | skipped_invalid_boxes |
|:------|-----------------:|-------------------:|----------------:|----------------:|---------------:|----------------------:|
| train | 3683             | 653                | 3030            | 904             | 33             | 0                     |
| val   | 435              | 64                 | 371             | 94              | 4              | 0                     |
| test  | 1031             | 803                | 228             | 1513            | 173            | 0                     |

Images without valid `weapon` objects are kept with empty YOLO label files.

### 6) Run a YOLO11n smoke test

```bash
python scripts/train_yolo_smoke.py
```

This is a short sanity-check training run intended to verify that:
- the YAML config is correct
- the dataset loads properly
- labels are read correctly
- the training pipeline starts without errors

The current smoke test is configured conservatively for local execution, using CPU by default.

---