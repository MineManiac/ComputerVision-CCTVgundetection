# Sprint 4 - Two-Phase Pipeline Protocol

## Objective

Sprint 4 extends the Sprint 3 `YOLO26n` baseline into a two-phase pipeline:

1. detect `person` candidates in the full CCTV frame
2. generate one expanded crop for each detected person
3. run the crop-stage weapon detector on every person crop
4. compare the final image-level detections against the original single-stage baseline

The `hold` / `no_hold` classifier is still available, but only as an optional ablation gate. The default path prioritizes recall and avoids rejecting candidate crops before Stage 2 runs.

## Label policy

- Positive weapon classes remain the same as Sprint 2 and Sprint 3:
  - `handgun`
  - `short_rifle`
- These classes are mapped to the single project class `weapon`.
- `knife` remains excluded from both training labels and evaluation.

This keeps Sprint 4 directly comparable to the earlier baselines.

## Stage 1 dataset rule

Stage 1 does not use new manual person annotations.

Instead, the script `scripts/build_two_phase_dataset.py` creates person crops with the following rule:

- run a COCO-pretrained YOLO person detector on each frame
- expand each detected person box with asymmetric padding plus a minimum crop size
- for each expanded crop, mark it positive when either:
  - the center of a ground-truth `weapon` falls inside the expanded crop
  - or the crop reaches the configured `weapon` intersection-over-area threshold
- if yes, label the classifier crop as `hold`
- otherwise, label the crop as `no_hold`

Additional constraints:

- the script also exports `data/interim/two_phase/yolo_crops/`, which stores one padded crop per detected person and clipped YOLO labels for Stage 2 training
- if an image contains a ground-truth weapon but no expanded crop covers it, the image is recorded as a **Stage 0 miss**
- no synthetic person boxes are created
- negative crops are capped per image to control imbalance while preserving real examples

## Models used

- **Stage 0 / person detector:** `yolo11n.pt` by default, configurable in `configs/two_phase.yaml`
- **Stage 1 / crop export:** `scripts/build_two_phase_dataset.py`
- **Stage 1 optional gate / hold classifier:** small CNN trained from scratch with `scripts/train_carry_classifier.py`
- **Stage 2 / weapon detector:** a YOLO checkpoint trained on `data/interim/two_phase/yolo_crops/dataset.yaml`, supplied through `--weapon-model` or `configs/two_phase.yaml`

## Threshold policy

- person detection confidence and IoU thresholds come from `configs/two_phase.yaml`
- the default pipeline does **not** use the `hold/no_hold` gate
- if the optional gate is enabled, it uses a **permissive hold/no_hold threshold** chosen on the validation split with a recall floor policy
- final image-level comparison uses IoU `0.50`
- reprojected two-phase detections are deduplicated with image-level NMS

## Required outputs

The sprint is considered reproducible when the following artifacts can be generated:

- person-crop dataset under `data/interim/two_phase/crops/`
- crop-stage YOLO dataset under `data/interim/two_phase/yolo_crops/`
- metadata CSV files under `data/interim/two_phase/metadata/`
- hold classifier checkpoint and metrics under `runs/two_phase/carry_classifier/`
- two-phase predictions under `runs/two_phase/predictions/`
- comparison tables under `runs/two_phase/evaluation/`

## Comparison tables

The final evaluation should report, at minimum:

- TP
- FP
- FN
- precision
- recall
- F1-score
- detections per image
- Stage 1 rejected-person count
- delta between the two-phase pipeline and the single-stage baseline

## Interpreting failures

The staged design introduces error propagation:

- if Stage 0 misses the relevant person, Stage 2 never runs on the true candidate
- if the expanded crop still does not cover the weapon well enough, Stage 2 never sees the object properly
- if the optional gate rejects a real `hold` crop, the correct weapon detection is also lost
- if Stage 2 is weak on cropped regions, the pipeline can lose recall even when Stage 0 and the crop expansion are correct

For that reason, Sprint 4 should be interpreted as an **ablation and extension** of the baseline, not only as a leaderboard comparison.
