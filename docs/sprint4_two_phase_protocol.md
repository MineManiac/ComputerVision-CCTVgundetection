# Sprint 4 - Two-Phase Pipeline Protocol

## Objective

Sprint 4 extends the Sprint 3 `YOLO26n` baseline into a two-phase pipeline:

1. detect `person` candidates in the full CCTV frame
2. classify each expanded person crop as `hold` or `no_hold`
3. run the weapon detector only on crops approved by the hold classifier
4. compare the final image-level detections against the original single-stage baseline

The goal is not to replace the Sprint 3 baseline immediately. The goal is to measure whether the staged design reduces false positives without causing an unacceptable recall drop.

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
- for each detected person box, check whether the **center** of at least one ground-truth `weapon` box falls inside that person box
- expand the crop around the detected person to preserve more local context
- if yes, label the crop as `hold`
- otherwise, label the crop as `no_hold`

Additional constraints:

- if an image contains a ground-truth weapon but no detected person contains its center, the image is recorded as a **Stage 0 miss**
- no synthetic person boxes are created
- negative crops are capped per image to control imbalance while preserving real examples

## Models used

- **Stage 0 / person detector:** `yolo11n.pt` by default, configurable in `configs/two_phase.yaml`
- **Stage 1 / hold classifier:** small CNN trained from scratch with `scripts/train_carry_classifier.py`
- **Stage 2 / weapon detector:** the Sprint 3 `YOLO26n` checkpoint supplied through `--weapon-model` or `configs/two_phase.yaml`

## Threshold policy

- person detection confidence and IoU thresholds come from `configs/two_phase.yaml`
- the Phase 1 gate uses a **permissive hold/no_hold threshold** chosen on the validation split with a recall floor policy
- the best-F1 threshold is reported separately, but the real pipeline uses the Stage 1 gate threshold
- final image-level comparison uses IoU `0.50`
- reprojected two-phase detections are deduplicated with image-level NMS

## Required outputs

The sprint is considered reproducible when the following artifacts can be generated:

- person-crop dataset under `data/interim/two_phase/crops/`
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
- if Stage 1 rejects a real `hold` crop, the correct weapon detection is also lost
- if Stage 2 is weak on cropped regions, the pipeline can lose recall even when Stage 0 and Stage 1 are correct

For that reason, Sprint 4 should be interpreted as an **ablation and extension** of the baseline, not only as a leaderboard comparison.
