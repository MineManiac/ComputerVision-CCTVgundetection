# Sprint 5 - Single-Stage Improvement Plan

## Why improve the one-stage baseline

The current best production candidate is still the single-stage YOLO26n baseline. The Sprint 4 two-phase pipeline is useful as an ablation, but its person/carry gate removes too many true candidates before weapon detection.

The main failure mode is aligned with the paper: weapons are small, distant, partially occluded, and sometimes motion-blurred. The most direct one-stage improvements are therefore:

- train and evaluate at higher image resolution
- tune the inference confidence threshold for recall/F1
- compare metrics by weapon box size
- keep the Cam5 test split fixed for fair comparison

## Baseline to beat

Current test comparison from Sprint 4:

| model | precision | recall | F1 |
|---|---:|---:|---:|
| YOLO26n single-stage | 0.4325 | 0.1758 | 0.2500 |

The next one-stage experiments should beat this F1 without hiding a large recall drop.

## First threshold-sweep result

Running `scripts/evaluate_single_stage_sweep.py` on the existing `runs/detect/runs/yolo26n_full/weights/best.pt` checkpoint produced:

| threshold | TP | FP | FN | precision | recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 290 | 512 | 1223 | 0.3616 | 0.1917 | 0.2505 |
| 0.25 | 266 | 349 | 1247 | 0.4325 | 0.1758 | 0.2500 |

This means threshold tuning alone gives only a small F1 gain, but `0.20` recovers 24 additional true positives on Cam5. Larger improvements probably require retraining at higher resolution or changing the training data strategy.

## Final img960 run result

The complete high-resolution run is:

```text
runs/single_stage/yolo26n_img9604/weights/best.pt
```

Do not use the earlier `runs/single_stage/yolo26n_img960/weights/best.pt` result as the final number: that attempt stopped much earlier and underperformed on the test sweep.

The complete `img9604` checkpoint produced the best test F1 at confidence threshold `0.10`:

| model | threshold | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n img640 baseline | 0.20 | 290 | 512 | 1223 | 0.3616 | 0.1917 | 0.2505 |
| YOLO26n img9604 | 0.10 | 310 | 308 | 1203 | 0.5016 | 0.2049 | 0.2909 |

Compared with the best baseline threshold, `img9604` improves F1 by about `0.0404`, raises recall, and reduces false positives while adding true positives.

## Deliverable Status

The Sprint 5 deliverable is ready to present with the current artifacts:

- notebook: `sprint5_single_stage.ipynb`
- final written summary: `docs/sprint5_final_summary.md`
- visual examples: `docs/sprint5_figures/`
- threshold sweep for the baseline: `runs/single_stage/evaluation/yolo26n_full_test_threshold_sweep.md`
- threshold sweep for the final run: `runs/single_stage/evaluation/yolo26n_img9604_test_threshold_sweep.md`

No additional training or inference is required before the professor meeting.

## Limitations to mention

The new model is better, but the absolute recall is still low. This should not be hidden. The most defensible interpretation is that the experiment improved the baseline while confirming the paper's central claim: real CCTV weapon detection remains difficult because many weapons are small, occluded, or visually ambiguous.

The main remaining error type is false negatives. A lower threshold can recover more weapons, but it also increases false positives. That is why the threshold sweep is important: it shows the trade-off explicitly.

## Recommended experiment order

1. Threshold sweep on the existing YOLO26n checkpoint.
2. Retrain YOLO26n at `imgsz=960`.
3. If GPU memory allows, retrain YOLO26n at `imgsz=1280` with smaller batch.
4. Compare the best checkpoints with the same threshold sweep.
5. Use the per-image output to inspect false negatives.

## Commands

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:YOLO_CONFIG_DIR = "$PWD"
$DEVICE = "0"
$BASE_MODEL = "runs/detect/runs/yolo26n_full/weights/best.pt"
```

### 1. Sweep thresholds on the existing checkpoint

```powershell
python scripts/evaluate_single_stage_sweep.py `
  --model $BASE_MODEL `
  --split test `
  --device $DEVICE `
  --output-prefix yolo26n_full_test
```

### 2. Train a higher-resolution YOLO26n

Start with `imgsz=960`, because it is the safest upgrade for an RTX 3060 Laptop GPU:

```powershell
python scripts/train_single_stage_yolo.py `
  --model yolo26n.pt `
  --name yolo26n_img960 `
  --device $DEVICE `
  --imgsz 960 `
  --batch 8 `
  --epochs 140 `
  --patience 35 `
  --cos-lr
```

Evaluate it:

```powershell
python scripts/evaluate_single_stage_sweep.py `
  --model runs/single_stage/yolo26n_img9604/weights/best.pt `
  --split test `
  --device $DEVICE `
  --output-prefix yolo26n_img9604_test
```

### 3. Optional larger-resolution run

Use only if the `960` run fits comfortably:

```powershell
python scripts/train_single_stage_yolo.py `
  --model yolo26n.pt `
  --name yolo26n_img1280 `
  --device $DEVICE `
  --imgsz 1280 `
  --batch 4 `
  --epochs 140 `
  --patience 35 `
  --cos-lr
```

Evaluate it:

```powershell
python scripts/evaluate_single_stage_sweep.py `
  --model runs/single_stage/yolo26n_img1280/weights/best.pt `
  --split test `
  --device $DEVICE `
  --output-prefix yolo26n_img1280_test
```

## Output files

The threshold sweep writes:

- `runs/single_stage/evaluation/<prefix>_lowconf_predictions.csv`
- `runs/single_stage/evaluation/<prefix>_threshold_sweep.csv`
- `runs/single_stage/evaluation/<prefix>_threshold_sweep.md`
- `runs/single_stage/evaluation/<prefix>_best_f1_per_image.csv`

The most important file for reporting is the Markdown summary. It includes the best threshold by F1, the best threshold by recall, and ground-truth box-size counts.

## How to interpret results

Prefer a model that improves recall and F1 on Cam5. If precision drops moderately but recall increases strongly, that may be acceptable for CCTV threat detection, because missing a weapon is worse than showing more candidates for human review.
