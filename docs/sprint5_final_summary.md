# Sprint 5 - Final Single-Stage Analysis Summary

## Objective

Sprint 5 focused on improving the current best practical model for CCTV weapon detection. After Sprint 4, the two-phase pipeline had shown an important limitation: the person/carry filtering stage removed too many candidate regions before the weapon detector could run. Because of that, Sprint 5 returned to the one-stage detector and tested whether a higher input resolution could improve detection of small weapons.

The main experiment retrained YOLO26n using `imgsz=960`, keeping the same dataset split and the same `weapon` class policy used in the previous sprints.


## Final Model Decision

The final one-stage model for Sprint 5 is:

```text
runs/single_stage/yolo26n_img9604/weights/best.pt
```

Earlier `img960` attempts exist in the `runs/single_stage/` directory, but `yolo26n_img9604` is the complete high-resolution run to report. The earlier `yolo26n_img960` sweep should not be used as the final Sprint 5 result because that run stopped much earlier and underperformed on the test split.

## Quantitative Result

The Cam5 test split was evaluated using a threshold sweep. This avoids judging the model only at a fixed confidence threshold and makes the operating point explicit.

| Model | Best threshold | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n baseline, `imgsz=640` | 0.20 | 290 | 512 | 1223 | 0.3616 | 0.1917 | 0.2505 |
| YOLO26n high-res, `imgsz=960` | 0.10 | 310 | 308 | 1203 | 0.5016 | 0.2049 | 0.2909 |

The high-resolution model improved the result in three useful ways:

- true positives increased from `290` to `310`
- false positives decreased from `512` to `308`
- F1 increased from `0.2505` to `0.2909`

The recall improvement is modest, but meaningful for this dataset. The false-positive reduction is also important because the model became more precise while still finding more true weapons.

## Visual Evidence

The Sprint 5 notebook includes visual examples generated from the saved predictions of the `img9604` run:

- successful detection: `docs/sprint5_figures/example_success_img9604.jpg`
- partial success: `docs/sprint5_figures/example_partial_img9604.jpg`
- false positive: `docs/sprint5_figures/example_false_positive_img9604.jpg`
- missed weapon: `docs/sprint5_figures/example_miss_img9604.jpg`

These examples are useful because they show that the model can detect small weapons in some cases, but still fails when weapons are heavily occluded, small, or visually ambiguous.

## One-Stage vs Two-Phase Visual Comparison

Additional side-by-side examples are available in:

```text
docs/sprint5_one_stage_vs_two_phase_examples.md
```

These figures compare the same test frames using:

- ground truth only
- one-stage YOLO26n `img9604`
- the available two-phase pipeline predictions

The comparison supports the current recommendation. The one-stage model is stronger overall, but the two-phase pipeline still provides useful failure-analysis evidence. In one example, the two-phase pipeline detects one weapon missed by the one-stage model; however, across the quantitative evaluation it still has much lower recall and F1.

## Person Crop Quality Check

The person-crop step was also inspected visually using saved crops and metadata from:

```text
data/interim/two_phase/
```

The crop quality examples are documented in:

```text
docs/sprint5_two_phase_crop_quality_check.md
```

These examples show that many crops are reasonable, especially when the detected person box includes the hands and nearby weapon region. However, they also expose the main weakness of the two-phase design: some weapons are outside the selected person crop, or the person detector does not produce a box that contains the weapon center. Those cases become unrecoverable for later stages.

This reinforces the recommendation that person crops should be used as an auxiliary second pass, not as a hard gate that replaces full-frame weapon detection.

## Interpretation

The Sprint 5 result supports the following conclusion:

> Increasing the YOLO input resolution improved the one-stage detector and produced the best model observed so far in the project, but the task remains difficult because the Cam5 test split contains small, distant, and partially occluded weapons.

This conclusion is consistent with the reference paper's framing of real CCTV weapon detection as an open problem. The model improvement is real, but the remaining false negatives show that resolution alone is not enough to solve the task.

## Current Recommendation

For the next checkpoint or discussion with the professor, the recommended project position is:

1. Use YOLO26n `img9604` as the current main model.
2. Keep the Sprint 4 two-phase pipeline as an ablation, not as the final model.
3. Present the threshold sweep instead of only a single fixed-threshold metric.
4. Emphasize that the improvement came from adapting the paper's small-object motivation to a YOLO-based pipeline.
5. Treat crop-assisted detection as the next research direction.

## Next Step After Feedback

The most promising next direction is not the previous carry/no_carry gate. A better extension would be:

```text
full image YOLO
  + person detection
  + YOLO on padded person crops
  + merge detections with NMS
```

This would keep the full-frame detector as a safety net while using person crops to make small weapons appear larger to the detector. It follows the professor's suggestion and avoids the biggest Sprint 4 failure mode: rejecting true weapon candidates before the weapon detector runs.
