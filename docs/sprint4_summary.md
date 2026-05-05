# Sprint 4 Summary — Threshold Calibration

The Sprint 4 objective was to implement and evaluate a two-phase detection formulation for CCTV gun detection. The implemented pipeline detects people, classifies each detected person crop as `carry` or `no_carry`, and runs the YOLO26n weapon detector only on crops classified as `carry`.

The first two-phase run used the classifier threshold selected during training: `0.85`. This configuration was too conservative. It detected `2785` people but filtered out `2590` of them, sending only `195` candidates to Stage 2. As a result, the two-phase pipeline achieved recall `0.0040` and F1 `0.0074`.

To test whether Stage 1 was the main bottleneck, we ran an additional experiment lowering the carry/no_carry threshold from `0.85` to `0.50`. This reduced the number of filtered candidates from `2590` to `2051` and increased the number of candidates passed to Stage 2 from `195` to `734`. Final weapon detections also increased from `118` to `357`.

Quantitatively, the threshold change improved the two-phase pipeline: true positives increased from `6` to `24`, recall increased from `0.0040` to `0.0159`, and F1 increased from `0.0074` to `0.0257`. This confirms that the original threshold was one of the main causes of the recall collapse.

However, the calibrated two-phase model still does not outperform the single-stage YOLO26n baseline. The single-stage model achieved precision `0.4325`, recall `0.1758`, and F1 `0.2500`, while the threshold-calibrated two-phase model achieved precision `0.0672`, recall `0.0159`, and F1 `0.0257`.

Therefore, the final recommendation remains to use the single-stage YOLO26n as the current main model. The two-phase pipeline is useful as a Sprint 4 ablation experiment: lowering the threshold improves the pipeline, but the staged architecture still propagates errors from Stage 1 and remains weaker than the baseline for this dataset.
