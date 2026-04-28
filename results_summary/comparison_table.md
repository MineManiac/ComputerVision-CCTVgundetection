# Sprint 3 Comparison Table

| Model   | True Positives | False Positives | False Negatives | Summary |
|:--------|---------------:|----------------:|----------------:|:--------|
| YOLO11n | 57             | 21              | 37              | Strong baseline, but weaker than YOLO26n in the local full runs |
| YOLO26n | 59             | 19              | 35              | Best local baseline in Sprint 3 |

## Notes

- Both models were trained under the same local Sprint 3 protocol.
- The confusion-matrix counts above come from the committed local result artifacts.
- For detailed metric curves and per-epoch history, see:
  - `results_summary/yolo11n_full_results.csv`
  - `results_summary/yolo26n_full_results.csv`
