# Sprint 3 Results Summary

## Objective

Sprint 3 focused on turning the prepared data pipeline from Sprint 2 into a real experimental comparison.  
The main goal was to train and compare two modern YOLO baselines under the same local project protocol:

- **YOLO11n**
- **YOLO26n**

Both models were trained using the same dataset version, the same class mapping (`weapon`), and the same split strategy defined in Sprint 2.

---

## Experimental Protocol

The experimental setup used in Sprint 3 was:

- **Train/Validation/Test split**
  - `cam1` + `cam7` -> train/validation
  - `cam5` -> test
- **Single detection class**
  - `handgun` -> `weapon`
  - `short_rifle` -> `weapon`
  - `knife` excluded from the initial baseline
- **Dataset format**
  - Pascal VOC XML annotations converted to YOLO format
- **Models compared**
  - `YOLO11n`
  - `YOLO26n`

This ensured that both runs were evaluated under the same local protocol.

---

## Main Artifacts

The main Sprint 3 result artifacts committed to the repository are:

- `results_summary/yolo11n_full_results.csv`
- `results_summary/yolo26n_full_results.csv`
- `results_summary/yolo11n_confusion_matrix.png`
- `results_summary/yolo26n_confusion_matrix.png`

These files summarize the full local runs without requiring the full `runs/` directory to be versioned.

---

## Model Comparison

### Validation-level comparison

The full local runs showed that **YOLO26n outperformed YOLO11n** under the current project protocol.

In general terms:

- YOLO26n achieved **higher validation mAP** than YOLO11n
- YOLO26n also showed a slightly better overall error profile
- The difference was not huge, but it was consistent across the main validation indicators

### Confusion matrix comparison

From the local confusion matrices:

- **YOLO11n**
  - True Positives: **57**
  - False Positives: **21**
  - False Negatives: **37**

- **YOLO26n**
  - True Positives: **59**
  - False Positives: **19**
  - False Negatives: **35**

This means that YOLO26n produced:

- slightly more correct detections
- fewer false alarms
- fewer missed detections

---

## Interpretation

The main takeaway from Sprint 3 is that **YOLO26n became the strongest local baseline** among the two tested models.

This is important because:

- Sprint 2 only validated the data pipeline and baseline setup
- Sprint 3 was the first stage with complete local runs and real comparative results
- the conclusion is now based on actual local training artifacts rather than only smoke tests or approximate validation estimates

Even though earlier Roboflow validation suggested a slight advantage for YOLO11, the **full local training runs favored YOLO26n**. This indicates that model ranking can depend on the exact training environment, augmentation settings, and evaluation protocol.

---

## Conclusion

Sprint 3 can be considered successful because it delivered:

- complete local runs for YOLO11n and YOLO26n
- comparable result artifacts
- real confusion matrices
- a clear baseline decision for the next stage

**Current decision:** carry **YOLO26n** forward as the main baseline for the next sprint.

---