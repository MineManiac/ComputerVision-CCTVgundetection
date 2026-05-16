# Two-Phase Crop Subset Diagnostic

Use this when you want to inspect a small GT-positive subset and verify whether the person crops used by the two-phase pipeline make visual sense.

The script samples positive images from a split, runs the person detector, applies the same padded crop rule from `configs/two_phase.yaml`, resizes the crop to the classifier input size, and saves visual files for inspection.

## Default PowerShell Run

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\diagnose_two_phase_crop_subset.py `
  --split test `
  --num-images 50 `
  --output-dir results\debug\two_phase_crop_subset_test50 `
  --overwrite
```

## GPU Run

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\diagnose_two_phase_crop_subset.py `
  --split test `
  --num-images 50 `
  --device 0 `
  --output-dir results\debug\two_phase_crop_subset_test50_gpu `
  --overwrite
```

## Random Positive Subset

```powershell
python scripts\diagnose_two_phase_crop_subset.py `
  --split test `
  --num-images 50 `
  --sample-mode random `
  --seed 42 `
  --output-dir results\debug\two_phase_crop_subset_test50_random `
  --overwrite
```

## Save Every Positive Crop

By default, the script saves only the highest-confidence matched person crop per positive image. Use this if you want all matched person crops:

```powershell
python scripts\diagnose_two_phase_crop_subset.py `
  --split test `
  --num-images 50 `
  --save-all-positive-crops `
  --output-dir results\debug\two_phase_crop_subset_test50_all_crops `
  --overwrite
```

## Output

The selected output directory will contain:

- `panels/`: side-by-side view with original annotation and resized crop
- `originals/`: original image with GT weapon boxes, selected person bbox, and padded crop bbox
- `raw_crops/`: crop before resize
- `resized_crops/`: exact resized crop without overlays
- `resized_crops_annotated/`: resized crop with projected GT weapon boxes
- `diagnostic_manifest.csv`: metadata for each saved case
- `summary.md`: quick counts and legend

Color legend:

- red: GT weapon box
- cyan: detected person bbox
- yellow: padded crop box before resize
