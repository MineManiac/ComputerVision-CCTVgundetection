# Sprint 2 - Split Summary

## Strategy

- Test split fixed as all images from `cam5`
- Train/val pool built from `cam1` and `cam7`
- Validation built with grouped chunks and positive/negative stratification
- Validation fraction: `0.10`
- Group chunk size: `15` images
- Random seed: `42`

## Notes

- This split strategy is aligned with the real CCTV setup described in the reference paper, where `Cam1` and `Cam7` are used for training and `Cam5` is used for testing. :contentReference[oaicite:2]{index=2}
- `total_boxes` below refers to the original Pascal VOC annotations before YOLO remapping.
- In the YOLO pipeline, `handgun` and `short_rifle` are mapped to `weapon`, while `knife` is excluded from labels.

## Totals by split

| split   | images | positive_images | total_boxes |
|:--------|------:|----------------:|------------:|
| train   | 3683  | 653             | 937         |
| val     | 435   | 64              | 98          |
| test    | 1031  | 803             | 1686        |

## Per-camera summary

| split | camera_id | images | positive_images | total_boxes |
|:------|:----------|------:|----------------:|------------:|
| train | cam1      | 532   | 248             | 426         |
| train | cam7      | 3151  | 405             | 511         |
| val   | cam1      | 75    | 37              | 67          |
| val   | cam7      | 360   | 27              | 31          |
| test  | cam5      | 1031  | 803             | 1686        |