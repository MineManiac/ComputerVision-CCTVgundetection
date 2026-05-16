# Sprint 4 - Two-Phase Dataset Summary

## Split summary

| split   |   images_processed |   images_with_weapon |   gt_weapon_boxes |   person_detections |   raw_person_cover |   expanded_crop_cover |   hold_crops |   no_hold_crops |   carry_crops |   no_carry_crops |   yolo_crops |   yolo_positive_crops |   yolo_negative_crops |   stage0_miss_images |   stage0_missed_weapon_boxes |
|:--------|-------------------:|---------------------:|------------------:|--------------------:|-------------------:|----------------------:|-------------:|----------------:|--------------:|-----------------:|-------------:|----------------------:|----------------------:|---------------------:|-----------------------------:|
| train   |               3683 |                  653 |               904 |                5228 |                861 |                   899 |         1212 |            2994 |          1212 |             2994 |         5228 |                  1212 |                  4016 |                    4 |                            5 |
| val     |                435 |                   64 |                94 |                 667 |                 86 |                    94 |          134 |             385 |           134 |              385 |          667 |                   134 |                   533 |                    0 |                            0 |
| test    |               1031 |                  803 |              1513 |                3435 |               1474 |                  1493 |         1886 |            1183 |          1886 |             1183 |         3435 |                  1886 |                  1549 |                    9 |                           20 |

## Labeling rule

- `hold`: the expanded crop contains the weapon center or reaches the configured weapon IOA threshold.
- `no_hold`: detected person crop with no weapon match after crop expansion.
- `knife` is ignored exactly as in Sprint 3.
- `yolo_crops/` stores one padded person crop per detected person, with clipped YOLO labels for matched weapons.
