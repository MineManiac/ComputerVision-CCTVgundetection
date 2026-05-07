# Sprint 4 - Two-Phase Dataset Summary

## Split summary

| split   |   images_processed |   images_with_weapon |   gt_weapon_boxes |   person_detections |   hold_crops |   no_hold_crops |   stage0_miss_images |   stage0_missed_weapon_boxes |
|:--------|-------------------:|---------------------:|------------------:|--------------------:|-------------:|----------------:|---------------------:|-----------------------------:|
| train   |               3683 |                  653 |               904 |                4653 |          885 |            3025 |                   48 |                           81 |
| val     |                435 |                   64 |                94 |                 620 |           89 |             407 |                    5 |                           12 |
| test    |               1031 |                  803 |              1513 |                2785 |         1466 |            1093 |                   42 |                          118 |

## Labeling rule

- `hold`: the center of at least one ground-truth `weapon` box falls inside a detected person box.
- `no_hold`: detected person box with no ground-truth `weapon` center inside it.
- `knife` is ignored exactly as in Sprint 3.
