# Sprint 2 - VOC to YOLO Conversion Summary

## Mapping used

- `handgun` -> `weapon` (class `0`)
- `short_rifle` -> `weapon` (class `0`)
- `knife` -> excluded from YOLO baseline labels

## Notes

- Images without valid `weapon` objects are kept in the dataset with empty YOLO label files.
- This converted dataset is intended for the YOLO11n baseline preparation in Sprint 2.

## Conversion summary by split

| split | images_processed | images_with_weapon | negative_images | converted_boxes | excluded_boxes | skipped_invalid_boxes |
|:------|-----------------:|-------------------:|----------------:|----------------:|---------------:|----------------------:|
| train | 3683             | 653                | 3030            | 904             | 33             | 0                     |
| val   | 435              | 64                 | 371             | 94              | 4              | 0                     |
| test  | 1031             | 803                | 228             | 1513            | 173            | 0                     |