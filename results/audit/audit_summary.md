# Sprint 2 - Annotation Audit Summary

- Total valid images: **5149**
- Total boxes (all original classes): **2721**
- Total boxes mapped to `weapon`: **2511**
- Total excluded boxes (`knife`): **210**
- Total unknown boxes: **0**
- Positive images after mapping to `weapon`: **1520**

## Weapon box sizes (COCO-style buckets)

- small: **349**
- medium: **2049**
- large: **113**

## Camera distribution

| camera_id   |   total_images |   positive_images_after_mapping |   total_boxes |   total_weapon_boxes |   total_excluded_boxes |   total_unknown_boxes |
|:------------|---------------:|--------------------------------:|--------------:|---------------------:|-----------------------:|----------------------:|
| cam1        |            607 |                             285 |           493 |                  480 |                     13 |                     0 |
| cam5        |           1031 |                             803 |          1686 |                 1513 |                    173 |                     0 |
| cam7        |           3511 |                             432 |           542 |                  518 |                     24 |                     0 |

## Original class distribution

| original_class   |   count |
|:-----------------|--------:|
| handgun          |    1714 |
| short_rifle      |     797 |
| knife            |     210 |

## Mapped class distribution

| mapped_class   |   count |
|:---------------|--------:|
| weapon         |    2511 |
| exclude        |     210 |

