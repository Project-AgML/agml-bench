# Subsampling

Settings: `configs/base_data.yaml`. Selection is seeded per dataset, so it is reproducible. Test splits are preferred, then validation, then train.

| Type | Rule |
|------|------|
| Classification | Up to 5 images per class. |
| Detection | Every class appears in at least 5 selected images (rarest class first; selected images count toward each class they contain). |
| Segmentation | Same as detection, using the classes present in each mask (up to 3,000 masks scanned). |
| Image-text-to-text | Stratified by `raw_metadata` fields: config (if several) + one primary key (`task`, `question_type`, `source`, …) + optional secondary key (`category`, `crop`, …) + question format (multiple choice / yes-no / numeric / open-ended). Round-robin across strata, at least 5 per stratum, up to 250 per dataset. Datasets with no usable fields fall back to TF-IDF k-means over the question text. |

Augmented HF configs are skipped for classification, detection and segmentation.
