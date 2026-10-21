# Benchmarking

Settings: `src/agbench/eval/prompts.yaml`, `models.yaml`, `ag_tasks.yaml`, `seg_classes.yaml`. Options and NOTA items are seeded per sample, so they are reproducible.

| Type | Rule |
|------|------|
| Classification: MCQA within | Up to 4 classes from the same dataset, then "None of the above" and "I don't know". In 30% of items the true class is removed and "None of the above" is correct. |
| Classification: MCQA across | Same, but distractors come from other datasets' classes in the same ag task. |
| Classification: OEQ | One-sentence open question chosen by ag task, graded by the judge. |
| Detection | Model returns JSON `box_2d` on a 0–1000 scale; matched to ground truth at IoU 0.5 (labels must agree when the dataset has more than one class). |
| Segmentation | Ground-truth boxes from mask connected components (≥ 0.02% of the image), then scored like detection. Masks published without class names are named in `seg_classes.yaml` after checking each value visually; datasets that cannot be named are skipped. |
| Image-text-to-text | Native question. Multiple choice gets "None of the above" and "I don't know" appended; yes/no and numeric are matched directly; open-ended is graded by the judge. |
| Crop | Per-sample metadata, then the dataset's single crop, then the crop named in the class label, then the list of dataset crops. Species ID always uses "plant". |
| Ag task | Per class: overrides in `ag_tasks.yaml` (from agtuned-vlm plus a pest/abiotic keyword scan), then the dataset's task type, then the catalog `agricultural_task`. Buckets: disease, pest_damage (incl. abiotic stress), quality_maturity, species_id, localization, other. ITT is labelled per question from its metadata. |
| Answer matching | Exact, then substring, then fuzzy similarity ≥ 0.6. Thinking is stripped; an `<answer>` tag is preferred when present. |
| Thinking | On for every model that supports it, `medium` effort. Models without thinking get a `<think>`/`<answer>` prompt. Truncated answers are reported separately. |
| Metrics | Accuracy and F1 per dataset. Classification F1 is macro over classes; detection/segmentation accuracy is TP/(TP+FP+FN) per image and F1 is box F1. Summaries by CV task and by ag task, macro (mean over datasets) and micro (pooled). |
| Judge | `RedHatAI/gemma-4-31B-it-FP8-dynamic`, thinking off, temperature 0. |
