"""Ag-task bucket per class (classification/detection/segmentation) or per sample (ITT)."""

import re
from functools import cache

from agml_bench.eval.resources import ag_tasks, norm


def split_classes(classes) -> list[str]:
    """Catalog classes; some entries are one comma-joined string."""
    out = []
    for c in classes or []:
        out += [x.strip() for x in str(c).split(", ")] if ", " in str(c) else [str(c)]
    return out


def dataset_bucket(name: str, entry: dict) -> str:
    cfg = ag_tasks()
    ds = cfg["datasets"].get(name) or {}
    if ds.get("task_type"):
        return ds["task_type"]
    at = entry.get("agricultural_task")
    if at in cfg["agricultural_task"]:
        return cfg["agricultural_task"][at]
    for pattern, bucket in cfg["name_keywords"]:
        if re.search(pattern, name, re.I):
            return bucket
    return "other"


@cache
def _overrides(name: str) -> dict[str, str]:
    return {norm(k): v for k, v in ((ag_tasks()["datasets"].get(name) or {}).get("class_overrides") or {}).items()}


def class_bucket(name: str, entry: dict, label: str) -> str:
    return _overrides(name).get(norm(label)) or dataset_bucket(name, entry)


def itt_bucket(sample: dict) -> str:
    rule = ag_tasks()["itt"].get(sample["dataset"])
    if not rule:
        return "other"
    value = str((sample.get("strata") or {}).get(rule.get("field", ""), "") or (sample.get("raw_metadata") or {}).get(rule.get("field", ""), ""))
    if value in (rule.get("map") or {}):
        return rule["map"][value]
    for needle, bucket in (rule.get("contains") or {}).items():
        if needle.lower() in value.lower():
            return bucket
    return rule.get("default", "other")


def objects_bucket(name: str, entry: dict, labels: list[str]) -> str:
    """Detection/segmentation image: most common bucket among its GT labels."""
    if not labels:
        return dataset_bucket(name, entry)
    buckets = [class_bucket(name, entry, l) for l in labels]
    return max(sorted(set(buckets)), key=buckets.count)  # a tie goes to the first bucket alphabetically, not to hash order


def check(entries: list[dict]) -> list[str]:
    """Problems to triage: unmapped datasets, overrides naming unknown classes, keyword hits with no override."""
    cfg = ag_tasks()
    kw = re.compile(cfg["pest_keywords"], re.I)
    out = []
    for e in entries:
        name, classes = e["name"], split_classes(e.get("classes"))
        ds = cfg["datasets"].get(name) or {}
        if e["task"] == "image_text_to_text":
            if name not in cfg["itt"]:
                out.append(f"{name}: no itt rule")
            continue
        if not ds and e.get("agricultural_task") not in cfg["agricultural_task"]:
            out.append(f"{name}: no task_type and agricultural_task={e.get('agricultural_task')!r} -> {dataset_bucket(name, e)}")
        known = {norm(c) for c in classes}
        for k in ds.get("class_overrides") or {}:
            if norm(k) not in known:
                out.append(f"{name}: override class {k!r} not in catalog classes")
        if dataset_bucket(name, e) != "pest_damage":
            for c in classes:
                if kw.search(c) and class_bucket(name, e, c) != "pest_damage":
                    out.append(f"{name}: class {c!r} looks like pest/abiotic but maps to {class_bucket(name, e, c)}")
    return out
