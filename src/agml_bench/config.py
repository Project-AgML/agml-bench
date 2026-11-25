import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base_data.yaml"
CATALOG_DIR = REPO_ROOT / "catalog"
MANIFEST_DIR = REPO_ROOT / "manifests" / "base-data"
TRAIN_MANIFEST_DIR = REPO_ROOT / "manifests" / "train-data"


def load_config(path: Path | str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def sampling_hash(cfg: dict, name: str, task: str) -> str:
    """Hash of the settings that affect which samples this dataset gets."""
    s = cfg["sampling"]
    relevant = {k: s[k] for k in ("seed", "split_priority")}
    if task == "image_text_to_text":
        relevant["itt"] = s["itt"]
    else:
        relevant.update(samples_per_class=s["samples_per_class"], skip_config_keywords=s["skip_config_keywords"])
        if task == "semantic_segmentation":
            relevant["seg_max_index_rows"] = s["seg_max_index_rows"]
    payload = {"sampling": relevant, "override": cfg.get("overrides", {}).get(name, {})}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def train_sampling_hash(cfg: dict, name: str, task: str) -> str:
    """Hash of the settings that affect a dataset's training samples (including which samples are evaluated)."""
    s = cfg["sampling"]
    payload = {"train": cfg["train_sampling"], "eval": sampling_hash(cfg, name, task),
               "skip_config_keywords": s["skip_config_keywords"], "seg_max_index_rows": s["seg_max_index_rows"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
