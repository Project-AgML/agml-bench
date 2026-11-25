"""Fetch the AgML catalog, normalize it, and diff it against the last snapshot."""

import json
from collections import Counter
from datetime import date

import requests
from huggingface_hub import HfApi

from agml_bench.config import CATALOG_DIR

ITT_TAGS = {"task_categories:image-text-to-text", "task_categories:visual-question-answering"}
TASK_ALIASES = {"image-text-to-text": "image_text_to_text"}
DIFF_FIELDS = ["task", "hf_repo", "sensor_modality", "annotation_format", "classes", "num_images", "hf_sha"]


def _list(v) -> list:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return [v[k] for k in sorted(v, key=lambda k: int(k) if str(k).isdigit() else k)]
    return [v]


def _lower_list(v) -> list[str]:
    return [str(x).lower() for x in _list(v)]


def fetch_website_data(cfg: dict) -> tuple[str, dict]:
    c = cfg["catalog"]
    sha = c["branch"]
    try:
        r = requests.get(
            f"https://api.github.com/repos/{c['github_repo']}/commits",
            params={"path": c["data_path"], "sha": c["branch"], "per_page": 1},
            timeout=30,
        )
        r.raise_for_status()
        sha = r.json()[0]["sha"]
    except Exception:
        pass
    files = {}
    for name in ["hf_datasets.json", "datasets.json", "dataset_history.json"]:
        url = f"https://raw.githubusercontent.com/{c['github_repo']}/{sha}/{c['data_path']}/{name}"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        files[name] = r.json()
    return sha, files


def fetch_hf_org(org: str) -> dict:
    out = {}
    for d in HfApi().list_datasets(author=org, full=True):
        out[d.id] = {
            "sha": d.sha,
            "last_modified": d.last_modified.isoformat() if d.last_modified else None,
            "tags": list(d.tags or []),
        }
    return out


def normalize_entry(e: dict) -> dict:
    hf_link = e.get("hf_link") or ""
    hf_repo = hf_link.split("/datasets/", 1)[1].strip("/") if "/datasets/" in hf_link else None
    task = TASK_ALIASES.get(e.get("machine_learning_task"), e.get("machine_learning_task"))
    itt = None
    if task == "image_text_to_text":
        comp = e.get("composition") or {}
        itt = {
            "qa_type": _list(e.get("qa_type")),
            "task_dimensions": _list(e.get("task_dimensions")),
            "source_datasets": _list(comp.get("source_datasets")),
            "num_rows": e.get("num_rows"),
            "conversation_format": e.get("conversation_format"),
        }
    return {
        "name": e["name"],
        "hf_repo": hf_repo,
        "task": task,
        "agricultural_task": e.get("agricultural_task") or None,
        "sensor_modality": _lower_list(e.get("sensor_modality")),
        "annotation_format": e.get("annotation_format"),
        "classes": [str(c) for c in _list(e.get("classes"))],
        "num_images": e.get("num_images"),
        "crops": _list(e.get("crop_types")),
        "location": _list(e.get("location")) or _list(e.get("country")),
        "platform": _list(e.get("platform")),
        "environment": e.get("environment"),
        "real_or_synthetic": e.get("real_or_synthetic"),
        "license": e.get("license"),
        "citation": e.get("bibtex") or e.get("citation"),
        "documentation": e.get("documentation"),
        "example_image": e.get("examples_image_url"),
        "itt": itt,
        "catalog_source": e.get("source"),
    }


def build_catalog(cfg: dict, website: dict, hf: dict, site_sha: str, old: dict | None = None) -> dict:
    entries: dict[str, dict] = {}
    for e in website["hf_datasets.json"]:
        entries[e["name"]] = normalize_entry(e)
    # datasets.json is mostly iNatAg children; keep only top-level entries.
    for e in website["datasets.json"]:
        if e.get("parent_dataset") in (None, "", "None") and e["name"] not in entries:
            entries[e["name"]] = normalize_entry(e)

    linked = {e["hf_repo"] for e in entries.values() if e["hf_repo"]}
    for repo_id, info in hf.items():
        if repo_id in linked or not ITT_TAGS & set(info["tags"]):
            continue
        name = repo_id.split("/", 1)[1]
        entries[name] = normalize_entry({"name": name, "machine_learning_task": "image_text_to_text",
                                         "hf_link": f"https://huggingface.co/datasets/{repo_id}",
                                         "source": "huggingface_org"})

    _resolve_links(entries, hf, cfg["catalog"]["hf_org"])
    for e in entries.values():
        info = hf.get(e["hf_repo"]) if e["hf_repo"] else None
        e["hf_sha"] = info["sha"] if info else None
        e["hf_last_modified"] = info["last_modified"] if info else None
        e["eligible"], e["exclusion_reason"] = eligibility(cfg, e, info is not None)
        prev = (old or {}).get("datasets", {}).get(e["name"], {})
        e["first_seen"] = prev.get("first_seen") or date.today().isoformat()

    return {
        "generated": date.today().isoformat(),
        "website_sha": site_sha,
        "datasets": dict(sorted(entries.items(), key=lambda kv: kv[0].lower())),
    }


def _resolve_links(entries: dict, hf: dict, org: str) -> None:
    """Repair website hf_links that are missing on HF or shared with another dataset."""
    counts = Counter(e["hf_repo"] for e in entries.values() if e["hf_repo"])
    by_lower = {r.lower(): r for r in hf}
    for name, e in entries.items():
        repo = e["hf_repo"]
        if not repo:
            continue
        same_name = by_lower.get(f"{org}/{name}".lower(), f"{org}/{name}")
        conflict = counts[repo] > 1 and repo.split("/", 1)[1] != name
        if repo not in hf or conflict:
            e["hf_repo_note"] = f"website link {repo}"
            if repo.lower() in by_lower and not conflict:
                e["hf_repo"] = by_lower[repo.lower()]
            elif same_name in hf:
                e["hf_repo"] = same_name
            elif conflict:
                e["hf_repo"] = None
                e["hf_repo_note"] += " is shared with another dataset"


def eligibility(cfg: dict, e: dict, on_hf: bool) -> tuple[bool, str | None]:
    rules = cfg["eligibility"]
    ov = cfg.get("overrides", {}).get(e["name"], {})
    if "include" in ov:
        return bool(ov["include"]), (None if ov["include"] else f"override:{ov.get('reason', 'excluded')}")
    if e["task"] not in rules["tasks"]:
        return False, f"task:{e['task']}"
    if not e["hf_repo"]:
        return False, "no_hf_link"
    if not on_hf:
        return False, "hf_repo_not_found"
    is_itt = e["task"] == "image_text_to_text"
    if not (is_itt and rules.get("itt_ignore_sensor")):
        mods = e["sensor_modality"]
        if not mods or any(m not in rules["sensor_modalities"] for m in mods):
            return False, f"sensor:{','.join(mods) or 'unknown'}"
    return True, None


def diff_catalogs(old: dict | None, new: dict) -> dict:
    old_ds = (old or {}).get("datasets", {})
    new_ds = new["datasets"]
    added = sorted(set(new_ds) - set(old_ds))
    removed = sorted(set(old_ds) - set(new_ds))
    changed = {}
    for name in sorted(set(new_ds) & set(old_ds)):
        fields = [f for f in DIFF_FIELDS + ["eligible"] if old_ds[name].get(f) != new_ds[name].get(f)]
        if fields:
            changed[name] = fields
    return {"added": added, "removed": removed, "changed": changed}


def load_catalog() -> dict | None:
    p = CATALOG_DIR / "catalog.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_catalog(cat: dict, history: dict, diff: dict, first: bool = False) -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "catalog.json").write_text(json.dumps(cat, indent=1, ensure_ascii=False))
    (CATALOG_DIR / "dataset_history.json").write_text(json.dumps(history, indent=1))
    if first:
        lines = [f"## {cat['generated']} (website {cat['website_sha'][:7]})", "",
                 f"Initial catalog: {len(cat['datasets'])} datasets."]
        (CATALOG_DIR / "CHANGES.md").write_text("# Catalog changes\n\n" + "\n".join(lines) + "\n")
    elif diff["added"] or diff["removed"] or diff["changed"]:
        lines = [f"## {cat['generated']} (website {cat['website_sha'][:7]})", ""]
        for key, label in [("added", "Added"), ("removed", "Removed")]:
            if diff[key]:
                lines.append(f"**{label} ({len(diff[key])})**: " + ", ".join(diff[key]))
        if diff["changed"]:
            lines.append(f"**Changed ({len(diff['changed'])})**: "
                         + ", ".join(f"{n} [{'/'.join(f)}]" for n, f in diff["changed"].items()))
        p = CATALOG_DIR / "CHANGES.md"
        prev = p.read_text() if p.exists() else "# Catalog changes\n\n"
        head, _, rest = prev.partition("\n\n")
        p.write_text(head + "\n\n" + "\n".join(lines) + "\n\n" + rest)
