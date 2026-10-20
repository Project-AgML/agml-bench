"""Per-dataset sampling run, status tracking, and committed manifests."""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from agbench.config import MANIFEST_DIR, sampling_hash

TASK_DIRS = {
    "image_classification": "classification",
    "object_detection": "detection",
    "semantic_segmentation": "segmentation",
    "image_text_to_text": "image_text_to_text",
}


def base_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "base-data"


def dataset_dir(data_dir: Path, entry: dict) -> Path:
    return base_dir(data_dir) / TASK_DIRS[entry["task"]] / entry["name"]


def scratch_dir(data_dir: Path, name: str) -> Path:
    return Path(data_dir) / ".scratch" / name


def failure_path(data_dir: Path, name: str) -> Path:
    return base_dir(data_dir) / "_failures" / f"{name}.json"


def manifest_path(entry: dict) -> Path:
    return MANIFEST_DIR / TASK_DIRS[entry["task"]] / f"{entry['name']}.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_meta(data_dir: Path, entry: dict) -> dict | None:
    p = dataset_dir(data_dir, entry) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def status_of(cfg: dict, data_dir: Path, entry: dict) -> str:
    if not entry["eligible"]:
        return "excluded"
    meta = read_meta(data_dir, entry)
    if meta:
        if entry.get("hf_sha") and meta.get("hf_sha") != entry["hf_sha"]:
            return "stale_upstream"
        if meta.get("sampling_hash") != sampling_hash(cfg, entry["name"], entry["task"]):
            return "stale_config"
        return "done"
    if failure_path(data_dir, entry["name"]).exists():
        return "failed"
    return "new"


# ---------------------------------------------------------------- worker (runs in a subprocess)

def run_dataset(cfg: dict, entry: dict, data_dir: Path, from_manifest: bool) -> dict:
    from agbench import build
    from agbench.hub import Repo
    from agbench.progress import log
    from agbench.sampling import rng_for, select_greedy_cover, select_itt, select_per_class

    name, task = entry["name"], entry["task"]
    s_cfg = cfg["sampling"]
    ov = cfg.get("overrides", {}).get(name, {})
    rng = rng_for(s_cfg["seed"], name)
    t0 = time.time()

    manifest = json.loads(manifest_path(entry).read_text()) if from_manifest else None
    repo = Repo(entry["hf_repo"], revision=manifest["hf_sha"] if manifest else entry.get("hf_sha"))

    if manifest:
        log(f"using manifest ({len(manifest['samples'])} samples, revision {manifest['hf_sha'][:7]})")
        info = manifest["info"]
        recs = [{"ref": s["ref"], "keys": s.get("keys", []), "ann": s.get("ann", {}), "strata": s.get("strata", {})}
                for s in manifest["samples"]]
        strata_keys = manifest.get("strata_keys")
    else:
        layout = repo.layout()
        log(f"layout: {layout}, revision {repo.revision[:7]}")
        is_itt = task == "image_text_to_text"
        if layout == "parquet":
            files = repo.data_files(".parquet", all_configs=is_itt, skip_keywords=s_cfg["skip_config_keywords"],
                                    prefer=ov.get("config"))
            if not files:
                raise build.Unsupported("no parquet files for chosen config")
            configs = sorted({f.config for f in files})
            log(f"{len(files)} parquet file(s), config(s): {', '.join(configs)}")
            if is_itt:
                keep = list(dict.fromkeys(s_cfg["itt"]["primary_keys"] + s_cfg["itt"].get("secondary_keys", []) + ov.get("strata", [])))
                recs, info = build.index_itt(repo, files, keep)
            else:
                recs, info = build.index_parquet(repo, files, task, entry["classes"], s_cfg["seg_max_index_rows"], rng)
        elif layout == "metadata_jsonl" and task != "image_text_to_text":
            recs, info = build.index_metadata_jsonl(repo, task, entry["classes"], s_cfg["skip_config_keywords"],
                                                    s_cfg["seg_max_index_rows"], rng)
        else:
            raise build.Unsupported(f"layout:{layout}")
        if not recs:
            raise build.Unsupported("no labelled rows")
        info["indexed_rows"] = len(recs)
        log(f"indexed {len(recs):,} rows")

        n, sp = s_cfg["samples_per_class"], s_cfg["split_priority"]
        strata_keys = None
        if task == "image_classification":
            idx = select_per_class(recs, n, rng, sp)
        elif task in ("object_detection", "semantic_segmentation"):
            idx = select_greedy_cover(recs, n, rng, sp)
        else:
            idx, strata_keys = select_itt(recs, s_cfg["itt"], rng, sp, ov.get("strata"))
            info["strata_counts_full"] = _strata_counts(recs, strata_keys)
        recs = [recs[i] for i in idx]
        log(f"selected {len(recs)} samples" + (f" (strata: {', '.join(strata_keys)})" if strata_keys else ""))

    out = dataset_dir(data_dir, entry)
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    samples = build.extract(repo, name, task, info, recs, tmp, ov.get("bbox_format"))
    if strata_keys:
        for s in samples:
            s["stratum"] = {k: s["strata"].get(k) for k in strata_keys}

    with open(tmp / "samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    meta = {
        "name": name, "task": task, "hf_repo": entry["hf_repo"], "hf_sha": repo.revision,
        "sampling_hash": sampling_hash(cfg, name, task), "created": now(), "seconds": round(time.time() - t0, 1),
        "n_samples": len(samples), "info": info, "strata_keys": strata_keys, "catalog": entry,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str))
    # Manifest first: a dataset only counts as done once meta.json appears via the rename.
    write_manifest(entry, meta, samples)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)

    failure_path(data_dir, name).unlink(missing_ok=True)
    return meta


def _strata_counts(recs: list[dict], keys: list[str]) -> dict:
    counts: dict[str, int] = {}
    for r in recs:
        k = " | ".join(str(r["strata"].get(x)) for x in keys)
        counts[k] = counts.get(k, 0) + 1
    return counts


def write_manifest(entry: dict, meta: dict, samples: list[dict]) -> None:
    p = manifest_path(entry)
    p.parent.mkdir(parents=True, exist_ok=True)
    slim = []
    for s in samples:
        ann = {k: s[k] for k in ("label", "objects", "classes_present") if k in s}
        if "objects" in ann:
            ann["objects"] = [{"label": o["label"], "bbox_raw": o["bbox_raw"]} for o in ann["objects"]]
        slim.append({"sample_id": s["sample_id"], "ref": s["ref"], "keys": s.get("keys", []), "ann": ann,
                     "strata": s.get("strata", {})})
    p.write_text(json.dumps({
        "name": meta["name"], "task": meta["task"], "hf_repo": meta["hf_repo"], "hf_sha": meta["hf_sha"],
        "sampling_hash": meta["sampling_hash"], "created": meta["created"], "info": meta["info"],
        "strata_keys": meta["strata_keys"], "samples": slim,
    }, indent=1, ensure_ascii=False, default=str))


def worker_main(argv: list[str]) -> int:
    """Entry point for `python -m agbench.pipeline NAME DATA_DIR CONFIG [--from-manifest]`."""
    from agbench.catalog import load_catalog
    from agbench.config import load_config

    name, data_dir, config_path = argv[:3]
    from_manifest = "--from-manifest" in argv
    cfg = load_config(config_path)
    entry = load_catalog()["datasets"][name]
    data_dir = Path(data_dir)
    try:
        meta = run_dataset(cfg, entry, data_dir, from_manifest)
        print(f"  ok: {meta['n_samples']} samples in {meta['seconds']}s")
        return 0
    except Exception as e:
        record_failure(data_dir, name, f"{type(e).__name__}: {e}", traceback.format_exc())
        print(f"  failed: {type(e).__name__}: {e}")
        return 1


def record_failure(data_dir: Path, name: str, reason: str, tb: str = "") -> None:
    p = failure_path(data_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"name": name, "reason": reason, "time": now(), "traceback": tb}, indent=1))


# ---------------------------------------------------------------- orchestrator

def run_many(cfg: dict, config_path: str, entries: list[dict], data_dir: Path, timeout: int, from_manifest: bool) -> dict:
    from agbench.progress import fmt_duration

    results = {"ok": [], "failed": []}
    t_start = time.time()
    for i, e in enumerate(entries, 1):
        name = e["name"]
        elapsed = time.time() - t_start
        timing = f"elapsed {fmt_duration(elapsed)}"
        if i > 1:
            timing += f", ETA {fmt_duration(elapsed / (i - 1) * (len(entries) - i + 1))}"
        print(f"[{i}/{len(entries)}] {name} ({e['task']})  [{timing}]", flush=True)
        scratch = scratch_dir(data_dir, name)
        scratch.mkdir(parents=True, exist_ok=True)
        # Any HF cache or Xet chunk written during this dataset lands in scratch and is deleted after.
        env = dict(os.environ, HF_HUB_CACHE=str(scratch / "hub"), HF_XET_CACHE=str(scratch / "xet"),
                   HF_DATASETS_CACHE=str(scratch / "datasets"), HF_XET_CHUNK_CACHE_SIZE_BYTES="0")
        cmd = [sys.executable, "-m", "agbench.pipeline", name, str(data_dir), config_path]
        if from_manifest:
            cmd.append("--from-manifest")
        try:
            rc = subprocess.run(cmd, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            record_failure(data_dir, name, f"timeout after {timeout}s")
            print(f"  failed: timeout after {timeout}s")
            rc = 1
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            shutil.rmtree(dataset_dir(data_dir, e).with_name(name + ".partial"), ignore_errors=True)
        results["ok" if rc == 0 else "failed"].append(name)
    shutil.rmtree(Path(data_dir) / ".scratch", ignore_errors=True)
    results["seconds"] = time.time() - t_start
    return results


if __name__ == "__main__":
    sys.exit(worker_main(sys.argv[1:]))
