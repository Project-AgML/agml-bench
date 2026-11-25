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

from agml_bench.config import MANIFEST_DIR, TRAIN_MANIFEST_DIR, sampling_hash, train_sampling_hash

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


def failure_path(data_dir: Path, name: str, train: bool = False) -> Path:
    return (train_root(data_dir) if train else base_dir(data_dir)) / "_failures" / f"{name}.json"


def train_root(data_dir: Path) -> Path:
    return Path(data_dir) / "train-data"


def train_dir(data_dir: Path, entry: dict) -> Path:
    return train_root(data_dir) / TASK_DIRS[entry["task"]] / entry["name"]


def train_manifest_path(entry: dict) -> Path:
    return TRAIN_MANIFEST_DIR / TASK_DIRS[entry["task"]] / f"{entry['name']}.json"


def train_status_of(cfg: dict, data_dir: Path, entry: dict) -> str:
    if not entry["eligible"] or entry["task"] == "image_text_to_text":
        return "excluded"
    if not manifest_path(entry).exists():
        return "no_eval_samples"
    p = train_dir(data_dir, entry) / "meta.json"
    if p.exists():
        meta = json.loads(p.read_text())
        return "done" if meta.get("sampling_hash") == train_sampling_hash(cfg, entry["name"], entry["task"]) else "stale_config"
    return "failed" if failure_path(data_dir, entry["name"], train=True).exists() else "new"


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

def run_dataset(cfg: dict, entry: dict, data_dir: Path, from_manifest: bool, manifest: dict | None = None,
                bundle: dict | None = None) -> dict:
    """`manifest` (with `bundle`): download exactly these samples for a frozen benchmark; manifests/ is not touched."""
    from agml_bench import build
    from agml_bench.hub import Repo
    from agml_bench.progress import log
    from agml_bench.sampling import rng_for, select_greedy_cover, select_itt, select_per_class

    name, task = entry["name"], entry["task"]
    s_cfg = cfg["sampling"]
    ov = cfg.get("overrides", {}).get(name, {})
    rng = rng_for(s_cfg["seed"], name)
    t0 = time.time()

    if manifest is None and from_manifest:
        manifest = json.loads(manifest_path(entry).read_text())
    repo = Repo(entry["hf_repo"], revision=manifest["hf_sha"] if manifest else entry.get("hf_sha"))

    if manifest:
        log(f"using manifest ({len(manifest['samples'])} samples, revision {manifest['hf_sha'][:7]})")
        info = manifest["info"]
        recs = [{"ref": s["ref"], "keys": s.get("keys", []), "ann": s.get("ann", {}), "strata": s.get("strata", {}),
                 "sample_id": s["sample_id"]} for s in manifest["samples"]]
        strata_keys = manifest.get("strata_keys")
    else:
        recs, info = index_dataset(cfg, entry, repo, rng)

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
    if bundle:
        meta["benchmark"] = bundle
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str))
    # Manifest first: a dataset only counts as done once meta.json appears via the rename.
    if not bundle:
        write_manifest(entry, meta, samples)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)

    failure_path(data_dir, name).unlink(missing_ok=True)
    return meta


def index_dataset(cfg: dict, entry: dict, repo, rng) -> tuple[list[dict], dict]:
    """Every labelled row of a dataset as a selection record ({ref, keys, ann, strata}), plus layout info."""
    from agml_bench import build
    from agml_bench.progress import log

    name, task = entry["name"], entry["task"]
    s_cfg = cfg["sampling"]
    ov = cfg.get("overrides", {}).get(name, {})
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
    return recs, info


def _ref_key(ref: dict) -> str:
    return json.dumps(ref, sort_keys=True, default=str)


def run_train_dataset(cfg: dict, entry: dict, data_dir: Path, from_manifest: bool) -> dict:
    """Few-shot training samples: `subsets` seeded draws of each shot count, never an evaluated sample."""
    from agml_bench import build
    from agml_bench.hub import Repo
    from agml_bench.progress import log
    from agml_bench.sampling import rng_for, select_greedy_cover, select_per_class

    name, task = entry["name"], entry["task"]
    t_cfg = cfg["train_sampling"]
    ov = cfg.get("overrides", {}).get(name, {})
    t0 = time.time()
    if task == "image_text_to_text":
        raise build.Unsupported("no training samples for image-text-to-text")
    evaluated = json.loads(manifest_path(entry).read_text())
    manifest = json.loads(train_manifest_path(entry).read_text()) if from_manifest else None
    # Same revision as the evaluated samples, so their rows can be recognised and excluded.
    repo = Repo(entry["hf_repo"], revision=evaluated["hf_sha"])

    if manifest:
        log(f"using manifest ({len(manifest['samples'])} samples, revision {manifest['hf_sha'][:7]})")
        info = manifest["info"]
        recs = [{"ref": s["ref"], "keys": s.get("keys", []), "ann": s.get("ann", {}), "strata": {}} for s in manifest["samples"]]
        subsets = manifest["subsets"]
    else:
        recs, info = index_dataset(cfg, entry, repo, rng_for(t_cfg["seed"], name))
        excluded = {_ref_key(s["ref"]) for s in evaluated["samples"]}
        recs = [r for r in recs if _ref_key(r["ref"]) not in excluded]
        log(f"{len(recs):,} rows left after excluding {len(excluded)} evaluated samples")
        if not recs:
            raise build.Unsupported("no rows outside the evaluated samples")
        select = select_per_class if task == "image_classification" else select_greedy_cover
        picks = {}
        for s in range(t_cfg["subsets"]):
            for k in t_cfg["shots"]:
                # a fresh generator per shot count: the same image order, so smaller sets nest in larger ones
                picks[(s, k)] = select(recs, k, rng_for(t_cfg["seed"] + s, name), t_cfg["split_priority"])
        used = sorted(set().union(*picks.values()))
        ids = {i: build.sample_id(name, recs[i]["ref"]) for i in used}
        subsets = {f"s{s}": {str(k): [ids[i] for i in picks[(s, k)]] for k in t_cfg["shots"]} for s in range(t_cfg["subsets"])}
        recs = [recs[i] for i in used]
        log(f"selected {len(recs)} training samples over {t_cfg['subsets']} subsets x shots {t_cfg['shots']}")

    out = train_dir(data_dir, entry)
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    samples = build.extract(repo, name, task, info, recs, tmp, ov.get("bbox_format"))
    have = {s["sample_id"] for s in samples}
    subsets = {s: {k: [i for i in v if i in have] for k, v in ks.items()} for s, ks in subsets.items()}
    with open(tmp / "samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    (tmp / "subsets.json").write_text(json.dumps(subsets, indent=1))
    meta = {
        "name": name, "task": task, "hf_repo": entry["hf_repo"], "hf_sha": repo.revision,
        "sampling_hash": train_sampling_hash(cfg, name, task), "created": now(), "seconds": round(time.time() - t0, 1),
        "n_samples": len(samples), "info": info, "strata_keys": None, "catalog": entry, "train_sampling": t_cfg,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str))
    write_manifest(entry, meta, samples, path=train_manifest_path(entry), extra={"subsets": subsets})
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    failure_path(data_dir, name, train=True).unlink(missing_ok=True)
    return meta


def _strata_counts(recs: list[dict], keys: list[str]) -> dict:
    counts: dict[str, int] = {}
    for r in recs:
        k = " | ".join(str(r["strata"].get(x)) for x in keys)
        counts[k] = counts.get(k, 0) + 1
    return counts


def write_manifest(entry: dict, meta: dict, samples: list[dict], path: Path | None = None, extra: dict | None = None) -> None:
    p = path or manifest_path(entry)
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
        "strata_keys": meta["strata_keys"], "samples": slim, **(extra or {}),
    }, indent=1, ensure_ascii=False, default=str))


def worker_main(argv: list[str]) -> int:
    """Entry point for `python -m agml_bench.pipeline NAME DATA_DIR CONFIG [--from-manifest]`."""
    from agml_bench.catalog import load_catalog
    from agml_bench.config import load_config

    name, data_dir, config_path = argv[:3]
    from_manifest = "--from-manifest" in argv
    train = "--train" in argv
    cfg = load_config(config_path)
    data_dir = Path(data_dir)
    repo, revision = None, None
    try:
        if "--bundle" in argv:
            from agml_bench.bench import load

            b = load(argv[argv.index("--bundle") + 1])
            repo, revision = b.entry(name)["hf_repo"], b.manifests[name]["hf_sha"]
            meta = run_dataset(cfg, b.entry(name), data_dir, True, manifest=b.manifests[name],
                               bundle={"name": b.name, "hash": b.hash})
        else:
            entry = load_catalog()["datasets"][name]
            repo = entry["hf_repo"]
            meta = (run_train_dataset if train else run_dataset)(cfg, entry, data_dir, from_manifest)
        print(f"  ok: {meta['n_samples']} samples in {meta['seconds']}s")
        return 0
    except Exception as e:
        reason = explain(e, repo, revision)
        record_failure(data_dir, name, reason, traceback.format_exc(), train=train)
        print(f"  failed: {reason}")
        return 1


def explain(e: Exception, repo: str | None, revision: str | None) -> str:
    """A failure reason a user can act on, for the errors that come from the source repo rather than the code."""
    text, kind = str(e), type(e).__name__
    rev = f" {revision[:7]}" if revision else ""
    if kind == "RepositoryNotFoundError" or "(repository not found)" in text or "Repository Not Found" in text:
        return f"source repo {repo} not found (deleted, renamed or private)"
    if kind == "RevisionNotFoundError" or "(revision not found)" in text or "Invalid rev id" in text:
        return (f"pinned revision{rev} no longer exists in {repo} (repo re-uploaded or history rewritten; "
                f"maintainers: agml-bench bench repin)")
    if kind == "GatedRepoError" or "gated" in text.lower():
        return f"{repo} is gated: accept its terms on huggingface.co, then rerun"
    if "403" in text or "Forbidden" in text:
        return f"no permission for this on {repo}: your Hugging Face token or org role is read-only"
    if "401" in text or "Unauthorized" in text:
        return "not authorized by Hugging Face: run hf auth login, then rerun"
    return f"{kind}: {text}"[:300]


def record_failure(data_dir: Path, name: str, reason: str, tb: str = "", train: bool = False) -> None:
    p = failure_path(data_dir, name, train)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"name": name, "reason": reason, "time": now(), "traceback": tb}, indent=1))


# ---------------------------------------------------------------- orchestrator

def run_many(cfg: dict, config_path: str, entries: list[dict], data_dir: Path, timeout: int, from_manifest: bool,
             train: bool = False, bundle: str | None = None) -> dict:
    """One subprocess per dataset. In a terminal: a progress bar and one line per dataset, with each dataset's
    detailed output appended to <data_dir>/_logs/<command>.log. Otherwise (a log file, a batch job): every line."""
    from contextlib import nullcontext

    from agml_bench.progress import fmt_duration

    results = {"ok": [], "failed": []}
    t_start = time.time()
    tty = sys.stdout.isatty() and os.environ.get("AGML_BENCH_PLAIN") != "1"
    log_path = Path(data_dir) / "_logs" / f"{'pull' if bundle else 'sample-train' if train else 'sample'}.log"
    bar = None
    if tty:
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

        log_path.parent.mkdir(parents=True, exist_ok=True)
        bar = Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                       TextColumn("left"), TimeRemainingColumn())
        job = bar.add_task("", total=len(entries))
    with bar or nullcontext():
        for i, e in enumerate(entries, 1):
            name = e["name"]
            if bar:
                bar.update(job, description=f"{name[:40]:<40}")
            else:
                elapsed = time.time() - t_start
                timing = f"elapsed {fmt_duration(elapsed)}"
                if i > 1:
                    timing += f", ETA {fmt_duration(elapsed / (i - 1) * (len(entries) - i + 1))}"
                print(f"[{i}/{len(entries)}] {name} ({e['task']})  [{timing}]", flush=True)
            t0 = time.time()
            rc = _run_one(e, data_dir, config_path, timeout, from_manifest, train, bundle, log_path if bar else None)
            results["ok" if rc == 0 else "failed"].append(name)
            if bar:
                if rc == 0:
                    bar.console.print(f"[green]done[/]   {name}  ({fmt_duration(time.time() - t0)})")
                else:
                    why = json.loads(failure_path(data_dir, name, train).read_text())["reason"]
                    bar.console.print(f"[red]failed[/] {name}: {why}")
                bar.advance(job)
    try:  # only if empty: another run may be downloading into its own subfolder (each dataset's is removed above)
        (Path(data_dir) / ".scratch").rmdir()
    except OSError:
        pass
    results["seconds"] = time.time() - t_start
    results["log"] = str(log_path) if bar else None
    return results


def _run_one(e: dict, data_dir: Path, config_path: str, timeout: int, from_manifest: bool, train: bool,
             bundle: str | None, log_path: Path | None) -> int:
    name = e["name"]
    scratch = scratch_dir(data_dir, name)
    scratch.mkdir(parents=True, exist_ok=True)
    # Any HF cache or Xet chunk written during this dataset lands in scratch and is deleted after.
    env = dict(os.environ, HF_HUB_CACHE=str(scratch / "hub"), HF_XET_CACHE=str(scratch / "xet"),
               HF_DATASETS_CACHE=str(scratch / "datasets"), HF_XET_CHUNK_CACHE_SIZE_BYTES="0")
    cmd = [sys.executable, "-m", "agml_bench.pipeline", name, str(data_dir), config_path]
    if from_manifest:
        cmd.append("--from-manifest")
    if train:
        cmd.append("--train")
    if bundle:
        cmd += ["--bundle", bundle]
    log = open(log_path, "a") if log_path else None
    try:
        if log:
            log.write(f"\n=== {name} ({e['task']}) {now()}\n")
            log.flush()
        rc = subprocess.run(cmd, env=env, timeout=timeout, stdout=log, stderr=subprocess.STDOUT if log else None).returncode
    except subprocess.TimeoutExpired:
        record_failure(data_dir, name, f"timeout after {timeout}s (rerun to retry; a slow connection may need --timeout)", train=train)
        if not log:
            print(f"  failed: timeout after {timeout}s")
        rc = 1
    finally:
        if log:
            log.close()
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree((train_dir if train else dataset_dir)(data_dir, e).with_name(name + ".partial"), ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(worker_main(sys.argv[1:]))
