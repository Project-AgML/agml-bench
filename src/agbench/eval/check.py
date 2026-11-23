"""run_manifest.json (what a run was, and whether its scores compare with the reference results) and `agbench eval
check`: a run dir's files against the schema and its benchmark."""

import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from agbench.eval import metrics as mx

SCHEMA = 1
REQUIRED = ("item_id", "dataset", "task", "format", "answer", "text", "correct", "parse_ok", "needs_judge", "judge_correct",
            "finish_reason", "error")
BY_TASK = {"classification": ("gt_label",), "detection": ("tp", "fp", "fn", "mean_iou", "pred_boxes"),
           "segmentation": ("tp", "fp", "fn", "mean_iou", "pred_boxes"), "image_text_to_text": ("qa_format",)}


def _version() -> str | None:
    try:
        return version("agbench")
    except PackageNotFoundError:
        return None


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def _bundle(name: str | None):
    from agbench import bench

    try:
        return bench.load(name) if name else None
    except KeyError:
        return None


def comparability(run: dict, records: list[dict]) -> list[str]:
    """Reasons a run's scores are not comparable with the benchmark's reference results (empty: comparable)."""
    from agbench.bench import judge_prompt_sha

    bm = run.get("benchmark", {}).get("benchmark")
    if not bm:
        return ["no frozen benchmark (--benchmark none): items built from the local data"]
    why = []
    b = _bundle(bm["name"])
    if b is None:
        why.append(f"benchmark {bm['name']} is not installed")
    elif b.hash != bm["hash"]:
        why.append(f"benchmark hash {bm['hash']} differs from the installed {bm['name']} ({b.hash})")
    if bm.get("variant"):
        why.append(f"item {bm['variant']} (seed or multiple-choice options changed)")
    if any(r.get("needs_judge") for r in records):
        j = run.get("judge") or {}
        ref = (b.spec["judge"] if b else {}) or {}
        if any(r.get("needs_judge") and r.get("judge_correct") is None for r in records):
            why.append("open-ended answers not judged yet (agbench eval judge)")
        elif j:
            if ref and j.get("model") != ref.get("model"):
                why.append(f"judge {j.get('model')} is not the reference judge {ref.get('model')}")
            if ref and j.get("prompt_sha", judge_prompt_sha()) != ref.get("prompt_sha"):
                why.append("judge prompt differs from the reference judge prompt")
    return why


def manifest(run_dir, records: list[dict] | None = None) -> dict | None:
    """Summary of a run: agbench version, benchmark (name, hash, variant), method and sampling, item seed, judge,
    coverage and comparability."""
    run_dir = Path(run_dir)
    if not (run_dir / "run.json").exists():
        return None
    run = json.loads((run_dir / "run.json").read_text())
    records = records if records is not None else _read_jsonl(run_dir / "predictions.jsonl")
    args = run.get("benchmark", {})
    bm = args.get("benchmark")
    b = _bundle(bm["name"]) if bm else None
    skipped = json.loads((run_dir / "skipped.json").read_text()) if (run_dir / "skipped.json").exists() else []
    answered = {r["item_id"] for r in records if not r.get("error")}
    cov = {"items": run.get("n_items"), "answered": len(answered),
           "errors": sum(1 for r in records if r.get("error")),
           "pending_judge": sum(1 for r in records if r.get("needs_judge") and r.get("judge_correct") is None),
           "skipped_datasets": sorted({s["dataset"] for s in skipped if "dataset" in s})}
    if b:
        filters = {k: args.get(k) for k in ("tasks", "formats", "only", "limit_per_dataset") if args.get(k)}
        cov |= {"benchmark_items": len(b.raw_items), "filters": filters,
                "unavailable_datasets": sorted(set(b.unavailable) & set(cov["skipped_datasets"]))}
        cov["complete"] = not filters and len(answered) == len(b.raw_items) - sum(
            1 for r in b.raw_items if r["dataset"] in cov["unavailable_datasets"])
    why = comparability(run, records)
    return {"schema": SCHEMA, "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agbench": {"version": _version(), "git": run.get("agbench_git")}, "benchmark": bm,
            "method": run.get("method"), "item_seed": args.get("seed"), "mcqa": args.get("mcqa"),
            "judge": run.get("judge"), "coverage": cov, "comparable": not why, "not_comparable": why}


def write_manifest(run_dir, records: list[dict] | None = None) -> dict | None:
    m = manifest(run_dir, records)
    if m:
        (Path(run_dir) / "run_manifest.json").write_text(json.dumps(m, indent=2))
    return m


def _items_match_bundle(items: list[dict], b) -> list[str]:
    """Run items that differ from the benchmark's frozen items (any field; images compared by relative path)."""
    frozen = {r["item_id"]: r for r in b.raw_items}
    bad = []
    for it in items:
        f = frozen.get(it["item_id"])
        if f is None:
            bad.append(f"{it['item_id']}: not in {b.name}")
            continue
        imgs_ok = len(it["images"]) == len(f["images"]) and all(a.endswith(p) for a, p in zip(it["images"], f["images"]))
        diff = [k for k in f if k != "images" and it.get(k) != f[k]]
        if diff or not imgs_ok:
            bad.append(f"{it['item_id']}: {', '.join(diff + ([] if imgs_ok else ['images']))} differ from {b.name}")
    return bad


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b) or abs(a - b) <= 1e-9 * max(abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def check_run(run_dir) -> dict:
    """{'errors': [...], 'warnings': [...], 'stats': {...}}. Errors: the run's files are broken, inconsistent or do not
    match its benchmark. Warnings: the run is valid but incomplete or not comparable with the reference results."""
    run_dir = Path(run_dir)
    errors, warnings, stats = [], [], {}
    for f in ("run.json", "items.jsonl", "predictions.jsonl"):
        if not (run_dir / f).exists():
            errors.append(f"{f} missing")
    if errors:
        return {"errors": errors, "warnings": warnings, "stats": stats}
    run = json.loads((run_dir / "run.json").read_text())
    items = _read_jsonl(run_dir / "items.jsonl")
    recs = _read_jsonl(run_dir / "predictions.jsonl")
    ids = {it["item_id"] for it in items}

    dup = [k for k, n in Counter(r.get("item_id") for r in recs).items() if n > 1]
    if dup:
        errors.append(f"{len(dup)} items with more than one prediction (e.g. {dup[0]})")
    extra = [r["item_id"] for r in recs if r.get("item_id") not in ids]
    if extra:
        errors.append(f"{len(extra)} predictions for items not in items.jsonl (e.g. {extra[0]})")
    bad_schema = Counter()
    for r in recs:
        for k in REQUIRED + BY_TASK.get(r.get("task"), ()):
            if k not in r:
                bad_schema[k] += 1
        if r.get("task") in mx.BOX_TASKS:
            if not isinstance(r.get("correct"), (int, float)) or isinstance(r.get("correct"), bool):
                bad_schema["correct (number for box tasks)"] += 1
        elif not isinstance(r.get("correct"), bool):
            bad_schema["correct (bool)"] += 1
    for k, n in bad_schema.items():
        errors.append(f"{n} predictions missing or with a wrong type for '{k}'")

    bm = run.get("benchmark", {}).get("benchmark")
    b = _bundle(bm["name"]) if bm else None
    if bm and b and not bm.get("variant"):
        diff = _items_match_bundle(items, b)
        if diff:
            errors.append(f"{len(diff)} items differ from the frozen benchmark (e.g. {diff[0]})")

    if (run_dir / "metrics.json").exists() and not bad_schema:
        stored = json.loads((run_dir / "metrics.json").read_text())
        fresh = json.loads(json.dumps(mx.compute([r for r in recs if r.get("item_id") in ids])))
        # A dataset row's ag_task label (its most common bucket) broke ties by set order before agbench 0.1.0's fix;
        # runs from then can carry either label for a tied dataset. Scores are compared; that label is not.
        strip = lambda m: {**m, "datasets": [{k: v for k, v in r.items() if k != "ag_task"} for r in m.get("datasets", [])]}
        if not _same(strip(fresh), strip(stored)):
            errors.append("metrics.json does not match predictions.jsonl (rewrite it: agbench eval check --rescore)")
    elif not (run_dir / "metrics.json").exists():
        warnings.append("metrics.json missing (run not finished)")

    have = {r["item_id"] for r in recs if not r.get("error")}
    n_err = sum(1 for r in recs if r.get("error"))
    n = len(recs) or 1
    stats = {"items": len(items), "predictions": len(recs), "answered": len(have), "errors": n_err,
             "parse_fail_rate": round(sum(1 for r in recs if not r.get("parse_ok") and not r.get("error")) / n, 4),
             "truncated_rate": round(sum(1 for r in recs if r.get("truncated")) / n, 4),
             "idk_rate": round(sum(1 for r in recs if r.get("idk")) / n, 4),
             "empty_answer_rate": round(sum(1 for r in recs if not (r.get("text") or "").strip() and not r.get("error")) / n, 4)}
    if len(have) < len(items):
        warnings.append(f"{len(items) - len(have)} of {len(items)} items without an answer ({n_err} request errors); "
                        "resume with --run-dir")
    judges = Counter(r.get("judge_model") for r in recs if r.get("judge_correct") is not None)
    if len(judges) > 1:
        warnings.append(f"open-ended answers judged by more than one judge: {dict(judges)}")
    m = manifest(run_dir, recs)
    stats["comparable"] = m["comparable"]
    stats["complete"] = m["coverage"].get("complete")
    for w in m["not_comparable"]:
        warnings.append(f"not comparable: {w}")
    cov = m["coverage"]
    if cov.get("complete") is False and len(have) == len(items):
        warnings.append(f"covers {len(items)} of {cov['benchmark_items']} benchmark items"
                        + (f" ({', '.join(f'{k}={v}' for k, v in cov['filters'].items())})" if cov["filters"] else ""))
    if m["coverage"].get("unavailable_datasets"):
        warnings.append("skipped (source removed): " + ", ".join(m["coverage"]["unavailable_datasets"]))
    return {"errors": errors, "warnings": warnings, "stats": stats}
