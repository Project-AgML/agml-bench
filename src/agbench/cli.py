import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import typer

from agbench import catalog as cat
from agbench import pipeline
from agbench.config import DEFAULT_CONFIG, REPO_ROOT, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
catalog_app = typer.Typer(no_args_is_help=True)
app.add_typer(catalog_app, name="catalog", help="Sync and diff the AgML catalog.")

ConfigOpt = typer.Option(DEFAULT_CONFIG, "--config", help="Sampling config YAML.")
DataDirOpt = typer.Option(..., "--data-dir", help="Where sampled images are written.")


def _require_catalog() -> dict:
    c = cat.load_catalog()
    if c is None:
        typer.echo("No catalog yet. Run: agbench catalog sync")
        raise typer.Exit(1)
    return c


def _filter(entries: dict, task: list[str] | None, only: str | None) -> list[dict]:
    out = []
    for e in entries.values():
        if task and pipeline.TASK_DIRS.get(e["task"]) not in task and e["task"] not in task:
            continue
        if only and not re.search(only, e["name"]):
            continue
        out.append(e)
    return out


@catalog_app.command("sync")
def catalog_sync(config: Path = ConfigOpt):
    """Fetch the latest AgML catalog and show what changed."""
    cfg = load_config(config)
    sha, website = cat.fetch_website_data(cfg)
    hf = cat.fetch_hf_org(cfg["catalog"]["hf_org"])
    old = cat.load_catalog()
    new = cat.build_catalog(cfg, website, hf, sha, old)
    diff = cat.diff_catalogs(old, new)
    cat.save_catalog(new, website["dataset_history.json"], diff, first=old is None)
    _print_diff(diff, first=old is None)
    _print_summary(new)


@catalog_app.command("diff")
def catalog_diff(config: Path = ConfigOpt):
    """Show what changed upstream without saving."""
    cfg = load_config(config)
    sha, website = cat.fetch_website_data(cfg)
    hf = cat.fetch_hf_org(cfg["catalog"]["hf_org"])
    old = cat.load_catalog()
    new = cat.build_catalog(cfg, website, hf, sha, old)
    _print_diff(cat.diff_catalogs(old, new), first=old is None)


@catalog_app.command("list")
def catalog_list(
    task: list[str] = typer.Option(None, "--task", help="classification | detection | segmentation | image_text_to_text"),
    only: str = typer.Option(None, "--only", help="Regex on dataset name."),
    all_: bool = typer.Option(False, "--all", help="Include excluded datasets."),
):
    """List datasets in the catalog."""
    c = _require_catalog()
    for e in _filter(c["datasets"], task, only):
        if e["eligible"] or all_:
            reason = "" if e["eligible"] else f"  [{e['exclusion_reason']}]"
            typer.echo(f"{pipeline.TASK_DIRS.get(e['task'], e['task']):<20} {e['name']}{reason}")


def _print_diff(diff: dict, first: bool) -> None:
    if first:
        typer.echo("First sync: catalog created.")
        return
    if not (diff["added"] or diff["removed"] or diff["changed"]):
        typer.echo("No changes since last sync.")
        return
    for k in ("added", "removed"):
        if diff[k]:
            typer.echo(f"{k.capitalize()} ({len(diff[k])}):")
            for n in diff[k]:
                typer.echo(f"  {n}")
    if diff["changed"]:
        typer.echo(f"Changed ({len(diff['changed'])}):")
        for n, f in diff["changed"].items():
            typer.echo(f"  {n}  [{', '.join(f)}]")


def _print_summary(c: dict) -> None:
    el = Counter(pipeline.TASK_DIRS[e["task"]] for e in c["datasets"].values() if e["eligible"])
    ex = Counter(e["exclusion_reason"].split(":")[0] for e in c["datasets"].values() if not e["eligible"])
    typer.echo(f"\n{len(c['datasets'])} datasets, {sum(el.values())} eligible")
    for k, v in sorted(el.items()):
        typer.echo(f"  {k:<20} {v}")
    typer.echo("Excluded: " + ", ".join(f"{k}={v}" for k, v in ex.most_common()))


@app.command()
def status(
    data_dir: Path = DataDirOpt,
    config: Path = ConfigOpt,
    task: list[str] = typer.Option(None, "--task"),
    only: str = typer.Option(None, "--only"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="List every dataset."),
):
    """Show sampled / new / stale / failed datasets."""
    cfg = load_config(config)
    c = _require_catalog()
    rows = [(e, pipeline.status_of(cfg, data_dir, e)) for e in _filter(c["datasets"], task, only)]
    counts = Counter(s for _, s in rows if s != "excluded")
    typer.echo("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for e, s in rows:
        if s == "excluded" or (s == "done" and not verbose):
            continue
        extra = ""
        if s == "failed":
            extra = "  " + json.loads(pipeline.failure_path(data_dir, e["name"]).read_text())["reason"][:120]
        typer.echo(f"{s:<15} {pipeline.TASK_DIRS[e['task']]:<20} {e['name']}{extra}")


@app.command()
def sample(
    data_dir: Path = DataDirOpt,
    config: Path = ConfigOpt,
    task: list[str] = typer.Option(None, "--task", help="classification | detection | segmentation | image_text_to_text"),
    only: str = typer.Option(None, "--only", help="Regex on dataset name."),
    refresh: bool = typer.Option(False, "--refresh", help="Resample datasets that are already done."),
    retry_failed: bool = typer.Option(True, "--retry-failed/--skip-failed"),
    from_manifest: bool = typer.Option(False, "--from-manifest", help="Re-download the exact samples in manifests/."),
    timeout: int = typer.Option(None, "--timeout", help="Seconds per dataset."),
):
    """Download, subsample, and clean up each dataset."""
    cfg = load_config(config)
    c = _require_catalog()
    todo = []
    for e in _filter(c["datasets"], task, only):
        s = pipeline.status_of(cfg, data_dir, e)
        if s == "excluded":
            continue
        if from_manifest:
            if pipeline.manifest_path(e).exists():
                todo.append(e)
            continue
        if s == "done" and not refresh:
            continue
        if s == "failed" and not retry_failed:
            continue
        todo.append(e)
    if not todo:
        typer.echo("Nothing to do.")
        return
    res = pipeline.run_many(cfg, str(Path(config).resolve()), todo, Path(data_dir).resolve(),
                            timeout or cfg["download"]["timeout_s"], from_manifest)
    from agbench.progress import fmt_duration

    typer.echo(f"\n{len(res['ok'])} succeeded, {len(res['failed'])} failed in {fmt_duration(res['seconds'])}")
    for n in res["failed"]:
        typer.echo(f"  failed: {n}")


@app.command("sample-train")
def sample_train(
    data_dir: Path = DataDirOpt,
    config: Path = ConfigOpt,
    task: list[str] = typer.Option(None, "--task", help="classification | detection | segmentation"),
    only: str = typer.Option(None, "--only", help="Regex on dataset name."),
    refresh: bool = typer.Option(False, "--refresh", help="Resample datasets that are already done."),
    retry_failed: bool = typer.Option(True, "--retry-failed/--skip-failed"),
    from_manifest: bool = typer.Option(False, "--from-manifest", help="Re-download the exact samples in manifests/train-data/."),
    timeout: int = typer.Option(None, "--timeout", help="Seconds per dataset."),
):
    """Few-shot training images for the supervised baselines, never an evaluated sample (train_sampling in the config)."""
    cfg = load_config(config)
    c = _require_catalog()
    todo = []
    for e in _filter(c["datasets"], task, only):
        s = pipeline.train_status_of(cfg, data_dir, e)
        if s in ("excluded", "no_eval_samples"):
            continue
        if from_manifest:
            if pipeline.train_manifest_path(e).exists():
                todo.append(e)
            continue
        if s == "done" and not refresh:
            continue
        if s == "failed" and not retry_failed:
            continue
        todo.append(e)
    if not todo:
        typer.echo("Nothing to do.")
        return
    res = pipeline.run_many(cfg, str(Path(config).resolve()), todo, Path(data_dir).resolve(),
                            timeout or cfg["download"]["timeout_s"], from_manifest, train=True)
    from agbench.progress import fmt_duration

    typer.echo(f"\n{len(res['ok'])} succeeded, {len(res['failed'])} failed in {fmt_duration(res['seconds'])}")
    for n in res["failed"]:
        typer.echo(f"  failed: {n}")


@app.command()
def view(data_dir: Path = DataDirOpt, port: int = typer.Option(8501, "--port")):
    """Open the dataset viewer."""
    app_path = REPO_ROOT / "viewer" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port),
                    "--", "--data-dir", str(Path(data_dir).resolve())])


# ---------------------------------------------------------------- frozen benchmarks

bench_app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(bench_app, name="bench", help="Frozen, named benchmark versions (e.g. agml-full-v1).")
app.add_typer(data_app, name="data", help="Download and check a benchmark's images.")
BenchArg = typer.Argument("agml-full-v1", help="Benchmark name (agbench bench list).")


def _bundle(name: str):
    from agbench import bench

    try:
        return bench.load(name)
    except KeyError as e:
        typer.echo(e.args[0])
        raise typer.Exit(1)


@bench_app.command("list")
def bench_list():
    """Available benchmarks."""
    from agbench import bench

    for n in bench.list_bundles():
        b = bench.load(n)
        typer.echo(f"{n:<20} {b.spec['datasets']['total']:>4} datasets  {b.spec['items']['total']:>6} items  "
                   f"hash {b.hash}  {b.spec['description']}")


@bench_app.command("show")
def bench_show(name: str = BenchArg):
    """Size, tasks, judge and hash of a benchmark; checks its files against the hash."""
    import yaml

    b = _bundle(name)
    typer.echo(yaml.safe_dump({k: v for k, v in b.spec.items() if k != "files"}, sort_keys=False, allow_unicode=True).rstrip())
    problems = b.check()
    typer.echo("files: ok" if not problems else "files: " + "; ".join(problems))
    if problems:
        raise typer.Exit(1)


@bench_app.command("freeze")
def bench_freeze(
    name: str = typer.Argument(..., help="New benchmark name, e.g. agml-full-v2."),
    items: Path = typer.Option(..., "--items", help="items.jsonl of a run over the full base-data (agbench eval run)."),
    description: str = typer.Option(..., "--description"),
    config: Path = ConfigOpt,
):
    """Freeze the committed manifests and a run's items into benchmarks/<name>/ (never overwrites one)."""
    from agbench import bench

    b = bench.freeze(name, items, description, config_path=config)
    typer.echo(f"{b.path}  hash {b.hash}  {b.spec['items']['total']} items, {b.spec['datasets']['total']} datasets")


@bench_app.command("repin")
def bench_repin(
    name: str = typer.Argument(..., help="Benchmark name."),
    dataset: list[str] = typer.Option(..., "--dataset", help="Dataset whose source repo was re-uploaded (repeatable)."),
    reference: Path = typer.Option(..., "--reference", help="Data dir holding the benchmark's images (the frozen copy)."),
    write: bool = typer.Option(False, "--write", help="Update the bundle if every sample matched (default: report only)."),
):
    """Re-point datasets at a re-uploaded repo's current revision; samples must match the reference copy exactly."""
    from agbench import bench

    bad = False
    from agbench.pipeline import explain

    for ds in dataset:
        b = _bundle(name)
        try:
            r = bench.repin(b, ds, reference, write)
        except Exception as e:
            typer.echo(f"{ds}: NOT re-pinned: {explain(e, b.entry(ds)['hf_repo'], None)}")
            bad = True
            continue
        status = "written" if r.get("written") else ("ok (dry run)" if r["ok"] else "NOT re-pinned")
        typer.echo(f"{ds}: {r['samples']} samples, {r['old_revision'][:7]} -> {r['new_revision'][:7]}: {status}")
        for k, why in (("unmatched", "no row with these images"), ("different", "downloaded images not byte-identical")):
            if r[k]:
                typer.echo(f"  {len(r[k])} {why} (e.g. {', '.join(r[k][:3])})")
        if r["text_changed"]:
            typer.echo(f"  {r['text_changed']} rows with different conversation text (frozen items unaffected)")
        bad |= not r["ok"]
    if bad:
        raise typer.Exit(1)


@data_app.command("pull")
def data_pull(
    name: str = BenchArg,
    data_dir: Path = DataDirOpt,
    task: list[str] = typer.Option(None, "--task", help="classification | detection | segmentation | image_text_to_text"),
    only: str = typer.Option(None, "--only", help="Regex on dataset name."),
    refresh: bool = typer.Option(False, "--refresh", help="Download datasets again even if present."),
    timeout: int = typer.Option(None, "--timeout", help="Seconds per dataset."),
):
    """Download a benchmark's images from the source AgML repos at their pinned revisions."""
    from agbench import bench
    from agbench.progress import fmt_duration

    b = _bundle(name)
    cfg = load_config(b.config_path)
    status = bench.data_status(b, data_dir, task, only)
    gone = [n for n, s in status.items() if s == "unavailable"]
    todo = [b.entry(n) for n, s in status.items() if s not in ("unavailable", "done") or (refresh and n not in b.unavailable)]
    typer.echo(f"{name}: {len(status)} datasets selected, {len(status) - len(todo) - len(gone)} already present, {len(todo)} to download"
               + (f", {len(gone)} unavailable" if gone else ""))
    _print_unavailable(b, gone)
    if not todo:
        return
    res = pipeline.run_many(cfg, str(b.config_path), todo, Path(data_dir).resolve(),
                            timeout or cfg["download"]["timeout_s"], True, bundle=name)
    typer.echo(f"\n{len(res['ok'])} downloaded, {len(res['failed'])} failed in {fmt_duration(res['seconds'])}")
    _print_failures(data_dir, res["failed"])
    if res.get("log"):
        typer.echo(f"Details: {res['log']}")
    if res["failed"]:
        typer.echo("Rerun the same command to retry the failed datasets (finished ones are skipped).")
        raise typer.Exit(1)


def _print_unavailable(b, names: list[str]) -> None:
    if not names:
        return
    n_items = Counter(r["dataset"] for r in b.raw_items)
    typer.echo("Unavailable (kept in the benchmark; skipped without a local copy):")
    for n in names:
        typer.echo(f"  {n}  {b.unavailable[n]}; {n_items[n]} items")


def _print_failures(data_dir: Path, names: list[str]) -> None:
    if not names:
        return
    typer.echo("\nFailed:")
    w = max(len(n) for n in names)
    for n in names:
        p = pipeline.failure_path(Path(data_dir), n)
        reason = json.loads(p.read_text())["reason"] if p.exists() else "unknown (no failure record)"
        typer.echo(f"  {n:<{w}}  {reason}")


@data_app.command("status")
def data_status(name: str = BenchArg, data_dir: Path = DataDirOpt, task: list[str] = typer.Option(None, "--task"),
                only: str = typer.Option(None, "--only"), verbose: bool = typer.Option(False, "-v", "--verbose")):
    """Which of a benchmark's datasets are present (done), missing, stale or failed in a data dir."""
    from agbench import bench

    b = _bundle(name)
    st = bench.data_status(b, data_dir, task, only)
    typer.echo("  ".join(f"{k}={v}" for k, v in sorted(Counter(st.values()).items())))
    for n, s in st.items():
        if s == "unavailable":
            typer.echo(f"{s:<11} {n}: {b.unavailable[n]}")
        elif s == "failed":
            reason = json.loads(pipeline.failure_path(Path(data_dir), n).read_text())["reason"]
            typer.echo(f"{s:<11} {n}: {reason}")
        elif s != "done" or verbose:
            typer.echo(f"{s:<11} {n}")


@data_app.command("verify")
def data_verify(name: str = BenchArg, data_dir: Path = DataDirOpt, task: list[str] = typer.Option(None, "--task"),
                only: str = typer.Option(None, "--only"),
                decode: bool = typer.Option(True, "--decode/--no-decode", help="Open every image (slower).")):
    """Check a local copy against the benchmark: revision, sample ids, every evaluated image present and readable."""
    from agbench import bench

    b = _bundle(name)
    problems = b.check()
    if problems:
        typer.echo("benchmark files: " + "; ".join(problems))
    bad = bench.verify(b, data_dir, task, only, decode)
    n = len(b.datasets(task, only))
    for ds, errs in bad.items():
        typer.echo(f"{ds}: " + "; ".join(errs[:5]) + (f" (+{len(errs) - 5} more)" if len(errs) > 5 else ""))
    gone = [d for d, s in bench.data_status(b, data_dir, task, only).items() if s == "unavailable"]
    typer.echo(f"{n - len(bad) - len(gone)} of {n} datasets match {name} (hash {b.hash})")
    _print_unavailable(b, gone)
    if bad or problems:
        raise typer.Exit(1)


from agbench.eval.cli import app as eval_app  # noqa: E402

app.add_typer(eval_app, name="eval")
