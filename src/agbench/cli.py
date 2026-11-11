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


from agbench.eval.cli import app as eval_app  # noqa: E402

app.add_typer(eval_app, name="eval")
