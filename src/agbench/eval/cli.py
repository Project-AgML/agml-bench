import importlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="Benchmark models on base-data.")

DataDirOpt = typer.Option(..., "--data-dir", help="Data dir with base-data/.")
TaskOpt = typer.Option(None, "--task", help="classification | detection | segmentation | image_text_to_text")
FormatOpt = typer.Option(None, "--format", help="mcqa_within | mcqa_across | oeq | grounding | native | multiple_choice (ITT)")
ModelsFileOpt = typer.Option(None, "--models-file", help="YAML of extra/overriding models (models.yaml layout).")


def _register(files: list[Path] | None):
    from agbench.eval.resources import register_models

    for f in files or []:
        register_models(f)


def _run_dir(out: Path, name: str) -> Path:
    return out / f"{name.replace('/', '__')}_{datetime.now():%Y%m%d-%H%M%S}"


def _load_method(path: str, kwargs: str | None):
    mod, _, cls = path.partition(":")
    return getattr(importlib.import_module(mod), cls)(**json.loads(kwargs or "{}"))


@app.command("models")
def models_cmd(models_file: list[Path] = ModelsFileOpt):
    """List registered models with VRAM estimates."""
    from agbench.eval.resources import models

    _register(models_file)
    reg = models()
    typer.echo(f"{'model':<52} {'backend':<9} {'weights GB':>10} {'est. VRAM GB':>13}")
    for m, s in reg["models"].items():
        typer.echo(f"{m:<52} {s['backend']:<9} {s.get('weights_gb', ''):>10} {s.get('est_vram_gb', ''):>13}")
    typer.echo(f"\njudge: {reg['judge']}")


@app.command()
def run(
    data_dir: Path = DataDirOpt,
    model: str = typer.Option(None, "--model", help="Model id: registry entry, any HF repo or local path (served with vLLM)."),
    method: str = typer.Option(None, "--method", help="Custom method as module:Class."),
    method_kwargs: str = typer.Option(None, "--method-kwargs", help="JSON kwargs for --method."),
    endpoint: str = typer.Option(None, "--endpoint", help="Use a running OpenAI-compatible server instead of launching one."),
    task: list[str] = TaskOpt,
    fmt: list[str] = FormatOpt,
    only: str = typer.Option(None, "--only", help="Regex on dataset name."),
    limit_per_dataset: int = typer.Option(None, "--limit-per-dataset"),
    concurrency: int = typer.Option(None, "--concurrency"),
    thinking: bool = typer.Option(True, "--thinking/--no-thinking"),
    effort: str = typer.Option("low", "--reasoning-effort", help="low | medium | high"),
    max_tokens: int = typer.Option(None, "--max-tokens"),
    max_model_len: int = typer.Option(None, "--max-model-len"),
    seed: int = typer.Option(0, "--seed", help="Item seed: which images, distractors and NOTA items."),
    gen_seed: int = typer.Option(None, "--gen-seed", help="Sampling seed (per item); replicate runs use different values."),
    order_seed: int = typer.Option(None, "--order-seed", help="Multiple choice: reshuffle only the option order."),
    nota_position: str = typer.Option("end", "--nota-position", help="Multiple choice: end | random (NOTA/IDK placed among the options)."),
    nota_rate: float = typer.Option(None, "--nota-rate", help="Multiple choice: share of items whose true option is removed (default 0.3 classification, 0 ITT)."),
    idk: bool = typer.Option(True, "--idk/--no-idk", help="Multiple choice: offer \"I don't know\"."),
    out: Path = typer.Option(Path("runs"), "--out"),
    run_dir: Path = typer.Option(None, "--run-dir", help="Resume into this run dir."),
    judge: bool = typer.Option(False, "--judge", help="Judge open-ended answers after the run."),
    judge_endpoint: str = typer.Option(None, "--judge-endpoint"),
    models_file: list[Path] = ModelsFileOpt,
):
    """Run a model or custom method over the benchmark."""
    _register(models_file)
    from agbench.eval.gpu import sampler
    from agbench.eval.items import MCQAOptions
    from agbench.eval.methods import VLMMethod, key_formats, model_spec
    from agbench.eval.runner import Benchmark

    if not (model or method):
        raise typer.BadParameter("pass --model or --method")
    if nota_position not in ("end", "random"):
        raise typer.BadParameter("--nota-position: end | random")
    bench = Benchmark(data_dir, task or None, fmt or None, only, limit_per_dataset, seed,
                      mcqa=MCQAOptions(nota_rate, order_seed, nota_position, idk))
    typer.echo(f"{len(bench.items)} items; {len(bench.skipped)} datasets skipped")
    if method:
        m = _load_method(method, method_kwargs)
        rd = run_dir or _run_dir(out, m.name)
        res = bench.run(m, rd, concurrency or 8)
    else:
        spec = model_spec(model)
        conc = concurrency or (spec.get("parallel", 2) if spec["backend"] == "llamacpp" else spec["concurrency"])
        rd = run_dir or _run_dir(out, model)
        kf = key_formats(spec)
        if run_dir and not bench.remaining(rd):
            from agbench.eval.runner import finalize, read_jsonl

            typer.echo("all items already done; skipping the model server")
            ids = {it.item_id for it in bench.items}
            res = finalize(rd, [r for r in read_jsonl(rd / "predictions.jsonl") if r["item_id"] in ids])
        elif endpoint:
            res = bench.run(VLMMethod(endpoint, model, thinking, effort, max_tokens, gen_seed=gen_seed), rd, conc, key_formats=kf)
        else:
            with sampler() as gpu, _served(model, rd, "server.log", gpu, max_model_len=max_model_len) as srv:
                res = bench.run(VLMMethod(srv.endpoint, model, thinking, effort, max_tokens, gen_seed=gen_seed), rd, conc,
                                key_formats=kf, gpu=gpu)
    if judge and res.metrics["pending_judge"]:
        _judge([rd], judge_endpoint, None)
    _print_summary(rd)


@contextmanager
def _served(model_id: str, run_dir: Path, log_name: str, gpu, phase: str = "load", max_model_len: int | None = None):
    """Start a model server, logging its load time and GPU use to run_dir/compute.json."""
    from agbench.eval.runner import log_compute
    from agbench.eval.servers import Server

    t0 = time.time()
    with Server(model_id, run_dir / log_name, max_model_len=max_model_len) as srv:
        log_compute(run_dir, phase=phase, model=model_id, wall_s=round(srv.load_seconds, 1),
                    **(gpu.summary(t0, time.time()) if gpu else {}))
        yield srv


def _judge(run_dirs: list[Path], endpoint: str | None, judge_model: str | None, redo: bool = False):
    from agbench.eval.gpu import sampler
    from agbench.eval.judge import judge_run
    from agbench.eval.resources import models

    jm = judge_model or models()["judge"]
    if endpoint:
        for d in run_dirs:
            judge_run(d, endpoint, jm, redo=redo)
        return
    with sampler() as gpu, _served(jm, run_dirs[0], "judge_server.log", gpu, "judge_load") as srv:
        for d in run_dirs:
            judge_run(d, srv.endpoint, jm, gpu=gpu, redo=redo)


@app.command("judge")
def judge_cmd(
    run_dir: list[Path] = typer.Option(..., "--run", help="Run dir(s) to judge."),
    judge_model: str = typer.Option(None, "--judge-model"),
    endpoint: str = typer.Option(None, "--judge-endpoint"),
    redo: bool = typer.Option(False, "--redo", help="Re-judge answers that already have a verdict (e.g. after editing the judge prompts)."),
    models_file: list[Path] = ModelsFileOpt,
):
    """Judge open-ended answers with the judge model (default gemma-4-31B-it-FP8)."""
    _register(models_file)
    _judge(run_dir, endpoint, judge_model, redo)


def _print_summary(run_dir: Path):
    from agbench.eval.report import leaderboard_md, load_runs

    typer.echo(leaderboard_md(load_runs([run_dir])))
    typer.echo(f"results: {run_dir}")


@app.command()
def report(
    runs: list[Path] = typer.Option([Path("runs")], "--runs"),
    average: str = typer.Option("macro", "--average", help="macro | micro"),
    out: Path = typer.Option(None, "--out", help="Write the leaderboard markdown here."),
):
    """Leaderboard across runs."""
    from agbench.eval.report import leaderboard_md, load_runs

    md = leaderboard_md(load_runs(runs), average)
    typer.echo(md)
    if out:
        out.write_text(md)


@app.command()
def profile(
    data_dir: Path = DataDirOpt,
    models: list[str] = typer.Option(["all"], "--models", help="Model ids, or 'all'."),
    per_task: int = typer.Option(20, "--per-task", help="Items per task (per format for classification), batched."),
    per_task_seq: int = typer.Option(5, "--per-task-seq", help="Items per task run one at a time for latency."),
    concurrency: int = typer.Option(None, "--concurrency"),
    thinking: bool = typer.Option(True, "--thinking/--no-thinking"),
    effort: str = typer.Option("low", "--reasoning-effort"),
    judge: bool = typer.Option(True, "--judge/--no-judge"),
    cleanup_weights: bool = typer.Option(False, "--cleanup-weights", help="Delete each model's weights after profiling it."),
    out: Path = typer.Option(Path("reports/profile"), "--out"),
    redo: bool = typer.Option(False, "--redo", help="Re-profile models already in <out>/profile.json."),
    efforts: list[str] = typer.Option(None, "--efforts", help="Profile each of these per model, loaded once: off | low | medium | high."),
    models_file: list[Path] = ModelsFileOpt,
):
    """Measure VRAM, latency and throughput per model and task; estimate full-run compute."""
    from agbench.eval.profile import run_profile
    from agbench.eval.resources import models as reg

    _register(models_file)
    ids = list(reg()["models"]) if models == ["all"] else models
    if bad := set(efforts or []) - {"off", "low", "medium", "high"}:
        raise typer.BadParameter(f"--efforts: unknown {sorted(bad)}")
    run_profile(ids, data_dir, out, per_task, per_task_seq, concurrency, thinking, effort, judge, cleanup_weights, redo,
                efforts or None)


@app.command()
def stats(
    runs: list[Path] = typer.Option(..., "--runs", help="Run dirs (or parents) to analyse together, e.g. the seeds of every model."),
    out: Path = typer.Option(Path("reports/stats"), "--out"),
    title: str = typer.Option("MCQA statistics", "--title"),
):
    """NOTA/IDK behaviour, seed and option-order stability, and model-vs-model tests, with dataset-bootstrap CIs."""
    from agbench.eval.stats import write_report

    typer.echo(f"report: {write_report(runs, out, title)}")


@app.command()
def view(
    runs: list[Path] = typer.Option([Path("runs")], "--runs"),
    profile_dir: Path = typer.Option(None, "--profile"),
    data_dir: Path = typer.Option(None, "--data-dir", help="Folder with base-data/, if images live elsewhere than when the runs were made."),
    port: int = typer.Option(8502, "--port"),
):
    """Open the results viewer."""
    app_path = Path(__file__).parent / "viewer.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--",
           "--runs", *[str(Path(r).resolve()) for r in runs]]
    if profile_dir:
        cmd += ["--profile", str(profile_dir.resolve())]
    if data_dir:
        cmd += ["--data-dir", str(data_dir.resolve())]
    subprocess.run(cmd)


@app.command("check-agtasks")
def check_agtasks():
    """List datasets/classes whose ag-task mapping needs a look (uses catalog/catalog.json)."""
    from agbench import catalog as cat
    from agbench.eval.agtask import check

    c = cat.load_catalog()
    if c is None:
        typer.echo("No catalog. Run: agbench catalog sync")
        raise typer.Exit(1)
    problems = check([e for e in c["datasets"].values() if e["eligible"]])
    for p in problems:
        typer.echo(p)
    typer.echo(f"{len(problems)} to review")
