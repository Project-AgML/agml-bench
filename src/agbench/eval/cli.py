import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="Benchmark models on base-data.")

DataDirOpt = typer.Option(..., "--data-dir", help="Data dir with base-data/.")
TaskOpt = typer.Option(None, "--task", help="classification | detection | segmentation | image_text_to_text")
FormatOpt = typer.Option(None, "--format", help="mcqa_within | mcqa_across | oeq | grounding | native")


def _run_dir(out: Path, name: str) -> Path:
    return out / f"{name.replace('/', '__')}_{datetime.now():%Y%m%d-%H%M%S}"


def _load_method(path: str, kwargs: str | None):
    mod, _, cls = path.partition(":")
    return getattr(importlib.import_module(mod), cls)(**json.loads(kwargs or "{}"))


@app.command("models")
def models_cmd():
    """List registered models with VRAM estimates."""
    from agbench.eval.resources import models

    reg = models()
    typer.echo(f"{'model':<52} {'backend':<9} {'weights GB':>10} {'est. VRAM GB':>13}")
    for m, s in reg["models"].items():
        typer.echo(f"{m:<52} {s['backend']:<9} {s.get('weights_gb', ''):>10} {s.get('est_vram_gb', ''):>13}")
    typer.echo(f"\njudge: {reg['judge']}")


@app.command()
def run(
    data_dir: Path = DataDirOpt,
    model: str = typer.Option(None, "--model", help="Registry model id (see `agbench eval models`)."),
    method: str = typer.Option(None, "--method", help="Custom method as module:Class."),
    method_kwargs: str = typer.Option(None, "--method-kwargs", help="JSON kwargs for --method."),
    endpoint: str = typer.Option(None, "--endpoint", help="Use a running OpenAI-compatible server instead of launching one."),
    task: list[str] = TaskOpt,
    fmt: list[str] = FormatOpt,
    only: str = typer.Option(None, "--only", help="Regex on dataset name."),
    limit_per_dataset: int = typer.Option(None, "--limit-per-dataset"),
    concurrency: int = typer.Option(None, "--concurrency"),
    thinking: bool = typer.Option(True, "--thinking/--no-thinking"),
    effort: str = typer.Option("medium", "--reasoning-effort", help="low | medium | high"),
    max_tokens: int = typer.Option(None, "--max-tokens"),
    max_model_len: int = typer.Option(None, "--max-model-len"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("runs"), "--out"),
    run_dir: Path = typer.Option(None, "--run-dir", help="Resume into this run dir."),
    judge: bool = typer.Option(False, "--judge", help="Judge open-ended answers after the run."),
    judge_endpoint: str = typer.Option(None, "--judge-endpoint"),
    gguf_backend: str = typer.Option("llamacpp", "--gguf-backend", help="llamacpp | vllm (vllm needs vllm-gguf-plugin)"),
):
    """Run a model or custom method over the benchmark."""
    from agbench.eval.methods import VLMMethod, model_spec
    from agbench.eval.runner import Benchmark
    from agbench.eval.servers import Server

    if not (model or method):
        raise typer.BadParameter("pass --model or --method")
    bench = Benchmark(data_dir, task or None, fmt or None, only, limit_per_dataset, seed)
    typer.echo(f"{len(bench.items)} items; {len(bench.skipped)} datasets skipped")
    if method:
        m = _load_method(method, method_kwargs)
        rd = run_dir or _run_dir(out, m.name)
        res = bench.run(m, rd, concurrency or 8)
    else:
        spec = model_spec(model)
        llama = spec["backend"] == "llamacpp" and gguf_backend == "llamacpp"
        conc = concurrency or (spec.get("parallel", 2) if llama else spec["concurrency"])
        rd = run_dir or _run_dir(out, model)
        kf = spec.get("bbox_key_format")
        if endpoint:
            res = bench.run(VLMMethod(endpoint, model, thinking, effort, max_tokens), rd, conc, key_formats=kf)
        else:
            with Server(model, rd / "server.log", max_model_len=max_model_len, gguf_backend=gguf_backend) as srv:
                res = bench.run(VLMMethod(srv.endpoint, model, thinking, effort, max_tokens), rd, conc, key_formats=kf)
    if judge and res.metrics["pending_judge"]:
        _judge(rd, judge_endpoint, None)
    _print_summary(rd)


def _judge(run_dir: Path, endpoint: str | None, judge_model: str | None):
    from agbench.eval.judge import judge_run
    from agbench.eval.resources import models
    from agbench.eval.servers import Server

    jm = judge_model or models()["judge"]
    if endpoint:
        judge_run(run_dir, endpoint, jm)
    else:
        with Server(jm, run_dir / "judge_server.log") as srv:
            judge_run(run_dir, srv.endpoint, jm)


@app.command("judge")
def judge_cmd(
    run_dir: list[Path] = typer.Option(..., "--run", help="Run dir(s) to judge."),
    judge_model: str = typer.Option(None, "--judge-model"),
    endpoint: str = typer.Option(None, "--judge-endpoint"),
):
    """Judge open-ended answers with the judge model (default gemma-4-31B-it-FP8)."""
    from agbench.eval.judge import judge_run
    from agbench.eval.resources import models
    from agbench.eval.servers import Server

    jm = judge_model or models()["judge"]
    if endpoint:
        for d in run_dir:
            judge_run(d, endpoint, jm)
    else:
        with Server(jm, run_dir[0] / "judge_server.log") as srv:
            for d in run_dir:
                judge_run(d, srv.endpoint, jm)


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
    effort: str = typer.Option("medium", "--reasoning-effort"),
    judge: bool = typer.Option(True, "--judge/--no-judge"),
    cleanup_weights: bool = typer.Option(False, "--cleanup-weights", help="Delete each model's weights after profiling it."),
    out: Path = typer.Option(Path("reports/profile"), "--out"),
    gguf_backend: str = typer.Option("llamacpp", "--gguf-backend", help="llamacpp | vllm (vllm needs vllm-gguf-plugin)"),
):
    """Measure VRAM, latency and throughput per model and task; estimate full-run compute."""
    from agbench.eval.profile import run_profile
    from agbench.eval.resources import models as reg

    ids = list(reg()["models"]) if models == ["all"] else models
    run_profile(ids, data_dir, out, per_task, per_task_seq, concurrency, thinking, effort, judge, cleanup_weights, gguf_backend)


@app.command()
def view(
    runs: list[Path] = typer.Option([Path("runs")], "--runs"),
    profile_dir: Path = typer.Option(None, "--profile"),
    port: int = typer.Option(8502, "--port"),
):
    """Open the results viewer."""
    app_path = Path(__file__).parent / "viewer.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--",
           "--runs", *[str(Path(r).resolve()) for r in runs]]
    if profile_dir:
        cmd += ["--profile", str(profile_dir.resolve())]
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
