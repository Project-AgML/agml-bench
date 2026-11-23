import importlib
import json
import re
import shutil
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


@app.command()
def baseline(
    data_dir: Path = DataDirOpt,
    kind: str = typer.Option(..., "--kind", help="chance | prior (see agbench/eval/baselines.py)"),
    task: list[str] = TaskOpt,
    fmt: list[str] = FormatOpt,
    only: str = typer.Option(None, "--only", help="Regex on dataset name."),
    seed: int = typer.Option(0, "--seed", help="Item seed: which images, distractors and NOTA items."),
    gen_seed: int = typer.Option(0, "--gen-seed", help="chance: random-answer seed; replicate runs use different values."),
    out: Path = typer.Option(Path("runs"), "--out"),
):
    """Naive baselines (no image used), scored like a model run; no GPU or judge needed."""
    from agbench.eval.baselines import KINDS, NaiveBaseline, exact_oeq_verdicts, priors
    from agbench.eval.runner import Benchmark, finalize, read_jsonl, write_jsonl

    if kind not in KINDS:
        raise typer.BadParameter(f"--kind: {' | '.join(KINDS)}")
    bench = Benchmark(data_dir, task or None, fmt or None, only, None, seed)
    typer.echo(f"{len(bench.items)} items; {len(bench.skipped)} datasets skipped")
    m = NaiveBaseline(kind, priors(bench.items), gen_seed)
    rd = _run_dir(out, f"{m.name}_s{gen_seed}" if kind == "chance" else m.name)
    bench.run(m, rd, concurrency=64, resume=False)
    recs = exact_oeq_verdicts(read_jsonl(rd / "predictions.jsonl"))
    write_jsonl(rd / "predictions.jsonl", recs)
    finalize(rd, recs)
    _print_summary(rd)


def _claimer(claims: Path):
    """claim(key) -> True if this process now owns claims/key. A claim whose owner (a Slurm job, or a process on this
    host) is gone is stale and is taken over, so work held by a job that died is picked up again."""
    import os
    import socket

    me = f"job:{os.environ['SLURM_JOB_ID']}" if os.environ.get("SLURM_JOB_ID") else f"pid:{socket.gethostname()}:{os.getpid()}"

    def alive(owner: str) -> bool:
        kind, _, rest = owner.partition(":")
        if kind == "job":
            try:
                r = subprocess.run(["squeue", "-h", "-j", rest, "-o", "%T"], capture_output=True, text=True, timeout=30)
                if r.returncode != 0:  # a job Slurm has already purged is "Invalid job id": dead; any other error: assume alive
                    return "invalid job id" not in (r.stderr + r.stdout).lower()
                return bool(r.stdout.strip())
            except Exception:
                return True
        host, _, pid = rest.partition(":")
        if host != socket.gethostname():
            return True
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False

    def claim(key: str) -> bool:
        p = claims / key
        for _ in range(2):
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, me.encode()); os.close(fd)
                return True
            except FileExistsError:
                try:
                    owner = p.read_text().strip()
                except FileNotFoundError:
                    continue
                if owner and alive(owner):
                    return False
                p.unlink(missing_ok=True)  # stale claim from a job or process that died
        return False

    return claim


@app.command()
def yolo(
    data_dir: Path = DataDirOpt,
    shots: int = typer.Option(..., "--shots", help="Training images per class (a shot count in train_sampling)."),
    subset: int = typer.Option(0, "--subset", help="Training subset (0, 1, 2): which seeded draw of training images."),
    train_dir: Path = typer.Option(None, "--train-dir", help="Few-shot training data (default <data-dir>/train-data)."),
    task: list[str] = typer.Option(None, "--task", help="classification | detection | segmentation (default all three)"),
    only: str = typer.Option(None, "--only", help="Regex on dataset name."),
    size: str = typer.Option("x", "--size", help="YOLO11 size: n | s | m | l | x"),
    epochs: int = typer.Option(100, "--epochs"),
    out: Path = typer.Option(Path("runs/yolo"), "--out"),
    weights_dir: Path = typer.Option(Path.home() / ".cache" / "agbench" / "yolo", "--weights-dir", help="Pretrained YOLO11 weights."),
):
    """Few-shot YOLO11 baseline: per dataset, train on k images per class, score the evaluation items, delete the model.

    Several processes (on several GPUs or jobs) can work on the same run at once: each dataset is claimed with a lock
    file, its results go to their own file under parts/, and the process that finds every dataset done merges them.
    Exit code 0 when the run is complete, 3 when other processes still hold datasets of it."""
    import fcntl
    import os
    import time

    from agbench.eval import yolo as Y
    from agbench.eval.baselines import exact_oeq_verdicts
    from agbench.eval.gpu import sampler
    from agbench.eval.grade import grade
    from agbench.eval.runner import Benchmark, coverage_md, crop_table, finalize, read_jsonl, _git_sha

    dev = Y.device()  # fails fast if CUDA is unavailable, instead of every dataset silently training on the CPU
    typer.echo(f"YOLO device: {dev}")
    tasks = task or ["classification", "detection", "segmentation"]
    train_root = Path(train_dir or Path(data_dir) / "train-data")
    bench = Benchmark(data_dir, tasks, None, only, None, 0)
    names = Y.dataset_names(bench.items)
    label = f"yolo11{size}/{shots}shot"
    rd = out / f"{shots}shot" / f"s{subset}"
    parts, claims = rd / "parts", rd / "claims"
    for d in (rd, parts, claims):
        d.mkdir(parents=True, exist_ok=True)
    def write_once(path: Path, text: str):  # identical for every process; atomic, never half-written
        if not path.exists():
            tmp = path.with_name(f".{path.name}.{os.getpid()}")
            tmp.write_text(text)
            tmp.replace(path)

    write_once(rd / "run.json", json.dumps({
        "method": {"name": label, "baseline": "yolo", "size": size, "epochs": epochs, "shots": shots, "gen_seed": subset},
        "benchmark": bench.args, "n_items": len(bench.items), "agbench_git": _git_sha(),
        "started": datetime.now().astimezone().isoformat(timespec="seconds")}, indent=2))
    write_once(rd / "items.jsonl", "".join(json.dumps(it.to_dict()) + "\n" for it in bench.items))
    write_once(rd / "crops.json", json.dumps(crop_table(bench.items), indent=2))
    write_once(rd / "ag_task_coverage.md", coverage_md(bench.items))

    claim = _claimer(claims)

    by_ds: dict = {}
    for it in bench.items:
        by_ds.setdefault((it.task, it.dataset), []).append(it)
    key_of = lambda t, ds: f"{t}__{ds}"  # noqa: E731

    def finished() -> tuple[set, set]:
        done = {r["item_id"] for r in read_jsonl(rd / "predictions.jsonl")}
        for f in parts.glob("*.jsonl"):
            done |= {r["item_id"] for r in read_jsonl(f)}
        skipped = {f.name[: -len(".skipped.json")] for f in parts.glob("*.skipped.json")}
        return done, skipped

    done, skipped = finished()
    for i, ((t, ds), its) in enumerate(sorted(by_ds.items()), 1):
        key = key_of(t, ds)
        if key in skipped or all(it.item_id in done for it in its) or not claim(key):
            continue
        try:
            done, skipped = finished()  # another process may have finished it between the scan and the claim
            if key in skipped or all(it.item_id in done for it in its):
                continue
            typer.echo(f"[{i}/{len(by_ds)}] {t}/{ds}: {len(its)} items")
            ids = Y.subset_ids(train_root, t, ds, subset, shots)
            tr = Y.train_items(train_root, t, f"^{re.escape(ds)}$", ids) if ids else []
            if not tr:
                (parts / f"{key}.skipped.json").write_text(json.dumps({"dataset": ds, "task": t, "reason": "no training images"}))
                continue
            work = rd / "_work" / f"{ds}.{os.getpid()}"
            try:
                with sampler() as g:
                    t0 = time.time()
                    data = Y.prepare(t, tr, names[ds], work / "data", train_root / t / ds)
                    w = Y.train(t, data, work / "model", size, epochs, weights_dir, seed=subset)
                    t1 = time.time()
                    m = Y.YOLOMethod({ds: w}, names, label, subset)
                    recs = [grade(it, m.predict(it), {"bbox": "xyxy_px"}) for it in its]
                    t2 = time.time()
                end_t = datetime.now().astimezone().isoformat(timespec="seconds")
                comp = [{"ended": end_t, "phase": "train", "model": label, "dataset": ds, "train_images": len(tr), "wall_s": round(t1 - t0, 1),
                         **(g.summary(t0, t1) if g else {})},
                        {"ended": end_t, "phase": "generate", "model": label, "dataset": ds, "items": len(its), "wall_s": round(t2 - t1, 1),
                         **(g.summary(t1, t2) if g else {})}]
                tmp = parts / f".{key}.{os.getpid()}"
                tmp.write_text("".join(json.dumps(r) + "\n" for r in exact_oeq_verdicts(recs)))
                (parts / f"{key}.compute.json").write_text(json.dumps(comp))
                tmp.replace(parts / f"{key}.jsonl")  # results appear only once complete
            except Exception as e:  # one dataset failing must not end the run; after 2 failures it is skipped
                typer.echo(f"  failed: {type(e).__name__}: {e}")
                with open(rd / "failures.jsonl", "a") as f:
                    f.write(json.dumps({"dataset": ds, "task": t, "time": datetime.now().isoformat(timespec="seconds"),
                                        "error": f"{type(e).__name__}: {e}"[:500]}) + "\n")
                fails = sum(1 for r in read_jsonl(rd / "failures.jsonl") if r["dataset"] == ds and r["task"] == t)
                if fails >= 2:
                    (parts / f"{key}.skipped.json").write_text(json.dumps({"dataset": ds, "task": t, "reason": f"failed twice: {type(e).__name__}"}))
            finally:
                shutil.rmtree(work, ignore_errors=True)  # the trained model is not kept: thousands of them would not fit
        finally:
            (claims / key).unlink(missing_ok=True)

    # Complete only when every dataset is scored or skipped; the process that sees this merges the parts.
    done, skipped = finished()
    left = [key_of(t, ds) for (t, ds), its in by_ds.items() if key_of(t, ds) not in skipped and not all(it.item_id in done for it in its)]
    if left:
        typer.echo(f"{len(left)} datasets still in progress elsewhere or failed once; run again to continue")
        raise typer.Exit(3)
    with open(rd / ".merge.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ids = {it.item_id for it in bench.items}
        recs = {r["item_id"]: r for r in read_jsonl(rd / "predictions.jsonl")}
        for f in sorted(parts.glob("*.jsonl")):
            recs.update({r["item_id"]: r for r in read_jsonl(f)})
        merged = [recs[i] for i in (it.item_id for it in bench.items) if i in recs]
        tmp = rd / f".predictions.{os.getpid()}"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in merged))
        tmp.replace(rd / "predictions.jsonl")
        comp = json.loads((rd / "compute.json").read_text()) if (rd / "compute.json").exists() else []
        seen = {(c.get("dataset"), c.get("phase"), c.get("ended")) for c in comp}
        for f in sorted(parts.glob("*.compute.json")):
            comp += [c for c in json.loads(f.read_text()) if (c.get("dataset"), c.get("phase"), c.get("ended")) not in seen]
        (rd / "compute.json").write_text(json.dumps(comp, indent=2))
        (rd / "skipped.json").write_text(json.dumps([json.loads(f.read_text()) for f in sorted(parts.glob("*.skipped.json"))], indent=2))
        finalize(rd, [r for r in merged if r["item_id"] in ids])
    _print_summary(rd)

@app.command("yolo-shared")
def yolo_shared(
    data_dir: Path = DataDirOpt,
    shots: int = typer.Option(..., "--shots", help="Training images per class (a shot count in train_sampling)."),
    subset: int = typer.Option(0, "--subset", help="Training subset (0, 1, 2): which seeded draw of training images."),
    train_dir: Path = typer.Option(None, "--train-dir", help="Few-shot training data (default <data-dir>/train-data)."),
    task: list[str] = typer.Option(None, "--task", help="classification | detection | segmentation (default all three)"),
    only: str = typer.Option(None, "--only", help="Regex on dataset name (for tests)."),
    size: str = typer.Option("s", "--size", help="YOLO11 size: n | s | m | l | x"),
    epochs: int = typer.Option(100, "--epochs"),
    out: Path = typer.Option(Path("runs/yolo11s-shared"), "--out"),
    weights_dir: Path = typer.Option(Path.home() / ".cache" / "agbench" / "yolo", "--weights-dir", help="Pretrained YOLO11 weights."),
):
    """Few-shot YOLO11 baseline with one model per task: train on the k-shot training images of every dataset of the
    task together (classes merged by normalised name; names without letters stay per dataset), then score every
    evaluation item with predictions limited to its own dataset's classes.

    Each task is one unit of work, claimed with a lock file, so several processes (GPUs, jobs) can share a run.
    Exit code 0 when the run is complete, 3 when other processes still hold tasks of it."""
    import fcntl
    import os

    from agbench.eval import yolo as Y
    from agbench.eval.baselines import exact_oeq_verdicts
    from agbench.eval.gpu import sampler
    from agbench.eval.grade import grade
    from agbench.eval.runner import Benchmark, coverage_md, crop_table, finalize, read_jsonl, _git_sha

    dev = Y.device()
    typer.echo(f"YOLO device: {dev}")
    os.environ.setdefault("AGBENCH_YOLO_WORKERS", "8")  # one large training per GPU: more data-loading processes
    tasks = task or ["classification", "detection", "segmentation"]
    train_root = Path(train_dir or Path(data_dir) / "train-data")
    bench = Benchmark(data_dir, tasks, None, only, None, 0)
    names = Y.dataset_names(bench.items)
    label = f"yolo11{size}-shared/{shots}shot"
    rd = out / f"{shots}shot" / f"s{subset}"
    parts, claims = rd / "parts", rd / "claims"
    for d in (rd, parts, claims, rd / "models"):
        d.mkdir(parents=True, exist_ok=True)

    def write_once(path: Path, text: str):
        if not path.exists():
            tmp = path.with_name(f".{path.name}.{os.getpid()}")
            tmp.write_text(text)
            tmp.replace(path)

    write_once(rd / "run.json", json.dumps({
        "method": {"name": label, "baseline": "yolo-shared", "size": size, "epochs": epochs, "shots": shots, "gen_seed": subset,
                   "labels": "classes merged by normalised name; names without letters kept per dataset; predictions limited to the item's dataset"},
        "benchmark": bench.args, "n_items": len(bench.items), "agbench_git": _git_sha(),
        "started": datetime.now().astimezone().isoformat(timespec="seconds")}, indent=2))
    write_once(rd / "items.jsonl", "".join(json.dumps(it.to_dict()) + "\n" for it in bench.items))
    write_once(rd / "crops.json", json.dumps(crop_table(bench.items), indent=2))
    write_once(rd / "ag_task_coverage.md", coverage_md(bench.items))
    claim = _claimer(claims)

    by_task: dict = {}
    for it in bench.items:
        by_task.setdefault(it.task, []).append(it)

    def finished() -> tuple[set, set]:
        done = {r["item_id"] for r in read_jsonl(rd / "predictions.jsonl")}
        for f in parts.glob("*.jsonl"):
            done |= {r["item_id"] for r in read_jsonl(f)}
        return done, {f.name[: -len(".skipped.json")] for f in parts.glob("*.skipped.json")}

    for t in tasks:
        its = by_task.get(t, [])
        done, skipped = finished()
        if not its or t in skipped or all(it.item_id in done for it in its) or not claim(t):
            continue
        try:
            done, skipped = finished()
            if t in skipped or all(it.item_id in done for it in its):
                continue
            t_names = {ds: n for ds, n in names.items() if any(i.dataset == ds for i in its)}
            pairs = {(ds, sid) for ds in t_names for sid in Y.subset_ids(train_root, t, ds, subset, shots)}
            tr = [it for it in Y.train_items(train_root, t, only, {sid for _, sid in pairs}) if (it.dataset, it.sample_id) in pairs]
            typer.echo(f"{t}: {len(its)} items, {len(t_names)} datasets, {len(tr)} training images")
            if not tr:
                (parts / f"{t}.skipped.json").write_text(json.dumps({"task": t, "reason": "no training images"}))
                continue
            work = rd / "_work" / f"{t}.{os.getpid()}"
            try:
                saved, saved_cls = rd / "models" / f"{t}.pt", rd / "models" / f"{t}.classes.json"
                train_rec = parts / f"{t}.train.json"  # the training's compute record, kept if scoring fails
                with sampler() as g:
                    t0 = time.time()
                    if saved.exists() and saved_cls.exists():  # trained before, scoring failed: score again, do not retrain
                        w, classes = saved, json.loads(saved_cls.read_text())
                        typer.echo(f"  reusing the trained model {saved}")
                    else:
                        data, classes = Y.prepare_shared(t, tr, t_names, train_root, work / "data")
                        typer.echo(f"  {len(classes)} merged classes")
                        w = Y.train(t, data, work / "model", size, epochs, weights_dir, seed=subset,
                                    batch=64 if t == "classification" else 16)
                        shutil.copy(w, saved.with_suffix(".pt.tmp")); saved.with_suffix(".pt.tmp").replace(saved)
                        saved_cls.write_text(json.dumps(classes, indent=0))
                        t1 = time.time()
                        train_rec.write_text(json.dumps([{"ended": datetime.now().astimezone().isoformat(timespec="seconds"), "phase": "train",
                                                          "model": label, "task": t, "train_images": len(tr), "classes": len(classes),
                                                          "wall_s": round(t1 - t0, 1), **(g.summary(t0, t1) if g else {})}]))
                    t1 = time.time()
                    m = Y.SharedYOLOMethod(w, classes, t_names, label, subset)
                    recs = [grade(it, m.predict(it), {"bbox": "xyxy_px"}) for it in its]
                    t2 = time.time()
                end_t = datetime.now().astimezone().isoformat(timespec="seconds")
                comp = (json.loads(train_rec.read_text()) if train_rec.exists() else []) + [
                    {"ended": end_t, "phase": "generate", "model": label, "task": t, "items": len(its), "wall_s": round(t2 - t1, 1),
                     **(g.summary(t1, t2) if g else {})}]
                tmp = parts / f".{t}.{os.getpid()}"
                tmp.write_text("".join(json.dumps(r) + "\n" for r in exact_oeq_verdicts(recs)))
                (parts / f"{t}.compute.json").write_text(json.dumps(comp))
                tmp.replace(parts / f"{t}.jsonl")
            except Exception as e:
                typer.echo(f"  failed: {type(e).__name__}: {e}")
                with open(rd / "failures.jsonl", "a") as f:
                    f.write(json.dumps({"task": t, "time": datetime.now().isoformat(timespec="seconds"), "error": f"{type(e).__name__}: {e}"[:500]}) + "\n")
                if sum(1 for r in read_jsonl(rd / "failures.jsonl") if r.get("task") == t) >= 2:
                    (parts / f"{t}.skipped.json").write_text(json.dumps({"task": t, "reason": f"failed twice: {type(e).__name__}"}))
            finally:
                shutil.rmtree(work, ignore_errors=True)
        finally:
            (claims / t).unlink(missing_ok=True)

    done, skipped = finished()
    left = [t for t in tasks if by_task.get(t) and t not in skipped and not all(it.item_id in done for it in by_task[t])]
    if left:
        typer.echo(f"tasks still in progress elsewhere or failed once: {', '.join(left)}; run again to continue")
        raise typer.Exit(3)
    with open(rd / ".merge.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recs = {}
        for f in sorted(parts.glob("*.jsonl")):
            recs.update({r["item_id"]: r for r in read_jsonl(f)})
        merged = [recs[it.item_id] for it in bench.items if it.item_id in recs]
        tmp = rd / f".predictions.{os.getpid()}"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in merged))
        tmp.replace(rd / "predictions.jsonl")
        comp = [c for f in sorted(parts.glob("*.compute.json")) for c in json.loads(f.read_text())]
        (rd / "compute.json").write_text(json.dumps(comp, indent=2))
        (rd / "skipped.json").write_text(json.dumps([json.loads(f.read_text()) for f in sorted(parts.glob("*.skipped.json"))], indent=2))
        finalize(rd, merged)
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
    csv: Path = typer.Option(None, "--csv", help="Write scores.csv (per task), ag_tasks.csv and datasets.csv to this dir."),
):
    """Leaderboard across runs."""
    from agbench.eval.report import leaderboard_md, load_runs, write_csv

    rs = load_runs(runs)
    md = leaderboard_md(rs, average)
    typer.echo(md)
    if out:
        out.write_text(md)
    if csv:
        typer.echo(f"csv: {write_csv(rs, csv)}")


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
