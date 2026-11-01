"""Run a Method over benchmark items: predictions.jsonl (resumable), metrics.json, crop and ag-task tables."""

import asyncio
import inspect
import json
import subprocess
import time
from contextlib import nullcontext
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from agbench.eval import metrics as mx
from agbench.eval.client import Prediction
from agbench.eval.gpu import sampler
from agbench.eval.grade import grade
from agbench.eval.items import MCQAOptions, build_items
from agbench.eval.methods import Method
from agbench.progress import Progress, fmt_duration, log


@dataclass
class Result:
    out_dir: Path
    metrics: dict
    records: list[dict]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def log_compute(out_dir, **session) -> None:
    """Append one timed session (load, generate, judge) to out_dir/compute.json."""
    p = Path(out_dir) / "compute.json"
    sessions = json.loads(p.read_text()) if p.exists() else []
    sessions.append({"ended": datetime.now(timezone.utc).isoformat(timespec="seconds"), **session})
    p.write_text(json.dumps(sessions, indent=2))


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:
        return None


def crop_table(items) -> list[dict]:
    rows = defaultdict(lambda: {"crops": Counter(), "sources": Counter()})
    for it in items:
        r = rows[(it.task, it.dataset)]
        r["crops"][it.crop] += 1
        r["sources"][it.crop_source] += 1
    return [{"task": t, "dataset": d, "crops": dict(v["crops"].most_common()), "sources": dict(v["sources"])}
            for (t, d), v in sorted(rows.items())]


def coverage_md(items) -> str:
    c = Counter((it.ag_task, it.task) for it in items)
    tasks = sorted({t for _, t in c})
    lines = ["| ag task | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for a in sorted({a for a, _ in c}):
        lines.append(f"| {a} | " + " | ".join(str(c.get((a, t), 0)) for t in tasks) + " |")
    return "\n".join(lines) + "\n"


class Benchmark:
    """The standard benchmark over a base-data directory.

    >>> Benchmark(data_dir, tasks=["classification"], formats=["mcqa_within"]).run(MyMethod(), out_dir="runs/x")
    """

    def __init__(self, data_dir=None, tasks=None, formats=None, only: str | None = None, limit_per_dataset: int | None = None,
                 seed: int = 0, items=None, mcqa: MCQAOptions | None = None):
        self.data_dir, self.seed = Path(data_dir) if data_dir else None, seed
        self.args = {"data_dir": str(self.data_dir), "tasks": tasks, "formats": formats, "only": only,
                     "limit_per_dataset": limit_per_dataset, "seed": seed, "mcqa": asdict(mcqa or MCQAOptions())}
        if items is None:
            items, self.skipped = build_items(self.data_dir, tasks, formats, only, limit_per_dataset, seed, mcqa)
        else:
            self.skipped = []
        self.items = items

    def remaining(self, out_dir) -> list:
        """Items without a successful prediction in out_dir."""
        done = {r["item_id"] for r in read_jsonl(Path(out_dir) / "predictions.jsonl") if not r.get("error")}
        return [it for it in self.items if it.item_id not in done]

    def run(self, method: Method, out_dir, concurrency: int = 8, resume: bool = True, key_formats: dict | None = None,
            gpu=None) -> Result:
        return asyncio.run(self.run_async(method, out_dir, concurrency, resume, key_formats, gpu))

    async def run_async(self, method: Method, out_dir, concurrency: int = 8, resume: bool = True,
                        key_formats: dict | None = None, gpu=None) -> Result:
        """`gpu`: a running GPUSampler (e.g. started before the model server); otherwise one is started here."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pred_path = out / "predictions.jsonl"
        done = {r["item_id"]: r for r in read_jsonl(pred_path)} if resume else {}
        done = {k: r for k, r in done.items() if not r.get("error")}  # retry failed requests
        if not resume:
            pred_path.unlink(missing_ok=True)
            (out / "compute.json").unlink(missing_ok=True)
        if done:
            write_jsonl(pred_path, list(done.values()))
        todo = [it for it in self.items if it.item_id not in done]
        (out / "run.json").write_text(json.dumps({
            "method": method.config(), "benchmark": self.args, "n_items": len(self.items), "agbench_git": _git_sha(),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "concurrency": concurrency}, indent=2))
        write_jsonl(out / "items.jsonl", [it.to_dict() for it in self.items])
        (out / "skipped.json").write_text(json.dumps(self.skipped, indent=2))
        (out / "crops.json").write_text(json.dumps(crop_table(self.items), indent=2))
        (out / "ag_task_coverage.md").write_text(coverage_md(self.items))
        log(f"{method.name}: {len(todo)} to run, {len(done)} already done")

        await method.setup()
        sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
        prog, t0 = Progress(method.name, len(todo), every=30), time.time()
        sync = not inspect.iscoroutinefunction(method.predict)

        async def one(it):
            async with sem:
                try:
                    pred = await (asyncio.to_thread(method.predict, it) if sync else method.predict(it))
                except Exception as e:
                    pred = Prediction(text="", finish_reason="error", error=f"{type(e).__name__}: {e}")
                try:
                    rec = grade(it, pred, key_formats)
                except Exception as e:  # one ungradable answer must not end the run; it is retried on resume
                    log(f"{it.item_id}: grading failed: {type(e).__name__}: {e}")
                    rec = grade(it, Prediction(text="", finish_reason="error", error=f"grading: {type(e).__name__}: {e}"), key_formats)
                rec["t_end"] = round(time.time(), 2)
            async with lock:
                with open(pred_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done[it.item_id] = rec
                prog.add()

        n_before = len(done)
        with nullcontext(gpu) if gpu else sampler() as g:
            try:
                await asyncio.gather(*(one(it) for it in todo))
            finally:  # log the session even if it dies, so its GPU time still counts
                await method.close()
                t1 = time.time()
                if todo:
                    log_compute(out, phase="generate", model=method.name, items=len(done) - n_before, concurrency=concurrency,
                                wall_s=round(t1 - t0, 1), **({"partial": True} if len(done) - n_before < len(todo) else {}),
                                **(g.summary(t0, t1) if g else {}))
        log(f"{method.name}: finished {len(todo)} items in {fmt_duration(t1 - t0)}")
        return finalize(out, [done[it.item_id] for it in self.items if it.item_id in done])


def finalize(out: Path, records: list[dict]) -> Result:
    m = mx.compute(records)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    return Result(out, m, records)
