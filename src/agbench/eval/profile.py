"""Compute profiling: VRAM, latency, throughput per model and task, extrapolated to the full data dir."""

import asyncio
import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from agbench.eval.items import EvalItem, build_items, spread
from agbench.eval.judge import judge_records
from agbench.eval.methods import VLMMethod, model_spec
from agbench.eval.resources import models
from agbench.eval.runner import Benchmark, read_jsonl
from agbench.eval.servers import Server, delete_weights
from agbench.progress import log

GPU_SIZES = [24, 32, 48, 80, 96, 141, 192]


class GPUSampler:
    """Samples NVML every `interval` s in a thread. Use as a context manager; read .summary()."""

    def __init__(self, interval: float = 0.5):
        import pynvml

        self.nv, self.interval, self.samples = pynvml, interval, []
        pynvml.nvmlInit()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self._stop = threading.Event()

    def info(self) -> dict:
        return {"gpus": [{"name": self.nv.nvmlDeviceGetName(h), "memory_gb": round(self.nv.nvmlDeviceGetMemoryInfo(h).total / 1e9, 1)}
                         for h in self.handles]}

    def _read(self) -> dict:
        mem = sum(self.nv.nvmlDeviceGetMemoryInfo(h).used for h in self.handles) / 1e9
        util = float(np.mean([self.nv.nvmlDeviceGetUtilizationRates(h).gpu for h in self.handles]))
        try:
            power = sum(self.nv.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000
        except self.nv.NVMLError:
            power = 0.0
        return {"t": time.time(), "mem_gb": mem, "util": util, "power_w": power}

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self._read())
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline_gb = self._read()["mem_gb"]
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self, t0: float = 0, t1: float = float("inf")) -> dict:
        s = [x for x in self.samples if t0 <= x["t"] <= t1]
        if not s:
            return {}
        return {"peak_vram_gb": round(max(x["mem_gb"] for x in s) - self.baseline_gb, 2),
                "mean_util": round(float(np.mean([x["util"] for x in s])), 1),
                "mean_power_w": round(float(np.mean([x["power_w"] for x in s])), 1)}


def _key(it_or_rec) -> str:
    d = it_or_rec if isinstance(it_or_rec, dict) else it_or_rec.__dict__
    return f"{d['task']}/{d['format']}" if d["task"] == "classification" else d["task"]


def pick(items: list[EvalItem], per_task: int, seed: int = 0) -> list[EvalItem]:
    by = defaultdict(list)
    for it in items:
        by[_key(it)].append(it)
    return [x for k in sorted(by) for x in spread(by[k], per_task, seed)]


def _stats(recs: list[dict], wall: float | None) -> dict:
    lat = [r["latency_s"] for r in recs if not r.get("error")]
    return {"n": len(recs), "p50_s": round(float(np.percentile(lat, 50)), 2) if lat else None,
            "p95_s": round(float(np.percentile(lat, 95)), 2) if lat else None,
            "throughput_per_s": round(len(recs) / wall, 3) if wall else None,
            "prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in recs])), 1) if recs else 0,
            "completion_tokens": round(float(np.mean([r["completion_tokens"] for r in recs])), 1) if recs else 0,
            "truncation_rate": round(float(np.mean([r["truncated"] for r in recs])), 3) if recs else 0,
            "error_rate": round(float(np.mean([bool(r.get("error")) for r in recs])), 3) if recs else 0}


def profile_model(model_id: str, all_items: list[EvalItem], out: Path, per_task: int, per_task_seq: int,
                  concurrency: int | None, thinking: bool, effort: str, gpu: GPUSampler) -> dict:
    spec = model_spec(model_id)
    llama = spec["backend"] == "llamacpp"
    conc = concurrency or (spec.get("parallel", 2) if llama else spec["concurrency"])
    batch_items = pick(all_items, per_task)
    seq_items = pick(all_items, per_task_seq)
    res = {"model": model_id, "backend": "llamacpp" if llama else "vllm", "weights_gb": spec.get("weights_gb"),
           "est_vram_gb": spec.get("est_vram_gb"), "concurrency": conc}
    t_start = time.time()
    try:
        with Server(model_id, out / "server.log") as srv:
            res["load_s"] = round(srv.load_seconds, 1)
            method = VLMMethod(srv.endpoint, model_id, thinking=thinking, effort=effort)
            kf = spec.get("bbox_key_format")
            t0 = time.time()
            Benchmark(items=seq_items).run(method, out / "seq", concurrency=1, resume=False, key_formats=kf)
            t1 = time.time()
            method = VLMMethod(srv.endpoint, model_id, thinking=thinking, effort=effort)
            Benchmark(items=batch_items).run(method, out / "batch", concurrency=conc, resume=False, key_formats=kf)
            t2 = time.time()
    except Exception as e:
        log(f"{model_id} failed: {e}")
        res["error"] = str(e)
        return res
    res["gpu_load"] = gpu.summary(t_start, t0)
    res["gpu_seq"], res["gpu_batch"] = gpu.summary(t0, t1), gpu.summary(t1, t2)
    res["peak_vram_gb"] = max(v.get("peak_vram_gb", 0) for v in (res["gpu_load"], res["gpu_seq"], res["gpu_batch"]))
    seq, batch = read_jsonl(out / "seq" / "predictions.jsonl"), read_jsonl(out / "batch" / "predictions.jsonl")
    batch_wall = t2 - t1
    res["tasks"] = {}
    for k in sorted({_key(r) for r in batch}):
        s, b = [r for r in seq if _key(r) == k], [r for r in batch if _key(r) == k]
        share = sum(r["latency_s"] for r in b) / max(sum(r["latency_s"] for r in batch), 1e-9)
        res["tasks"][k] = {"seq": _stats(s, None), "batch": _stats(b, batch_wall * share)}
    return res


def extrapolate(res: dict, counts: Counter) -> dict:
    """Hours for the full data dir: sequential (mean latency at c=1) and batched (measured throughput)."""
    out = {}
    for k, v in (res.get("tasks") or {}).items():
        n = counts.get(k, 0)
        seq_mean = v["seq"]["p50_s"] or 0
        tp = v["batch"]["throughput_per_s"] or 0
        out[k] = {"items": n, "hours_seq": round(n * seq_mean / 3600, 2), "hours_batch": round(n / tp / 3600, 2) if tp else None}
    return out


def recommend(peak_gb: float) -> int | None:
    return next((g for g in GPU_SIZES if peak_gb <= g * 0.95), None)


def run_profile(model_ids: list[str], data_dir, out_dir, per_task: int = 20, per_task_seq: int = 5,
                concurrency: int | None = None, thinking: bool = True, effort: str = "medium", judge: bool = True,
                cleanup_weights: bool = False, redo: bool = False) -> dict:
    """Profiles each model in turn. Models already profiled (no error) in out_dir/profile.json are kept unless redo."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_items, _ = build_items(Path(data_dir))
    counts = Counter(_key(it) for it in all_items)
    settings = {"per_task": per_task, "per_task_seq": per_task_seq, "thinking": thinking, "effort": effort}
    prev = json.loads((out / "profile.json").read_text()) if (out / "profile.json").exists() and not redo else {}
    kept = {r["model"]: r for r in prev.get("models", []) if not r.get("error")} if prev.get("settings", settings) == settings else {}
    results = [r for r in prev.get("models", []) if r["model"] in kept and r["model"] not in model_ids]
    with GPUSampler() as gpu:
        hw = gpu.info()
        for m in model_ids:
            if m in kept:
                log(f"{m}: already profiled, keeping (use --redo to re-run)")
                r = kept[m]
            else:
                r = profile_model(m, all_items, out / m.replace("/", "__"), per_task, per_task_seq, concurrency, thinking, effort, gpu)
                if cleanup_weights:
                    delete_weights(m)
            r["full_run"] = extrapolate(r, counts)
            results.append(r)
            (out / "profile.json").write_text(json.dumps({"hardware": hw, "counts": counts, "models": results,
                                                          "judge": prev.get("judge"), "settings": settings}, indent=2))
        n_open = sum(1 for it in all_items if it.format == "oeq" or it.qa_format == "open_ended")
        judge_res = prev.get("judge") if (prev.get("judge") and not prev["judge"].get("error") and not redo) else None
        if judge and judge_res is None:
            judge_res = profile_judge(out, gpu, n_open)
    data = {"hardware": hw, "counts": dict(counts), "models": results, "judge": judge_res, "settings": settings}
    (out / "profile.json").write_text(json.dumps(data, indent=2))
    write_report(data, out)
    return data


def profile_judge(out: Path, gpu: GPUSampler, open_items: int) -> dict:
    """Judge the open-ended answers collected during profiling; report judge throughput."""
    judge_model = models()["judge"]
    recs = [r for p in out.glob("*/batch/predictions.jsonl") for r in read_jsonl(p) if r.get("needs_judge")]
    if not recs:
        return {}
    for r in recs:
        r["judge_correct"] = None
    try:
        with Server(judge_model, out / "judge_server.log") as srv:
            t0 = time.time()
            asyncio.run(judge_records(recs, srv.endpoint, judge_model))
            t1 = time.time()
    except Exception as e:
        return {"model": judge_model, "error": str(e)}
    tp = len(recs) / (t1 - t0)
    return {"model": judge_model, "n": len(recs), "throughput_per_s": round(tp, 2), **gpu.summary(t0, t1),
            "open_items_per_model": open_items, "hours_per_model": round(open_items / tp / 3600, 2)}


def profile_frames(data: dict) -> dict:
    """Tables shared by the report and the viewer."""
    fit = pd.DataFrame([{
        "model": r["model"], "backend": r["backend"], "weights GB": r.get("weights_gb"), "est. VRAM GB": r.get("est_vram_gb"),
        "peak VRAM GB": r.get("peak_vram_gb"), "load s": r.get("load_s"), "min GPU GB": recommend(r.get("peak_vram_gb") or r.get("est_vram_gb") or 0),
        "util %": (r.get("gpu_batch") or {}).get("mean_util"), "power W": (r.get("gpu_batch") or {}).get("mean_power_w"),
        "error": r.get("error", "")} for r in data["models"]])
    rows = []
    for r in data["models"]:
        for k, v in (r.get("tasks") or {}).items():
            f = r["full_run"].get(k, {})
            rows.append({"model": r["model"], "task": k, "s/image (c=1, p50)": v["seq"]["p50_s"], "p95": v["seq"]["p95_s"],
                         "images/s (batched)": v["batch"]["throughput_per_s"], "completion tokens": v["batch"]["completion_tokens"],
                         "truncated %": round(100 * v["batch"]["truncation_rate"], 1), "full-run items": f.get("items"),
                         "full-run h (batched)": f.get("hours_batch"), "full-run h (c=1)": f.get("hours_seq")})
    per_task = pd.DataFrame(rows)
    totals = per_task.groupby("model")[["full-run h (batched)", "full-run h (c=1)"]].sum().reset_index() if rows else pd.DataFrame()
    judge = data.get("judge") or {}
    peak = max([r.get("peak_vram_gb") or r.get("est_vram_gb") or 0 for r in data["models"]] or [0])
    total_h = float(totals["full-run h (batched)"].sum()) if len(totals) else 0
    total_h += (judge.get("hours_per_model") or 0) * len(data["models"])
    hw = ", ".join(f"{g['name']} ({g['memory_gb']} GB)" for g in data["hardware"]["gpus"])
    return {"fit": fit, "per_task": per_task, "totals": totals, "judge": judge, "peak": peak, "total_h": total_h, "hw": hw}


def write_report(data: dict, out: Path) -> None:
    f = profile_frames(data)
    fit, per_task, totals, judge, peak, total_h, hw = (f[k] for k in ("fit", "per_task", "totals", "judge", "peak", "total_h", "hw"))
    rows = len(per_task)
    s = data["settings"]
    md = [f"# Compute profile\n", f"Hardware: {hw}  ", f"Thinking: {s['thinking']} (effort {s['effort']}); "
          f"{s['per_task']} items per task batched, {s['per_task_seq']} sequential.\n",
          "## Recommendation\n", f"- Minimum single GPU: {recommend(peak)} GB (largest peak {peak:.1f} GB)",
          f"- Full suite, all models + judge, batched on one GPU: {total_h:.1f} GPU-hours "
          f"({total_h / 24:.1f} days; {total_h / 4:.1f} h on 4 GPUs)\n",
          "## Memory\n", fit.to_markdown(index=False), "\n## Time per image and full-run estimate\n",
          per_task.to_markdown(index=False) if rows else "_no results_", "\n## Totals per model (hours)\n",
          totals.to_markdown(index=False) if len(totals) else "_no results_"]
    if judge:
        md += ["\n## Judge\n", pd.DataFrame([judge]).to_markdown(index=False)]
    (out / "report.md").write_text("\n".join(md) + "\n")
    html = ["<html><head><meta charset='utf-8'><title>Compute profile</title><style>body{font-family:sans-serif;margin:24px}"
            "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:4px 8px}</style></head><body>",
            f"<h1>Compute profile</h1><p>Hardware: {hw}</p><h2>Recommendation</h2><p>Minimum single GPU: {recommend(peak)} GB. "
            f"Full suite batched: {total_h:.1f} GPU-hours.</p><h2>Memory</h2>", fit.to_html(index=False),
            "<h2>Time per image</h2>", per_task.to_html(index=False), "<h2>Totals</h2>", totals.to_html(index=False)]
    if judge:
        html += ["<h2>Judge</h2>", pd.DataFrame([judge]).to_html(index=False)]
    (out / "report.html").write_text("".join(html) + "</body></html>")
    log(f"report: {out / 'report.md'}")
