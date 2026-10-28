"""Compute profiling: VRAM, latency, throughput per model and task, extrapolated to the full data dir."""

import asyncio
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from agbench.eval.gpu import GPUSampler
from agbench.eval.items import EvalItem, build_items, spread
from agbench.eval.judge import judge_records
from agbench.eval.methods import VLMMethod, key_formats, model_spec
from agbench.eval.resources import models
from agbench.eval.runner import Benchmark, finalize, read_jsonl, write_jsonl
from agbench.eval.servers import Server, delete_weights
from agbench.progress import log

GPU_SIZES = [24, 32, 48, 80, 96, 141, 192]


def _key(it_or_rec) -> str:
    d = it_or_rec if isinstance(it_or_rec, dict) else it_or_rec.__dict__
    return f"{d['task']}/{d['format']}" if d["task"] == "classification" else d["task"]


def pick(items: list[EvalItem], per_task: int, seed: int = 0) -> list[EvalItem]:
    """`per_task` items per task, shuffled so slow tasks don't cluster at the end of the batch."""
    by = defaultdict(list)
    for it in items:
        by[_key(it)].append(it)
    out = [x for k in sorted(by) for x in spread(by[k], per_task, seed)]
    random.Random(seed).shuffle(out)
    return out


def _stats(recs: list[dict], wall: float | None) -> dict:
    lat = [r["latency_s"] for r in recs if not r.get("error")]
    return {"n": len(recs), "mean_s": round(float(np.mean(lat)), 2) if lat else None, "p50_s": round(float(np.percentile(lat, 50)), 2) if lat else None,
            "p95_s": round(float(np.percentile(lat, 95)), 2) if lat else None,
            "throughput_per_s": round(len(recs) / wall, 3) if wall else None,
            "prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in recs])), 1) if recs else 0,
            "completion_tokens": round(float(np.mean([r["completion_tokens"] for r in recs])), 1) if recs else 0,
            "truncation_rate": round(float(np.mean([r["truncated"] for r in recs])), 3) if recs else 0,
            "error_rate": round(float(np.mean([bool(r.get("error")) for r in recs])), 3) if recs else 0}


def steady_wall(recs: list[dict], conc: int, wall: float) -> float:
    """Time the batch would take with every slot busy: throughput is counted between the first refill and the last
    request start (each start in that window replaced a finished request), so the ramp-down tail is left out."""
    starts = sorted(r["t_end"] - r["latency_s"] for r in recs if "t_end" in r)
    if len(starts) < conc + 3 or starts[-1] <= starts[conc]:
        return wall
    return len(recs) * (starts[-1] - starts[conc]) / (len(starts) - conc - 1)


def variant_settings(spec: dict, variant: str) -> tuple[bool, str | None, dict]:
    """(thinking, effort, extra request body) for an effort variant: off | low | medium | high (levels from models.yaml)."""
    return (False, None, {}) if variant == "off" else (True, variant, {})


def profile_model(model_id: str, all_items: list[EvalItem], out: Path, per_task: int, per_task_seq: int,
                  concurrency: int | None, thinking: bool, effort: str, gpu: GPUSampler,
                  variants: list[str] | None = None) -> list[dict]:
    """One result per variant (or one with the given thinking/effort); the model is loaded once."""
    spec = model_spec(model_id)
    llama = spec["backend"] == "llamacpp"
    conc = concurrency or (spec.get("parallel", 2) if llama else spec["concurrency"])
    batch_items = pick(all_items, per_task)
    seq_items = pick(all_items, per_task_seq)
    base = {"model": model_id, "backend": "llamacpp" if llama else "vllm", "weights_gb": spec.get("weights_gb"),
            "est_vram_gb": spec.get("est_vram_gb"), "concurrency": conc}
    plan = [(v, *variant_settings(spec, v)) for v in variants] if variants else [(None, thinking, effort, {})]
    name = lambda v: f"{model_id} · {v}" if v else model_id
    done = []
    t_start = time.time()
    try:
        with Server(model_id, out / "server.log") as srv:
            base["load_s"] = round(srv.load_seconds, 1)
            t_load, kf = time.time(), key_formats(spec)
            for v, th, ef, body in plan:
                d = out / v if v else out
                t0 = time.time()
                for items, sub, c in ((seq_items, "seq", 1), (batch_items, "batch", conc)):
                    if items:
                        method = VLMMethod(srv.endpoint, model_id, thinking=th, effort=ef, body=body)
                        method.name = name(v)
                        Benchmark(items=items).run(method, d / sub, concurrency=c, resume=False, key_formats=kf, gpu=gpu)
                    if sub == "seq":
                        t1 = time.time()
                done.append((v, d, t0, t1, time.time(), {"thinking": th, "effort": ef, **body}))
    except Exception as e:
        log(f"{model_id} failed: {e}")
        return [{**base, "model": name(v), "variant": v, "error": str(e)} for v, *_ in plan]
    results = []
    for v, d, t0, t1, t2, request in done:
        res = {**base, "model": name(v), "variant": v, "request": request}
        res["gpu_load"] = gpu.summary(t_start, t_load)
        res["gpu_seq"], res["gpu_batch"] = gpu.summary(t0, t1), gpu.summary(t1, t2)
        res["peak_vram_gb"] = max(x.get("peak_vram_gb", 0) for x in (res["gpu_load"], res["gpu_seq"], res["gpu_batch"]))
        seq, batch = read_jsonl(d / "seq" / "predictions.jsonl"), read_jsonl(d / "batch" / "predictions.jsonl")
        res["batch_wall_s"] = round(t2 - t1, 1)
        batch_wall = res["steady_wall_s"] = round(steady_wall(batch, conc, t2 - t1), 1)
        res["tasks"] = {}
        for k in sorted({_key(r) for r in batch}):
            sq, b = [r for r in seq if _key(r) == k], [r for r in batch if _key(r) == k]
            share = sum(r["latency_s"] for r in b) / max(sum(r["latency_s"] for r in batch), 1e-9)
            res["tasks"][k] = {"seq": _stats(sq, None), "batch": _stats(b, batch_wall * share)}
        results.append(res)
    return results


def extrapolate(res: dict, counts: Counter) -> dict:
    """Hours for the full data dir: batched (steady-state throughput at the run concurrency) and, for reference, c=1 (mean latency)."""
    out = {}
    for k, v in (res.get("tasks") or {}).items():
        n = counts.get(k, 0)
        seq_mean = v["seq"].get("mean_s") or v["seq"]["p50_s"] or 0
        tp = v["batch"]["throughput_per_s"] or 0
        out[k] = {"items": n, "hours_seq": round(n * seq_mean / 3600, 2), "hours_batch": round(n / tp / 3600, 2) if tp else None}
    return out


def recommend(peak_gb: float) -> int | None:
    return next((g for g in GPU_SIZES if peak_gb <= g * 0.95), None)


def need_gb(r: dict) -> float:
    """VRAM a model needs. vLLM pre-allocates gpu_memory_utilization of the GPU, so its peak is the
    reservation; use the weights + KV estimate instead. llama.cpp allocates what it uses: use the peak."""
    if r.get("backend") == "vllm":
        return r.get("est_vram_gb") or r.get("peak_vram_gb") or 0
    return r.get("peak_vram_gb") or r.get("est_vram_gb") or 0


def min_gpu(r: dict, hw_gb: float) -> int | None:
    """Smallest GPU size that fits; a model that ran on this GPU never needs a bigger one."""
    g = recommend(need_gb(r))
    ran_on = max((s for s in GPU_SIZES if s <= hw_gb + 1), default=None)
    return ran_on if (not r.get("error") and ran_on and (g is None or g > ran_on)) else g


def run_profile(model_ids: list[str], data_dir, out_dir, per_task: int = 20, per_task_seq: int = 5,
                concurrency: int | None = None, thinking: bool = True, effort: str = "low", judge: bool = True,
                cleanup_weights: bool = False, redo: bool = False, variants: list[str] | None = None) -> dict:
    """Profiles each model in turn (each `variants` entry as its own row). Rows already profiled (no error) in
    out_dir/profile.json are kept unless redo."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_items, _ = build_items(Path(data_dir))
    counts = Counter(_key(it) for it in all_items)
    settings = {"per_task": per_task, "per_task_seq": per_task_seq, "thinking": thinking, "effort": effort}
    if variants:
        settings["variants"] = variants
    names = {m: [f"{m} · {v}" for v in variants] if variants else [m] for m in model_ids}
    prev = json.loads((out / "profile.json").read_text()) if (out / "profile.json").exists() and not redo else {}
    kept = {r["model"]: r for r in prev.get("models", []) if not r.get("error")} if prev.get("settings", settings) == settings else {}
    results = [r for r in prev.get("models", []) if r["model"] in kept and not any(r["model"] in n for n in names.values())]
    with GPUSampler() as gpu:
        hw = gpu.info()
        for m in model_ids:
            if all(n in kept for n in names[m]):
                log(f"{m}: already profiled, keeping (use --redo to re-run)")
                rs = [kept[n] for n in names[m]]
            else:
                rs = profile_model(m, all_items, out / m.replace("/", "__"), per_task, per_task_seq, concurrency, thinking,
                                   effort, gpu, variants)
                if cleanup_weights:
                    delete_weights(m)
            for r in rs:
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
    """Judge the open-ended answers collected during profiling (verdicts saved with each model's batch); report judge throughput."""
    judge_model = models()["judge"]
    files = {p: read_jsonl(p) for p in out.glob("**/batch/predictions.jsonl")}
    recs = [r for rs in files.values() for r in rs if r.get("needs_judge")]
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
    for p, rs in files.items():
        write_jsonl(p, rs)
        finalize(p.parent, rs)
    tp = len(recs) / (t1 - t0)
    return {"model": judge_model, "n": len(recs), "throughput_per_s": round(tp, 2), **gpu.summary(t0, t1),
            "open_items_per_model": open_items, "hours_per_model": round(open_items / tp / 3600, 2)}


def _r(x, nd=2):
    return round(x, nd) if x is not None else None


def actual_frames(runs: list[dict], data: dict | None) -> dict:
    """Measured compute of real runs (their compute.json) next to the profile's estimate for the same items.
    Estimates use the profile's batched throughput; actual per-task hours split the generation wall time by latency share."""
    prof = {r["model"]: r for r in (data or {}).get("models", [])}
    judge = (data or {}).get("judge") or {}
    prof_effort = ((data or {}).get("settings") or {}).get("effort")
    model_rows, task_rows = [], []
    for run in runs:
        d = Path(run["dir"])
        if not (d / "compute.json").exists():
            continue
        sessions = json.loads((d / "compute.json").read_text())
        phase = lambda name: [x for x in sessions if x["phase"] == name]
        gen, jud, load = phase("generate"), phase("judge"), phase("load")
        last_redo = max((i for i, x in enumerate(jud) if x.get("redo")), default=0)
        jud = jud[last_redo:]  # judging since the last full re-judge
        model = run["run"]["method"].get("model_id") or run["name"]
        label, effort = run["name"], run["run"]["method"].get("effort")
        p = prof.get(model, {})
        n, lat = Counter(), defaultdict(float)
        for r in read_jsonl(d / "predictions.jsonl"):
            n[_key(r)] += 1
            lat[_key(r)] += r.get("latency_s", 0)
        gen_h = sum(x["wall_s"] for x in gen) / 3600
        est_gen, act_cov = [], 0.0
        for k in sorted(n):
            tp = (((p.get("tasks") or {}).get(k) or {}).get("batch") or {}).get("throughput_per_s")
            est = n[k] / tp / 3600 if tp else None
            act = gen_h * lat[k] / max(sum(lat.values()), 1e-9)
            est_gen.append(est)
            act_cov += act if est else 0
            task_rows.append({"run": label, "effort": effort, "task": k, "items": n[k], "est. h": _r(est), "actual h": _r(act),
                              "actual / est.": _r(act / est) if est else None})
        est_gen_h = sum(e for e in est_gen if e is not None) if any(e is not None for e in est_gen) else None
        n_judged, judge_h = sum(x["items"] for x in jud), sum(x["wall_s"] for x in jud) / 3600
        est_judge_h = n_judged / judge["throughput_per_s"] / 3600 if n_judged and judge.get("throughput_per_s") else None
        wall = sum(x["wall_s"] for x in gen) or 1
        model_rows.append({
            "run": label, "effort": effort, "profile effort": prof_effort if p else None, "items": sum(n.values()), "concurrency": gen[-1]["concurrency"] if gen else None,
            "load s (est.)": p.get("load_s"), "load s": load[-1]["wall_s"] if load else None,
            "generate h (est.)": _r(est_gen_h), "generate h": _r(gen_h),
            "judged": n_judged, "judge h (est.)": _r(est_judge_h), "judge h": _r(judge_h) if jud else None,
            "actual / est.": _r((act_cov + (judge_h if est_judge_h else 0)) / (est_gen_h + (est_judge_h or 0))) if est_gen_h else None,
            "peak VRAM GB (profile)": p.get("peak_vram_gb"),
            "peak VRAM GB": max((x.get("peak_vram_gb") or 0 for x in gen + load), default=None) or None,
            "util %": _r(sum(x.get("mean_util", 0) * x["wall_s"] for x in gen) / wall, 1),
            "power W": _r(sum(x.get("mean_power_w", 0) * x["wall_s"] for x in gen) / wall, 1)})
    return {"models": pd.DataFrame(model_rows), "tasks": pd.DataFrame(task_rows)}


def profile_frames(data: dict) -> dict:
    """Tables shared by the report and the viewer."""
    hw_gb = min((g["memory_gb"] for g in data["hardware"]["gpus"]), default=0)
    fit = pd.DataFrame([{
        "model": r["model"], "backend": r["backend"], "weights GB": r.get("weights_gb"), "est. VRAM GB": r.get("est_vram_gb"),
        "peak VRAM GB": r.get("peak_vram_gb"), "load s": r.get("load_s"), "needed GB": need_gb(r), "min GPU GB": min_gpu(r, hw_gb),
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
    peak = max([need_gb(r) for r in data["models"]] or [0])
    gpu = max([min_gpu(r, hw_gb) or 0 for r in data["models"]] or [0]) or None
    total_h = float(totals["full-run h (batched)"].sum()) if len(totals) else 0
    total_h += (judge.get("hours_per_model") or 0) * len(data["models"])
    hw = ", ".join(f"{g['name']} ({g['memory_gb']} GB)" for g in data["hardware"]["gpus"])
    return {"fit": fit, "per_task": per_task, "totals": totals, "judge": judge, "peak": peak, "gpu": gpu, "total_h": total_h, "hw": hw}


def write_report(data: dict, out: Path) -> None:
    f = profile_frames(data)
    fit, per_task, totals, judge, peak, gpu, total_h, hw = (f[k] for k in ("fit", "per_task", "totals", "judge", "peak", "gpu", "total_h", "hw"))
    rows = len(per_task)
    s = data["settings"]
    md = [f"# Compute profile\n", f"Hardware: {hw}  ", f"Thinking: {s['thinking']} (effort {s['effort']}); "
          f"{s['per_task']} items per task batched, {s['per_task_seq']} sequential.\n",
          "## Recommendation\n", f"- Minimum single GPU: {gpu} GB (largest need {peak:.1f} GB)",
          f"- Full suite, all models + judge, on one GPU: {total_h:.1f} GPU-hours "
          f"({total_h / 24:.1f} days; {total_h / 4:.1f} h on 4 GPUs), from steady-state batched throughput at each model's concurrency",
          "- vLLM reserves gpu_memory_utilization of the GPU up front, so its peak VRAM is the reservation; "
          "`needed GB` is the weights + KV estimate for vLLM and the measured peak for llama.cpp\n",
          "## Memory\n", fit.to_markdown(index=False), "\n## Time per image and full-run estimate\n",
          per_task.to_markdown(index=False) if rows else "_no results_", "\n## Totals per model (hours)\n",
          totals.to_markdown(index=False) if len(totals) else "_no results_"]
    if judge:
        md += ["\n## Judge\n", pd.DataFrame([judge]).to_markdown(index=False)]
    (out / "report.md").write_text("\n".join(md) + "\n")
    html = ["<html><head><meta charset='utf-8'><title>Compute profile</title><style>body{font-family:sans-serif;margin:24px}"
            "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:4px 8px}</style></head><body>",
            f"<h1>Compute profile</h1><p>Hardware: {hw}</p><h2>Recommendation</h2><p>Minimum single GPU: {gpu} GB. "
            f"Full suite: {total_h:.1f} GPU-hours.</p><h2>Memory</h2>", fit.to_html(index=False),
            "<h2>Time per image</h2>", per_task.to_html(index=False), "<h2>Totals</h2>", totals.to_html(index=False)]
    if judge:
        html += ["<h2>Judge</h2>", pd.DataFrame([judge]).to_html(index=False)]
    (out / "report.html").write_text("".join(html) + "</body></html>")
    log(f"report: {out / 'report.md'}")
