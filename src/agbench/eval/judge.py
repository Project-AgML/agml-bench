"""LLM judge for open-ended answers (classification OEQ and open-ended ITT)."""

import asyncio
import json
import re
import time
from pathlib import Path

from agbench.eval.client import Client, api_key_from_env
from agbench.eval.methods import model_spec
from agbench.eval.resources import models, prompts
from agbench.eval.runner import finalize, log_compute, read_jsonl, write_jsonl
from agbench.progress import Progress, log


def judge_prompt(r: dict) -> str:
    p = prompts()
    if r["task"] == "classification":
        return p["judge_classification"].format(class_list=", ".join(r.get("class_names") or []), gt=r["gt"], pred=r["answer"])
    return p["judge_qa"].format(question=r.get("question") or "", gt=r["gt"], pred=r["answer"])


def parse_verdict(text: str) -> bool | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            v = json.loads(m.group())["is_match"]
            return v if isinstance(v, bool) else str(v).lower() == "true"
        except (json.JSONDecodeError, KeyError):
            pass
    m = re.search(r'"?is_match"?\s*:\s*(true|false)', text or "", re.I)
    return m.group(1).lower() == "true" if m else None


async def judge_records(records: list[dict], endpoint: str, judge_model: str, concurrency: int = 32,
                        api_key: str = "EMPTY") -> int:
    spec = model_spec(judge_model)
    body = {**(spec.get("no_thinking") or {}), "temperature": 0.0}
    client = Client(endpoint, api_key=api_key, prefer=judge_model)
    todo = [r for r in records if r.get("needs_judge") and r.get("judge_correct") is None]
    sem, prog = asyncio.Semaphore(concurrency), Progress("judge", len(todo), every=30)

    async def one(r):
        async with sem:
            res = await client.chat(judge_prompt(r), [], system=prompts()["judge_system"], max_tokens=512, body=body)
            r["judge_correct"], r["judge_text"], r["judge_model"] = parse_verdict(res.text), res.text, judge_model
            prog.add()

    await asyncio.gather(*(one(r) for r in todo))
    return len(todo)


def judge_run(run_dir, endpoint: str, judge_model: str | None = None, concurrency: int = 32, gpu=None, redo: bool = False,
              api_key_env: str | None = None) -> dict:
    """Judge open-ended answers that have no verdict yet (`redo`: all of them, e.g. after changing the judge prompt).

    Any judge model behind an OpenAI-compatible endpoint works (`api_key_env`: variable holding a hosted API's key).
    Scores are comparable with the reference results only with the benchmark's reference judge and judge prompt
    (run_manifest.json: comparable)."""
    from agbench.bench import judge_prompt_sha

    run_dir = Path(run_dir)
    judge_model = judge_model or models()["judge"]
    recs = read_jsonl(run_dir / "predictions.jsonl")
    if redo:
        for r in recs:
            if r.get("needs_judge"):
                r["judge_correct"] = r["judge_text"] = None
    t0 = time.time()
    judged = {r.get("judge_model") for r in recs if r.get("judge_correct") is not None}
    judged.discard(None)
    if not redo and judged - {judge_model}:
        raise ValueError(f"{run_dir} already has verdicts from {', '.join(map(str, judged - {judge_model}))}; "
                         f"use --redo to re-judge every answer with {judge_model}")
    n = asyncio.run(judge_records(recs, endpoint, judge_model, concurrency, api_key_from_env(api_key_env)))
    t1 = time.time()
    if n:
        log_compute(run_dir, phase="judge", model=judge_model, items=n, concurrency=concurrency, wall_s=round(t1 - t0, 1), redo=redo,
                    **(gpu.summary(t0, t1) if gpu else {}))
    write_jsonl(run_dir / "predictions.jsonl", recs)
    run = json.loads((run_dir / "run.json").read_text())
    run["judge"] = {"model": judge_model, "prompt_sha": judge_prompt_sha(),
                    "self_judged": run["method"].get("model_id") == judge_model}
    (run_dir / "run.json").write_text(json.dumps(run, indent=2))
    log(f"judged {n} answers in {run_dir}")
    return finalize(run_dir, recs).metrics
