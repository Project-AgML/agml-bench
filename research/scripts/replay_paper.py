"""Replay the paper's saved model answers through a frozen benchmark and check that the scores are unchanged.

For each paper run, a Replay method returns the answer the model gave for each item (no model is called). The new
code path then loads items from the bundle, grades every answer and computes the metrics. Judge verdicts for
open-ended answers are carried over (re-judging would need the judge model). Any difference between the replayed
and the original scores comes from the benchmark code, since the answers are fixed.

usage: uv run python research/scripts/replay_paper.py --data-dir $DATA [--runs runs/rejudge_v4/exp/seeds] [--bench agml-full-v1]
"""
import argparse
import json
import math
import tempfile
from pathlib import Path

from agbench.eval.client import Prediction, encode_image
from agbench.eval.methods import Method, key_formats, model_spec
from agbench.eval.runner import Benchmark, finalize, read_jsonl, write_jsonl

SCORE_KEYS = ("correct", "acc", "parse_ok", "pred_label", "pred_class", "idk", "tp", "fp", "fn", "mean_iou", "rouge_l",
              "token_f1", "needs_judge", "truncated")
PRED_KEYS = ("text", "reasoning", "finish_reason", "prompt_tokens", "completion_tokens", "latency_s", "error")


class Replay(Method):
    def __init__(self, records: dict, name: str):
        self.records, self.name = records, name

    async def predict(self, item):
        r = self.records[item.item_id]
        pred = Prediction(**{k: r[k] for k in PRED_KEYS if r.get(k) is not None})
        if item.task in ("detection", "segmentation"):  # boxes are read at the size the image was sent at (not saved)
            pred.sent_size = encode_image(item.images[0])[1]
        return pred


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def replay(run: Path, data_dir: Path, bench: str, out: Path) -> dict:
    old = {r["item_id"]: r for r in read_jsonl(run / "predictions.jsonl")}
    meta = json.loads((run / "run.json").read_text())
    model = meta["method"].get("model_id") or meta["method"]["name"]
    kf = key_formats({**model_spec(model), **({"box_order": meta["method"]["box_order"]} if meta["method"].get("box_order") else {})})
    b = Benchmark(data_dir, bench=bench)
    items_missing = [it.item_id for it in b.items if it.item_id not in old]
    b.items = [it for it in b.items if it.item_id in old]
    # The paper's runs disagree among themselves on the ag_task of 6 cocoa-pod detection items (a tie broken by hash
    # order at the time; the bundle has one fixed value). Score with the run's own labels and count them separately.
    own = {r["item_id"]: r["ag_task"] for r in read_jsonl(run / "items.jsonl")}
    ag_relabelled = sum(it.ag_task != own[it.item_id] for it in b.items)
    for it in b.items:
        it.ag_task = own[it.item_id]
    res = b.run(Replay(old, model), out, concurrency=256, resume=False, key_formats=kf)
    recs = read_jsonl(out / "predictions.jsonl")
    for r in recs:  # judge verdicts carried over
        o = old[r["item_id"]]
        for k in ("judge_correct", "judge_text", "judge_model"):
            if k in o:
                r[k] = o[k]
    write_jsonl(out / "predictions.jsonl", recs)
    res = finalize(out, recs)
    diffs = {}
    for r in recs:
        o = old[r["item_id"]]
        for k in SCORE_KEYS:
            if not same(r.get(k), o.get(k)):
                diffs.setdefault(k, []).append(r["item_id"])
    old_metrics = json.loads((run / "metrics.json").read_text())
    # A dataset row's ag_task label (its most common bucket) broke ties by set order before the fix, so the paper's
    # metrics.json can carry either label for a tied dataset; compare everything else.
    labels = sum(a["ag_task"] != b["ag_task"] for a, b in zip(res.metrics["datasets"], old_metrics["datasets"]))
    strip = lambda m: {**m, "datasets": [{k: v for k, v in r.items() if k != "ag_task"} for r in m["datasets"]]}
    return {"run": str(run), "items": len(recs), "paper_items": len(old), "not_in_bench": len(set(old) - {r["item_id"] for r in recs}),
            "bench_items_without_answer": len(items_missing), "skipped_datasets": len(b.skipped),
            "item_diffs": {k: len(v) for k, v in diffs.items()}, "examples": {k: v[:3] for k, v in diffs.items()},
            "metrics_identical": same(strip(json.loads(json.dumps(res.metrics))), strip(old_metrics)), "tied_dataset_labels": labels,
            "items_ag_task_differs_from_bundle": ag_relabelled}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--runs", default="runs/rejudge_v4/exp/seeds", type=Path)
    ap.add_argument("--bench", default="agml-full-v1")
    ap.add_argument("--only", default=None, help="Regex on run dir path.")
    a = ap.parse_args()
    import re

    runs = sorted(p.parent for p in a.runs.glob("**/predictions.jsonl") if not a.only or re.search(a.only, str(p)))
    report = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, run in enumerate(runs):
            r = replay(run, a.data_dir, a.bench, Path(tmp) / str(i))
            report.append(r)
            print(json.dumps(r), flush=True)
    ok = all(r["metrics_identical"] and not r["item_diffs"] and not r["not_in_bench"] for r in report)
    print(f"\n{sum(r['metrics_identical'] for r in report)}/{len(report)} runs with identical metrics; "
          f"{sum(1 for r in report if not r['item_diffs'])}/{len(report)} with identical per-item scores -> {'OK' if ok else 'DIFFERENCES'}")


if __name__ == "__main__":
    main()
