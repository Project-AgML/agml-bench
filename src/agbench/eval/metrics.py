"""Accuracy and F1 per dataset, summarized by CV task and by ag task."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from agbench.eval.resources import norm

BOX_TASKS = ("detection", "segmentation")
CLOSED = ("multiple_choice", "yes_no", "numeric")


def is_correct(r: dict) -> float:
    if r.get("judge_correct") is not None:
        return float(r["judge_correct"])
    return float(r["correct"])


def cv_key(r: dict) -> str:
    return f"classification/{r['format']}" if r["task"] == "classification" else r["task"]


def _labels(r: dict, prefix: bool) -> tuple[str, str]:
    """(true, pred) label pair for macro-F1. `prefix` keeps labels distinct across datasets when pooling."""
    gt = str(r.get("gt_label"))
    if r["task"] == "classification" and r["format"] == "oeq":
        pred = gt if is_correct(r) else (r.get("pred_class") if r.get("pred_class") and norm(r["pred_class"]) != norm(gt) else "__other__")
    else:
        pred = gt if is_correct(r) else str(r.get("pred_label"))
        if pred == gt and not is_correct(r):
            pred = "__wrong__"
    pre = f"{r['dataset']}::" if prefix else ""
    return pre + gt, pre + str(pred)


def _macro_f1(recs: list[dict], prefix: bool) -> float:
    pairs = [_labels(r, prefix) for r in recs]
    y, yhat = [a for a, _ in pairs], [b for _, b in pairs]
    return float(f1_score(y, yhat, labels=sorted(set(y)), average="macro", zero_division=0))


def group(recs: list[dict], pooled: bool = False) -> dict:
    """Metrics for one set of records (a dataset, or a pooled set when `pooled`)."""
    n = len(recs)
    out = {"n": n, "accuracy": float(np.mean([is_correct(r) for r in recs])) if n else None}
    box = [r for r in recs if r["task"] in BOX_TASKS]
    lab = [r for r in recs if r["task"] == "classification" or (r["task"] == "image_text_to_text" and r.get("qa_format") in CLOSED)]
    opn = [r for r in recs if r["task"] == "image_text_to_text" and r.get("qa_format") not in CLOSED]
    parts = []
    if box:
        tp, fp, fn = (sum(r[k] for r in box) for k in ("tp", "fp", "fn"))
        parts.append((len(box), 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0))
        matched = [r for r in box if r["tp"]]
        out["mean_iou"] = float(np.mean([r["mean_iou"] for r in matched])) if matched else 0.0
    if lab:
        parts.append((len(lab), _macro_f1(lab, prefix=pooled)))
    if opn:
        parts.append((len(opn), float(np.mean([r.get("token_f1", 0.0) for r in opn]))))
    out["f1"] = sum(k * v for k, v in parts) / sum(k for k, _ in parts) if parts else None
    nota = [r for r in recs if r.get("is_nota")]
    out.update(
        idk_rate=float(np.mean([bool(r.get("idk")) for r in recs])) if n else 0.0,
        nota_accuracy=float(np.mean([is_correct(r) for r in nota])) if nota else None,
        truncation_rate=float(np.mean([bool(r.get("truncated")) for r in recs])) if n else 0.0,
        parse_fail_rate=float(np.mean([not r.get("parse_ok") for r in recs])) if n else 0.0,
        error_rate=float(np.mean([bool(r.get("error")) for r in recs])) if n else 0.0,
        judged=sum(r.get("judge_correct") is not None for r in recs), needs_judge=sum(bool(r.get("needs_judge")) for r in recs),
        latency_s=float(np.mean([r.get("latency_s", 0) for r in recs])) if n else 0.0,
        completion_tokens=float(np.mean([r.get("completion_tokens", 0) for r in recs])) if n else 0.0,
    )
    return out


def _summarize(rows: list[dict], recs_for, key: str) -> list[dict]:
    """Macro (mean over datasets) and micro (pooled records) per value of `key`."""
    out = []
    for k in sorted({r[key] for r in rows}):
        ds_rows = [r for r in rows if r[key] == k]
        pooled = group([x for r in ds_rows for x in recs_for(r)], pooled=True)
        f1s = [r["f1"] for r in ds_rows if r["f1"] is not None]
        out.append({key: k, "datasets": len({r["dataset"] for r in ds_rows}), "n": pooled["n"],
                    "accuracy_macro": float(np.mean([r["accuracy"] for r in ds_rows])), "f1_macro": float(np.mean(f1s)) if f1s else None,
                    "accuracy_micro": pooled["accuracy"], "f1_micro": pooled["f1"],
                    "idk_rate": pooled["idk_rate"], "nota_accuracy": pooled["nota_accuracy"],
                    "truncation_rate": pooled["truncation_rate"], "parse_fail_rate": pooled["parse_fail_rate"]})
    return out


def compute(records: list[dict]) -> dict:
    by_ds, by_ag = defaultdict(list), defaultdict(list)
    for r in records:
        by_ds[(r["dataset"], cv_key(r), None)].append(r)
        by_ag[(r["dataset"], cv_key(r), r["ag_task"])].append(r)
    datasets = [{"dataset": d, "cv_task": k, "ag_task": max({x["ag_task"] for x in v}, key=[x["ag_task"] for x in v].count),
                 **group(v)} for (d, k, _), v in sorted(by_ds.items())]
    ag_rows = [{"dataset": d, "cv_task": k, "ag_task": a, **group(v)} for (d, k, a), v in sorted(by_ag.items())]
    ag_recs = lambda r: by_ag[(r["dataset"], r["cv_task"], r["ag_task"])]
    ag_summary = []
    for cv in sorted({r["cv_task"] for r in ag_rows}):
        for row in _summarize([r for r in ag_rows if r["cv_task"] == cv], ag_recs, "ag_task"):
            ag_summary.append({"cv_task": cv, **row})
    for row in _summarize(ag_rows, ag_recs, "ag_task"):
        ag_summary.append({"cv_task": "all", **row})
    return {
        "datasets": datasets,
        "cv_task": _summarize(datasets, lambda r: by_ds[(r["dataset"], r["cv_task"], None)], "cv_task"),
        "ag_task": ag_summary,
        "n_records": len(records),
        "pending_judge": sum(bool(r.get("needs_judge")) and r.get("judge_correct") is None for r in records),
    }
