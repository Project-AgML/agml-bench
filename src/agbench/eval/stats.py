"""Statistics over repeated MCQA runs: seeds, option orders, NOTA rate, IDK, and model-vs-model tests.

Every rate is computed per dataset (or per dataset × ag task) and macro-averaged, like the benchmark's summaries.
Uncertainty comes from a bootstrap over datasets (B resamples, 95% percentile interval); model comparisons are
paired over datasets, with a Wilcoxon signed-rank test and an item-level McNemar test as checks.
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chisquare, wilcoxon

from agbench.eval.report import load_runs, variant
from agbench.eval.resources import prompts
from agbench.eval.runner import read_jsonl

B = 10_000
BUCKETS = ("disease", "pest_damage", "species_id", "quality_maturity")


# ---- loading ----

def load(paths) -> list[dict]:
    """Runs with MCQA records; each record gets `pos` (chosen option index) and `gt_pos` (correct option index)."""
    nota, out = prompts()["nota"], []
    for r in load_runs(paths):
        d = Path(r["dir"])
        items = {it["item_id"]: it for it in read_jsonl(d / "items.jsonl")} if (d / "items.jsonl").exists() else {}
        recs = []
        for x in read_jsonl(d / "predictions.jsonl"):
            if x["task"] != "classification" or not x["format"].startswith("mcqa"):
                continue
            ch = (items.get(x["item_id"]) or {}).get("choices") or []
            x["nota_pick"] = x.get("pred_label") == nota
            x["pos"] = ch.index(x["pred_label"]) if x.get("pred_label") in ch else None
            x["gt_pos"] = ch.index(x["gt"]) if x.get("gt") in ch else None
            x["n_opts"] = len(ch)
            recs.append(x)
        if recs:
            out.append({**r, "model": r["run"]["method"].get("model_id") or r["name"], "variant": variant(r["run"]), "recs": recs})
    return out


# ---- per-unit rates and the dataset bootstrap ----

def unit_rates(recs, value, where=lambda r: True, by_bucket=False) -> dict:
    """{unit: mean of value(r) over records passing `where`}; unit = dataset (or dataset, ag task)."""
    acc = defaultdict(list)
    for r in recs:
        if where(r):
            acc[(r["dataset"], r["ag_task"]) if by_bucket else r["dataset"]].append(float(value(r)))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def _boot_idx(n: int, b: int = B, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def ci(rates: dict) -> dict:
    """Macro mean over units with a 95% bootstrap interval over units."""
    v = np.array(list(rates.values()))
    if not len(v):
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = v[_boot_idx(len(v))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": len(v)}


def paired(a: dict, b: dict) -> dict:
    """a − b over the units both have: mean, 95% bootstrap CI, bootstrap p, Wilcoxon p."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    if not len(d):
        return {"diff": None, "n": 0}
    m = d[_boot_idx(len(d))].mean(1)
    p = min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))
    w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "p_boot": float(p), "p_wilcoxon": float(w), "n": len(d)}


def mcnemar(a: dict, b: dict) -> dict:
    """Exact McNemar on paired items: a, b = {item_id: bool}."""
    keys = set(a) & set(b)
    ab = sum(a[k] and not b[k] for k in keys)
    ba = sum(b[k] and not a[k] for k in keys)
    return {"a_only": ab, "b_only": ba, "p": float(binomtest(ab, ab + ba, 0.5).pvalue) if ab + ba else 1.0}


def holm(ps: list[float]) -> list[float]:
    order, adj, run = np.argsort(ps), [0.0] * len(ps), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


# ---- behaviour metrics ----

METRICS = {  # name: (value, where)
    "accuracy": (lambda r: r["correct"], lambda r: True),
    "nota_recall": (lambda r: r["nota_pick"], lambda r: r["is_nota"]),
    "over_commit": (lambda r: not r["nota_pick"] and not r.get("idk"), lambda r: r["is_nota"]),
    "false_nota": (lambda r: r["nota_pick"], lambda r: not r["is_nota"]),
    "idk_rate": (lambda r: bool(r.get("idk")), lambda r: True),
    "selective_acc": (lambda r: r["correct"], lambda r: not r.get("idk")),
}


def behaviour(recs, by_bucket=False) -> dict:
    return {m: unit_rates(recs, v, w, by_bucket) for m, (v, w) in METRICS.items()}


def _fmt(c: dict) -> str:
    return "—" if c.get("mean") is None else f"{100 * c['mean']:.1f} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"


def behaviour_table(runs: list[dict], label) -> pd.DataFrame:
    """Rows: run × format × (all + ag-task buckets); cells: macro rate [95% CI]."""
    rows = []
    for r in runs:
        for fmt in ("mcqa_within", "mcqa_across"):
            recs = [x for x in r["recs"] if x["format"] == fmt]
            if not recs:
                continue
            for scope in ("all",) + BUCKETS:
                sub = recs if scope == "all" else [x for x in recs if x["ag_task"] == scope]
                if sub:
                    b = behaviour(sub)
                    rows.append({"model": r["model"], "variant": label(r), "format": fmt, "ag_task": scope, "n_items": len(sub),
                                 **{m: _fmt(ci(v)) for m, v in b.items()}})
    return pd.DataFrame(rows)


# ---- experiment reports ----

def _by_model(runs):
    g = defaultdict(list)
    for r in runs:
        g[r["model"]].append(r)
    return g


def seeds_report(runs) -> pd.DataFrame:
    """Per model and format: accuracy per seed, sd across seeds, share of items answered identically in every seed."""
    rows = []
    for model, rs in _by_model(runs).items():
        rs = sorted(rs, key=lambda r: str(r["variant"]["gen_seed"]))
        for fmt in ("mcqa_within", "mcqa_across"):
            accs = [ci(unit_rates([x for x in r["recs"] if x["format"] == fmt], lambda x: x["correct"]))["mean"] for r in rs]
            picks = [{x["item_id"]: x["pred_label"] for x in r["recs"] if x["format"] == fmt} for r in rs]
            common = set.intersection(*(set(p) for p in picks)) if picks else set()
            agree = np.mean([len({p[k] for p in picks}) == 1 for k in common]) if common else None
            rows.append({"model": model, "format": fmt, "seeds": [r["variant"]["gen_seed"] for r in rs],
                         "accuracy per seed": [round(100 * a, 2) for a in accs if a is not None],
                         "mean": round(100 * np.mean(accs), 2), "sd": round(100 * np.std(accs, ddof=1), 2) if len(accs) > 1 else None,
                         "items identical across seeds %": round(100 * agree, 1) if agree is not None else None, "items": len(common)})
    return pd.DataFrame(rows)


def order_report(runs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy across option orders, answer consistency (same content in every order), and position bias."""
    rows, pos_rows = [], []
    for model, rs in _by_model(runs).items():
        for fmt in ("mcqa_within", "mcqa_across"):
            sets = [[x for x in r["recs"] if x["format"] == fmt] for r in rs]
            accs = [ci(unit_rates(s, lambda x: x["correct"]))["mean"] for s in sets]
            picks = [{x["item_id"]: x["pred_label"] for x in s} for s in sets]
            common = set.intersection(*(set(p) for p in picks)) if picks else set()
            consistent = np.mean([len({p[k] for p in picks}) == 1 for k in common]) if common else None
            recs = [x for s in sets for x in s if x["pos"] is not None and x["gt_pos"] is not None]
            n = max((x["n_opts"] for x in recs), default=0)
            chosen = np.bincount([x["pos"] for x in recs], minlength=n)
            correct_at = np.bincount([x["gt_pos"] for x in recs], minlength=n)
            p_chi = chisquare(chosen, correct_at * chosen.sum() / correct_at.sum()).pvalue if n and correct_at.all() else None
            rows.append({"model": model, "format": fmt, "orders": len(rs), "accuracy mean": round(100 * np.mean(accs), 2),
                         "sd across orders": round(100 * np.std(accs, ddof=1), 2) if len(accs) > 1 else None,
                         "same answer in every order %": round(100 * consistent, 1) if consistent is not None else None,
                         "position bias p (chi-square)": p_chi})
            for i in range(n):
                at = [x for x in recs if x["gt_pos"] == i]
                pos_rows.append({"model": model, "format": fmt, "position": i + 1,
                                 "chosen %": round(100 * chosen[i] / max(chosen.sum(), 1), 1),
                                 "correct answer here %": round(100 * correct_at[i] / max(correct_at.sum(), 1), 1),
                                 "accuracy when correct answer here %": round(100 * np.mean([x["correct"] for x in at]), 1) if at else None})
    return pd.DataFrame(rows), pd.DataFrame(pos_rows)


def model_pairs(runs, metric: str = "accuracy", fmt: str = "mcqa_across") -> pd.DataFrame:
    """All model pairs on one metric and format: paired dataset bootstrap, Wilcoxon, McNemar, Holm-adjusted p."""
    value, where = METRICS[metric]
    per = {r["model"]: [x for x in r["recs"] if x["format"] == fmt] for r in runs}
    rows = []
    for a, b in combinations(sorted(per), 2):
        pa = paired(unit_rates(per[a], value, where), unit_rates(per[b], value, where))
        mc = mcnemar({x["item_id"]: bool(value(x)) for x in per[a] if where(x)}, {x["item_id"]: bool(value(x)) for x in per[b] if where(x)})
        rows.append({"a": a, "b": b, "a − b (pts)": None if pa["diff"] is None else round(100 * pa["diff"], 2),
                     "95% CI": None if pa["diff"] is None else f"[{100 * pa['lo']:+.1f}, {100 * pa['hi']:+.1f}]",
                     "p (bootstrap)": pa.get("p_boot"), "p (Wilcoxon)": pa.get("p_wilcoxon"), "p (McNemar)": mc["p"],
                     "datasets": pa["n"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["p (bootstrap, Holm)"] = holm(df["p (bootstrap)"].tolist())
    return df


def within_vs_across(runs, metric: str = "false_nota", label=lambda r: "-") -> pd.DataFrame:
    """Same images, within- vs across-dataset distractors: paired over datasets and McNemar over samples."""
    value, where = METRICS[metric]
    rows = []
    for r in runs:
        w = [x for x in r["recs"] if x["format"] == "mcqa_within" and where(x)]
        a = [x for x in r["recs"] if x["format"] == "mcqa_across" and where(x)]
        pa = paired(unit_rates(w, value), unit_rates(a, value))
        mc = mcnemar({x["sample_id"]: bool(value(x)) for x in w}, {x["sample_id"]: bool(value(x)) for x in a})
        rows.append({"model": r["model"], "variant": label(r), "metric": metric,
                     "within − across (pts)": None if pa["diff"] is None else round(100 * pa["diff"], 2),
                     "95% CI": None if pa["diff"] is None else f"[{100 * pa['lo']:+.1f}, {100 * pa['hi']:+.1f}]",
                     "p (bootstrap)": pa.get("p_boot"), "p (McNemar)": mc["p"]})
    return pd.DataFrame(rows)


def write_report(paths, out: Path, title: str = "MCQA statistics") -> Path:
    runs = load(paths)
    out.mkdir(parents=True, exist_ok=True)
    differing = [k for k in (runs[0]["variant"] if runs else {}) if len({str(r["variant"][k]) for r in runs}) > 1]
    label = lambda r: ", ".join(f"{k}={r['variant'][k]}" for k in differing) or "-"
    parts = [f"# {title}\n", f"Runs: {len(runs)} · varying: {', '.join(differing) or 'nothing'} · bootstrap B={B} over datasets\n"]
    parts += ["## Behaviour (macro over datasets, 95% CI)\n", behaviour_table(runs, label).to_markdown(index=False)]
    if "gen_seed" in differing:
        parts += ["\n## Seeds\n", seeds_report(runs).to_markdown(index=False)]
    if "order_seed" in differing:
        summary, pos = order_report(runs)
        parts += ["\n## Option order\n", summary.to_markdown(index=False), "\n### Position\n", pos.to_markdown(index=False)]
    base = [r for r in runs if all(str(r["variant"][k]) == str(runs[0]["variant"][k]) for k in differing)]
    if len({r["model"] for r in base}) > 1:
        for fmt in ("mcqa_within", "mcqa_across"):
            parts += [f"\n## Model pairs: accuracy, {fmt} ({label(base[0])})\n", model_pairs(base, "accuracy", fmt).to_markdown(index=False)]
    parts += ["\n## Within vs across distractors (same images)\n",
              pd.concat([within_vs_across(runs, m, label) for m in ("accuracy", "false_nota", "over_commit")]).to_markdown(index=False)]
    p = out / "stats.md"
    p.write_text("\n".join(parts) + "\n")
    return p
