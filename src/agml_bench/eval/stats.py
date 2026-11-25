"""Statistics over repeated runs: seeds, option orders, NOTA rate, IDK, and model-vs-model tests.

Formats: mcqa_within, mcqa_across, oeq, detection, segmentation, itt_mc (ITT multiple choice), itt_other (open-ended,
yes/no, numeric). Accuracy uses the judge's verdict where there is one. NOTA/IDK behaviour and option order apply to
the multiple-choice formats (mcqa_*, itt_mc).

Every rate is computed per dataset (or per dataset × ag task) and macro-averaged, like the benchmark's summaries.
Uncertainty comes from a bootstrap over datasets (B resamples, 95% percentile interval); comparisons are paired over
datasets, with a Wilcoxon signed-rank test and an item-level McNemar test as checks.
"""

import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chisquare, wilcoxon

from agml_bench.eval.metrics import is_correct
from agml_bench.eval.report import load_runs, variant
from agml_bench.eval.resources import norm, prompts
from agml_bench.eval.runner import read_jsonl

B = 10_000
BUCKETS = ("disease", "pest_damage", "species_id", "quality_maturity")
FORMATS = ("mcqa_within", "mcqa_across", "oeq", "itt_mc", "itt_other", "detection", "segmentation")
MC = ("mcqa_within", "mcqa_across", "itt_mc")
LETTERS = "ABCDEFGHIJKLMNOP"


# ---- loading ----

def fmt_key(r: dict) -> str:
    if r["task"] == "classification":
        return r["format"]
    if r["task"] == "image_text_to_text":
        return "itt_mc" if r.get("qa_format") == "multiple_choice" else "itt_other"
    return r["task"]


def _is_nota_text(t) -> bool:
    return bool(t) and (t == prompts()["nota"] or re.match(r"none of (the )?above", norm(t)) is not None)


def load(paths) -> list[dict]:
    """Runs with their records. Each record gets `fk` (format key), `acc` (judge verdict where present), and for
    multiple choice `pick` (chosen option text), `nota_pick`, `pos` / `gt_pos` (chosen / correct option index)."""
    out = []
    for r in load_runs(paths):
        d = Path(r["dir"])
        items = {it["item_id"]: it for it in read_jsonl(d / "items.jsonl")} if (d / "items.jsonl").exists() else {}
        recs = []
        for x in read_jsonl(d / "predictions.jsonl"):
            x["fk"], x["acc"] = fmt_key(x), is_correct(x)
            if x["fk"] in MC:
                ch = (items.get(x["item_id"]) or {}).get("choices") or []
                if x["fk"] == "itt_mc":
                    picked = [LETTERS.index(c) for c in (x.get("pred_label") or "").split(",") if c in LETTERS[:len(ch)]]
                    gold = [LETTERS.index(c) for c in (x.get("gt_label") or "").split(",") if c in LETTERS[:len(ch)]]
                    x["pos"] = picked[0] if len(picked) == 1 else None
                    x["gt_pos"] = gold[0] if len(gold) == 1 else None
                    x["pick"] = ",".join(ch[i] for i in picked) or None
                else:
                    x["pos"] = ch.index(x["pred_label"]) if x.get("pred_label") in ch else None
                    x["gt_pos"] = ch.index(x["gt"]) if x.get("gt") in ch else None
                    x["pick"] = x.get("pred_label")
                x["nota_pick"] = x["pos"] is not None and _is_nota_text(ch[x["pos"]])
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

METRICS = {  # name: (value, where); NOTA/IDK metrics are for multiple-choice formats
    "accuracy": (lambda r: r["acc"], lambda r: True),
    "nota_recall": (lambda r: r["nota_pick"], lambda r: r["is_nota"]),
    "over_commit": (lambda r: not r["nota_pick"] and not r.get("idk"), lambda r: r["is_nota"]),
    "false_nota": (lambda r: r["nota_pick"], lambda r: not r["is_nota"]),
    "idk_rate": (lambda r: bool(r.get("idk")), lambda r: True),
    "selective_acc": (lambda r: r["acc"], lambda r: not r.get("idk")),
}


def behaviour(recs, by_bucket=False, mc=True) -> dict:
    return {m: unit_rates(recs, v, w, by_bucket) for m, (v, w) in METRICS.items() if mc or m == "accuracy"}


def _fmt(c: dict) -> str:
    return "—" if c.get("mean") is None else f"{100 * c['mean']:.1f} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"


def behaviour_table(runs: list[dict], label) -> pd.DataFrame:
    """Rows: run × format × (all + ag-task buckets); cells: macro rate [95% CI]."""
    rows = []
    for r in runs:
        for fk in FORMATS:
            recs = [x for x in r["recs"] if x["fk"] == fk]
            for scope in ("all",) + BUCKETS:
                sub = recs if scope == "all" else [x for x in recs if x["ag_task"] == scope]
                if sub:
                    b = behaviour(sub, mc=fk in MC)
                    rows.append({"model": r["model"], "variant": label(r), "format": fk, "ag_task": scope, "n_items": len(sub),
                                 **{m: _fmt(ci(b[m])) if m in b else "" for m in METRICS}})
    return pd.DataFrame(rows)


# ---- experiment reports ----

def _by_model(runs):
    g = defaultdict(list)
    for r in runs:
        g[r["model"]].append(r)
    return g


def _same_answer(fk: str):
    """What counts as the same answer across repeats: the option (multiple choice) or the outcome (other formats)."""
    return (lambda x: x["pick"]) if fk in MC else (lambda x: x["acc"] >= 0.5)


def seeds_report(runs) -> pd.DataFrame:
    """Per model and format: accuracy per seed, sd across seeds, share of items answered the same in every seed."""
    rows = []
    for model, rs in _by_model(runs).items():
        rs = sorted(rs, key=lambda r: str(r["variant"]["gen_seed"]))
        for fk in FORMATS:
            sets = [[x for x in r["recs"] if x["fk"] == fk] for r in rs]
            if not all(sets):
                continue
            accs = [ci(unit_rates(s, lambda x: x["acc"]))["mean"] for s in sets]
            same = _same_answer(fk)
            picks = [{x["item_id"]: same(x) for x in s} for s in sets]
            common = set.intersection(*(set(p) for p in picks))
            agree = np.mean([len({p[k] for p in picks}) == 1 for k in common]) if common else None
            rows.append({"model": model, "format": fk, "seeds": [r["variant"]["gen_seed"] for r in rs],
                         "accuracy per seed": [round(100 * a, 2) for a in accs], "mean": round(100 * np.mean(accs), 2),
                         "sd": round(100 * np.std(accs, ddof=1), 2) if len(accs) > 1 else None,
                         "same answer in every seed %": round(100 * agree, 1) if agree is not None else None, "items": len(common)})
    return pd.DataFrame(rows)


def order_report(runs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multiple choice: accuracy across option orders, answer consistency (same option in every order), position bias."""
    rows, pos_rows = [], []
    for model, rs in _by_model(runs).items():
        for fk in MC:
            sets = [[x for x in r["recs"] if x["fk"] == fk] for r in rs]
            if not all(sets):
                continue
            accs = [ci(unit_rates(s, lambda x: x["acc"]))["mean"] for s in sets]
            picks = [{x["item_id"]: x["pick"] for x in s} for s in sets]
            common = set.intersection(*(set(p) for p in picks))
            consistent = np.mean([len({p[k] for p in picks}) == 1 for k in common]) if common else None
            recs = [x for s in sets for x in s if x["pos"] is not None and x["gt_pos"] is not None]
            n = max((x["n_opts"] for x in recs), default=0)
            chosen = np.bincount([x["pos"] for x in recs], minlength=n)
            correct_at = np.bincount([x["gt_pos"] for x in recs], minlength=n)
            keep = correct_at > 0
            p_chi = chisquare(chosen[keep], correct_at[keep] * chosen[keep].sum() / correct_at[keep].sum()).pvalue if keep.sum() > 1 else None
            rows.append({"model": model, "format": fk, "orders": len(rs), "accuracy mean": round(100 * np.mean(accs), 2),
                         "sd across orders": round(100 * np.std(accs, ddof=1), 2) if len(accs) > 1 else None,
                         "same answer in every order %": round(100 * consistent, 1) if consistent is not None else None,
                         "position bias p (chi-square)": p_chi})
            for i in range(n):
                at = [x for x in recs if x["gt_pos"] == i]
                pos_rows.append({"model": model, "format": fk, "position": i + 1,
                                 "chosen %": round(100 * chosen[i] / max(chosen.sum(), 1), 1),
                                 "correct answer here %": round(100 * correct_at[i] / max(correct_at.sum(), 1), 1),
                                 "accuracy when correct answer here %": round(100 * np.mean([x["acc"] for x in at]), 1) if at else None})
    return pd.DataFrame(rows), pd.DataFrame(pos_rows)


def model_pairs(runs, metric: str = "accuracy", fk: str = "mcqa_across") -> pd.DataFrame:
    """All model pairs on one metric and format: paired dataset bootstrap, Wilcoxon, McNemar, Holm-adjusted p."""
    value, where = METRICS[metric]
    per = {r["model"]: [x for x in r["recs"] if x["fk"] == fk] for r in runs}
    per = {k: v for k, v in per.items() if v}
    rows = []
    for a, b in combinations(sorted(per), 2):
        pa = paired(unit_rates(per[a], value, where), unit_rates(per[b], value, where))
        mc = mcnemar({x["item_id"]: value(x) >= 0.5 for x in per[a] if where(x)}, {x["item_id"]: value(x) >= 0.5 for x in per[b] if where(x)})
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
        w = [x for x in r["recs"] if x["fk"] == "mcqa_within" and where(x)]
        a = [x for x in r["recs"] if x["fk"] == "mcqa_across" and where(x)]
        if not w or not a:
            continue
        pa = paired(unit_rates(w, value), unit_rates(a, value))
        mc = mcnemar({x["sample_id"]: bool(value(x)) for x in w}, {x["sample_id"]: bool(value(x)) for x in a})
        rows.append({"model": r["model"], "variant": label(r), "metric": metric,
                     "within − across (pts)": None if pa["diff"] is None else round(100 * pa["diff"], 2),
                     "95% CI": None if pa["diff"] is None else f"[{100 * pa['lo']:+.1f}, {100 * pa['hi']:+.1f}]",
                     "p (bootstrap)": pa.get("p_boot"), "p (McNemar)": mc["p"]})
    return pd.DataFrame(rows)


def _default_opts(r, *, order=True, nota=True, idk=True) -> bool:
    """True if the run uses the standard option layout (except for the settings being varied)."""
    v = r["variant"]
    return ((not order or (v["order_seed"] is None and v["nota_pos"] == "end"))
            and (not nota or v["nota_rate"] in ("default", 0.3)) and (not idk or v["idk"] is True))


def _rate(v) -> float:
    return 0.3 if v in ("default", None) else float(v)


def ablation_report(runs, vary: str, ref) -> pd.DataFrame:
    """Each variant vs the reference run of the same model (same items): accuracy and NOTA/IDK behaviour, paired over
    datasets. `vary` names the setting; `ref(r)` marks the reference runs."""
    rows = []
    for model, rs in _by_model(runs).items():
        refs = [r for r in rs if ref(r)]
        if not refs:
            continue
        base = sorted(refs, key=lambda r: str(r["variant"]["gen_seed"]))[0]
        for r in sorted(rs, key=lambda r: str(r["variant"][vary])):
            for fk in MC:
                a = [x for x in r["recs"] if x["fk"] == fk]
                b = [x for x in base["recs"] if x["fk"] == fk]
                if not a or not b:
                    continue
                row = {"model": model, vary: r["variant"][vary], "reference": "yes" if r is base else "", "format": fk}
                for m in ("accuracy", "nota_recall", "false_nota", "over_commit", "idk_rate"):
                    value, where = METRICS[m]
                    ra = unit_rates(a, value, where)
                    row[m] = _fmt(ci(ra))
                    if r is not base:
                        pa = paired(ra, unit_rates(b, value, where))
                        row[f"Δ{m} vs ref"] = None if pa["diff"] is None else f"{100 * pa['diff']:+.1f} [{100 * pa['lo']:+.1f}, {100 * pa['hi']:+.1f}] p={pa['p_boot']:.3g}"
                rows.append(row)
    return pd.DataFrame(rows)


def write_report(paths, out: Path, title: str = "Statistics") -> Path:
    runs = load(paths)
    out.mkdir(parents=True, exist_ok=True)
    differing = [k for k in (runs[0]["variant"] if runs else {}) if len({str(r["variant"][k]) for r in runs}) > 1]
    label = lambda r: ", ".join(f"{k}={r['variant'][k]}" for k in differing) or "-"
    parts = [f"# {title}\n", f"Runs: {len(runs)} · varying: {', '.join(differing) or 'nothing'} · bootstrap B={B} over datasets\n",
             "## Behaviour (macro over datasets, 95% CI; NOTA/IDK metrics for multiple choice)\n",
             behaviour_table(runs, label).to_markdown(index=False)]
    seed_runs = [r for r in runs if _default_opts(r)]
    if len({str(r["variant"]["gen_seed"]) for r in seed_runs}) > 1:
        parts += ["\n## Seeds (default options)\n", seeds_report(seed_runs).to_markdown(index=False)]
    order_runs = [r for r in runs if r["variant"]["order_seed"] is not None and _default_opts(r, order=False)]
    if order_runs:
        summary, pos = order_report(order_runs)
        parts += ["\n## Option order\n", summary.to_markdown(index=False), "\n### Position\n", pos.to_markdown(index=False)]
    nota_runs = [r for r in runs if _default_opts(r, nota=False)]
    if len({_rate(r["variant"]["nota_rate"]) for r in nota_runs}) > 1:
        parts += ["\n## NOTA rate (paired vs the default-rate run)\n",
                  ablation_report(nota_runs, "nota_rate", lambda r: _rate(r["variant"]["nota_rate"]) == 0.3).to_markdown(index=False)]
    idk_runs = [r for r in runs if _default_opts(r, idk=False)]
    if len({r["variant"]["idk"] for r in idk_runs}) > 1:
        parts += ["\n## IDK removed (paired vs the run with IDK)\n",
                  ablation_report(idk_runs, "idk", lambda r: r["variant"]["idk"] is True).to_markdown(index=False)]
    base = [r for r in seed_runs if str(r["variant"]["gen_seed"]) == str(min((str(x["variant"]["gen_seed"]) for x in seed_runs), default=""))] or seed_runs
    if len({r["model"] for r in base}) > 1:
        for fk in FORMATS:
            df = model_pairs(base, "accuracy", fk)
            if len(df):
                parts += [f"\n## Model pairs: accuracy, {fk} ({label(base[0])})\n", df.to_markdown(index=False)]
    wa = [within_vs_across(runs, m, label) for m in ("accuracy", "false_nota", "over_commit")]
    if any(len(x) for x in wa):
        parts += ["\n## Within vs across distractors (same images)\n", pd.concat(wa).to_markdown(index=False)]
    p = out / "stats.md"
    p.write_text("\n".join(parts) + "\n")
    return p
