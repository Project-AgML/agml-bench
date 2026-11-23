"""Appendix: which evaluation variables drive accuracy (relative-importance analysis).

Item-level linear probability model, correct (0/1) ~ main effects of
  dataset, model, answer format (MCQA vs OEQ), distractor source (within vs across), NOTA rate (10/30/50%),
  IDK offered, option order (default vs randomized, incl. NOTA/IDK position), generation seed (0/1/2),
weighted so that each dataset counts equally within each run and format (the paper's macro averages).
Importance = Shapley decomposition of R^2 over the variable groups (LMG). Effects = coefficients in points with
dataset-clustered 95% CIs. Interaction blocks are reported as the R^2 they add to the main-effects model.

Runs: the 7 models that took part in every experiment (Nemotron ran only Experiment 1).
  Experiment 1: runs/rejudge_v4/exp/seeds/s{0,1,2}   (OEQ judged with judge v4)
  Experiments 2-4: runs/runpod_exp/runs/{order/o1-o5, nota/n0.1, nota/n0.5, noidk}
Fits: classification (MCQA within, MCQA across, OEQ) and, separately, VQA multiple choice.

usage: uv run --with numpy python research/paper/scripts/make_drivers.py
writes research/paper/tables/drivers.tex and research/paper/drivers.json
"""
import collections
import glob
import itertools
import json
import math
import os
import re

import numpy as np

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(P))
# Inputs (override to reproduce from your own runs; see paper Appendix D and research/paper/scripts/reproduce.sh):
#   AGBENCH_SEEDS  Experiment 1 run dirs, one subdir per seed   (default runs/rejudge_v4/exp/seeds)
#   AGBENCH_EXP24  Experiments 2-4 run dirs                       (default runs/runpod_exp/runs)
#   AGBENCH_STATS  agbench eval stats outputs (seeds/, exp24/)    (default research/reports/stats)
SEEDS = os.path.join(ROOT, os.environ.get("AGBENCH_SEEDS", "runs/rejudge_v4/exp/seeds"))
MODELS = ["Qwen3.6-35B-A3B", "Qwen3.8-27B", "gemma-4-31B", "gemma-4-12B", "Muse-Glimmer-30B", "MiniCPM-V-4.7", "Ministral-3-14B"]
DISP = {"gemma-4-31B": "Gemma-4-31B", "gemma-4-12B": "Gemma-4-12B", "MiniCPM-V-4.7": "MiniCPM-V-4.7"}


def model_of(path):
    name = os.path.basename(path.rstrip("/")).split("__")[-1]
    return next((m for m in MODELS if name.startswith(m)), None)


def runs():
    """(dir, condition dict) for every run."""
    out = []
    base = dict(seed=0, order=0, nota=30, idk=1)
    for s in (0, 1, 2):
        for d in glob.glob(os.path.join(SEEDS, f"s{s}/*/")):
            out.append((d, dict(base, seed=s)))
    R = os.path.join(ROOT, os.environ.get("AGBENCH_EXP24", "runs/runpod_exp/runs"))
    for o in range(1, 6):
        out += [(d, dict(base, order=1)) for d in glob.glob(f"{R}/order/o{o}/*/")]
    for n in ("0.1", "0.5"):
        out += [(d, dict(base, nota=int(float(n) * 100))) for d in glob.glob(f"{R}/nota/n{n}/*/")]
    out += [(d, dict(base, idk=0)) for d in glob.glob(f"{R}/noidk/*/")]
    return [(d, c) for d, c in out if model_of(d)]


def score(p):
    if p.get("needs_judge"):
        return 1.0 if p.get("judge_correct") else 0.0
    return 1.0 if p.get("correct") else 0.0


def load(kind):
    """rows: (dataset, model, fmt, cond, y) for kind 'cls' or 'vqa'."""
    rows = []
    for d, c in runs():
        m = model_of(d)
        for line in open(os.path.join(d, "predictions.jsonl")):
            p = json.loads(line)
            if kind == "cls" and p["task"] == "classification":
                rows.append((p["dataset"], m, p["format"], c, score(p)))
            elif kind == "vqa" and p["task"] == "image_text_to_text" and p.get("qa_format") == "multiple_choice":
                # the ablation runs hold only the multiple-choice VQA questions; keep the same set in Experiment 1
                rows.append((p["dataset"], m, "vqa_mc", c, score(p)))
    return rows


def design(rows, kind):
    """X (n x k), y, w, cluster ids, and column groups."""
    dsets = sorted({r[0] for r in rows})
    cols, groups = ["const"], collections.OrderedDict()

    def add(group, names):
        groups.setdefault(group, [])
        for n in names:
            groups[group].append(len(cols)); cols.append(n)

    add("Dataset", [f"ds:{d}" for d in dsets[1:]])
    add("Model", [f"m:{m}" for m in MODELS[1:]])
    if kind == "cls":
        add("Answer format (OEQ vs.\\ MCQA)", ["oeq"])
        add("Distractor source (across vs.\\ within)", ["across"])
    add("NOTA rate", ["nota10", "nota50"])
    add("IDK offered", ["noidk"])
    add("Option order and NOTA/IDK position", ["order"])
    add("Generation seed", ["seed1", "seed2"])
    main_k = len(cols)
    # interaction blocks (not in the Shapley decomposition)
    inter = collections.OrderedDict()
    prompt = [c for c in ("oeq", "across", "nota10", "nota50", "noidk", "order") if c in cols]
    inter["Model $\\times$ prompt variables"] = [f"{m}*{c}" for m in MODELS[1:] for c in prompt]
    if kind == "cls":
        inter["Dataset $\\times$ answer format and distractor source"] = [f"{d}*{c}" for d in dsets[1:] for c in ("oeq", "across")]
    ci = {c: i for i, c in enumerate(cols)}
    icols = [c for v in inter.values() for c in v]
    k = len(cols) + len(icols)
    icol_idx = {c: len(cols) + i for i, c in enumerate(icols)}
    di, mi = {d: i for i, d in enumerate(dsets)}, {m: i for i, m in enumerate(MODELS)}

    # weights: every dataset counts equally within a (run, format) cell
    cnt = collections.Counter((r[1], json.dumps(r[3], sort_keys=True), r[2], r[0]) for r in rows)
    X = np.zeros((len(rows), k), dtype=np.float32)
    y = np.zeros(len(rows)); w = np.zeros(len(rows)); g = np.zeros(len(rows), dtype=np.int32)
    for i, (d, m, f, c, s) in enumerate(rows):
        x = X[i]
        x[0] = 1
        if di[d]: x[ci[f"ds:{d}"]] = 1
        if mi[m]: x[ci[f"m:{m}"]] = 1
        on = {"oeq": f == "oeq", "across": f == "mcqa_across", "nota10": c["nota"] == 10, "nota50": c["nota"] == 50,
              "noidk": c["idk"] == 0, "order": c["order"] == 1, "seed1": c["seed"] == 1, "seed2": c["seed"] == 2}
        for n, v in on.items():
            if v and n in ci:
                x[ci[n]] = 1
        for n in prompt:
            if on[n]:
                if mi[m]: x[icol_idx[f"{m}*{n}"]] = 1
                if kind == "cls" and n in ("oeq", "across") and di[d]: x[icol_idx[f"{d}*{n}"]] = 1
        y[i] = s
        w[i] = 1.0 / cnt[(m, json.dumps(c, sort_keys=True), f, d)]
        g[i] = di[d]
    return X, y, w, g, cols, groups, main_k, inter, icol_idx, dsets


def r2(XtX, Xty, yty, sw, swy, idx):
    """weighted R^2 of the model with the intercept plus columns idx (from the cross-products)."""
    idx = [0] + list(idx)
    A = XtX[np.ix_(idx, idx)]
    b = np.linalg.lstsq(A, Xty[idx], rcond=None)[0]
    sse = yty - b @ Xty[idx]
    sst = yty - swy ** 2 / sw
    return 1 - sse / sst


def fit(kind):
    rows = load(kind)
    X, y, w, g, cols, groups, main_k, inter, icol_idx, dsets = design(rows, kind)
    Xw = X * w[:, None]
    XtX = (Xw.T @ X).astype(np.float64)
    Xty = Xw.T @ y
    yty, sw, swy = float((w * y * y).sum()), float(w.sum()), float((w * y).sum())

    # Shapley (LMG) decomposition over the main-effect groups
    names = list(groups)
    G = len(names)
    cache = {}

    def R(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = r2(XtX, Xty, yty, sw, swy, [c for n in S for c in groups[n]]) if S else 0.0
        return cache[key]

    shap = {}
    for n in names:
        others = [o for o in names if o != n]
        v = 0.0
        for k in range(G):
            for S in itertools.combinations(others, k):
                v += math.factorial(k) * math.factorial(G - k - 1) / math.factorial(G) * (R(S + (n,)) - R(S))
        shap[n] = v
    full = R(tuple(names))

    # coefficients (points) with dataset-clustered CIs, main-effects model
    idx = list(range(main_k))
    A = XtX[np.ix_(idx, idx)]
    Ainv = np.linalg.pinv(A)
    b = Ainv @ Xty[idx]
    e = y - X[:, idx] @ b
    meat = np.zeros((main_k, main_k))
    for c in range(len(dsets)):
        sel = g == c
        u = (X[sel][:, idx] * (w[sel] * e[sel])[:, None]).sum(0)
        meat += np.outer(u, u)
    nG = len(dsets)
    V = Ainv @ meat @ Ainv * nG / (nG - 1)
    se = np.sqrt(np.diag(V))
    tcrit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}.get(nG - 1, 2.0 if nG < 60 else 1.97)  # t(G-1), 97.5%
    eff = {cols[i]: (100 * b[i], 100 * (b[i] - tcrit * se[i]), 100 * (b[i] + tcrit * se[i])) for i in idx if not cols[i].startswith("ds:")}

    # interaction blocks: R^2 added to the main-effects model
    added = {}
    for n, cs in inter.items():
        added[n] = r2(XtX, Xty, yty, sw, swy, idx[1:] + [icol_idx[c] for c in cs]) - full
    # spread of dataset effects (SD of dataset fixed effects, points)
    dfx = [0.0] + [100 * b[cols.index(f"ds:{d}")] for d in dsets[1:]]
    return dict(n=len(rows), datasets=nG, runs=len({(r[1], json.dumps(r[3], sort_keys=True)) for r in rows}),
                r2=full, shapley=shap, effects=eff, interactions=added, dataset_sd=float(np.std(dfx, ddof=1)))


LABEL = {"oeq": "OEQ (vs.\\ MCQA)", "across": "MCQA across (vs.\\ within)", "nota10": "NOTA rate 10\\% (vs.\\ 30\\%)",
         "nota50": "NOTA rate 50\\% (vs.\\ 30\\%)", "noidk": "IDK removed", "order": "Randomized order and NOTA/IDK position",
         "seed1": "Generation seed 1 (vs.\\ 0)", "seed2": "Generation seed 2 (vs.\\ 0)"}


def table(res):
    cls, vqa = res["cls"], res["vqa"]
    L = ["\\begin{tabular}{lrrrr}", "\\toprule",
         " & \\multicolumn{2}{c}{\\textbf{Classification}} & \\multicolumn{2}{c}{\\textbf{VQA multiple choice}} \\\\",
         "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
         "\\textbf{Variable} & \\textbf{R$^2$ (pts)} & \\textbf{Share} & \\textbf{R$^2$ (pts)} & \\textbf{Share} \\\\", "\\midrule"]
    order = sorted(cls["shapley"], key=lambda n: -cls["shapley"][n])
    for n in order:
        a = cls["shapley"][n]
        v = vqa["shapley"].get(n)
        L.append(f"{n} & ${100 * a:.2f}$ & ${100 * a / cls['r2']:.1f}\\%$ & "
                 + (f"${100 * v:.2f}$ & ${100 * v / vqa['r2']:.1f}\\%$" if v is not None else "-- & --") + " \\\\")
    L += ["\\midrule", f"Main effects, total & ${100 * cls['r2']:.2f}$ & $100\\%$ & ${100 * vqa['r2']:.2f}$ & $100\\%$ \\\\"]
    for n in cls["interactions"]:
        v = vqa["interactions"].get(n)
        L.append(f"\\quad + {n} & $+{100 * cls['interactions'][n]:.2f}$ & & " + (f"$+{100 * v:.2f}$ &" if v is not None else "-- &") + " \\\\")
    L += ["\\midrule", "\\multicolumn{5}{l}{\\textit{Effects on accuracy (points, 95\\% CI clustered by dataset)}} \\\\"]
    for c in ("oeq", "across", "nota10", "nota50", "noidk", "order", "seed1", "seed2"):
        def cell(r):
            if c not in r["effects"]:
                return "\\multicolumn{2}{c}{--}"
            e, lo, hi = r["effects"][c]
            return f"\\multicolumn{{2}}{{c}}{{${e:+.1f}$ [${lo:+.1f}$, ${hi:+.1f}$]}}"
        L.append(f"{LABEL[c]} & {cell(cls)} & {cell(vqa)} \\\\")
    L.append(f"Dataset effects, SD across datasets & \\multicolumn{{2}}{{c}}{{${cls['dataset_sd']:.1f}$}} & \\multicolumn{{2}}{{c}}{{${vqa['dataset_sd']:.1f}$}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(L)


def main():
    res = {k: fit(k) for k in ("cls", "vqa")}
    for k, r in res.items():
        print(k, "n", r["n"], "datasets", r["datasets"], "runs", r["runs"], "R2", round(r["r2"], 4))
        for n, v in sorted(r["shapley"].items(), key=lambda t: -t[1]):
            print(f"   {n:45s} {100 * v:7.3f}  {100 * v / r['r2']:5.1f}%")
        for n, v in r["interactions"].items():
            print(f"   + {n:43s} {100 * v:7.3f}")
        for c, e in r["effects"].items():
            print(f"   {c:20s} {e[0]:+6.2f} [{e[1]:+6.2f}, {e[2]:+6.2f}]")
        print("   dataset SD", round(r["dataset_sd"], 2))
    open(os.path.join(P, "tables", "drivers.tex"), "w").write(table(res))
    json.dump(res, open(os.path.join(P, "drivers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
