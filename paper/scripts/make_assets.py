"""Build the paper's tables (tables/*.tex), figures (*.png) and numbers.json from the runs.

Run from the repo root:  python3 paper/scripts/make_assets.py
Inputs:  runs/rejudge_v4/exp/seeds/s*/<model>/   (experiment 1: generation seeds, default options; judge v4 verdicts)
         runs/exp/seeds/s*/<model>/compute.json   (generation and original judge time)
         reports/stats/seeds/stats.md  (agbench eval stats --runs runs/rejudge_v4/exp/seeds)
         reports/stats/exp24/stats.md  (agbench eval stats --runs runs/runpod_exp/runs --runs runs/rejudge_v4/exp/seeds/s0)
"""
import collections as C
import glob
import json
import os
import random
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from mdtables import sections  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Inputs (override to reproduce from your own runs; see paper Appendix D and paper/scripts/reproduce.sh):
#   AGBENCH_SEEDS  Experiment 1 run dirs, one subdir per seed   (default runs/rejudge_v4/exp/seeds)
#   AGBENCH_EXP24  Experiments 2-4 run dirs                       (default runs/runpod_exp/runs)
#   AGBENCH_STATS  agbench eval stats outputs (seeds/, exp24/)    (default reports/stats)
SEEDS = os.path.join(ROOT, os.environ.get("AGBENCH_SEEDS", "runs/rejudge_v4/exp/seeds"))
STATS = os.path.join(ROOT, os.environ.get("AGBENCH_STATS", "reports/stats"))
EXP24 = os.path.join(ROOT, os.environ.get("AGBENCH_EXP24", "runs/runpod_exp/runs"))
#   AGBENCH_BASELINES  naive baselines: prior/<run>, chance/s*/<run> (agbench eval baseline)   (default runs/baselines)
BASELINES = os.path.join(ROOT, os.environ.get("AGBENCH_BASELINES", "runs/baselines"))
#   AGBENCH_YOLO  few-shot YOLO11 runs: <shots>shot/s<subset>/ (agbench eval yolo-shared)   (default runs/yolo11s-shared)
YOLO = os.path.join(ROOT, os.environ.get("AGBENCH_YOLO", "runs/yolo11s-shared"))
YOLO_SHOTS = 10  # training images per class of the YOLO baseline in the main results
YOLO_LABEL = f"YOLO11s, {YOLO_SHOTS} images per class"
BASE_KINDS = [("chance", "Chance (random answer)")]  # the prior (best constant answer) runs exist but are not reported
PAPER = os.path.join(ROOT, "paper")
TAB = os.path.join(PAPER, "tables")
os.makedirs(TAB, exist_ok=True)

MODELS = [  # (run-dir prefix, short name, display name)
    ("Qwen__Qwen3.6-35B-A3B-FP8", "Qwen/Qwen3.6-35B-A3B-FP8", "Qwen3.6-35B-A3B"),
    ("Qwen__Qwen3.8-27B-FP8", "Qwen/Qwen3.8-27B-FP8", "Qwen3.8-27B"),
    ("RedHatAI__gemma-4-31B-it-FP8-dynamic", "RedHatAI/gemma-4-31B-it-FP8-dynamic", "Gemma-4-31B$^\\dagger$"),
    ("google__gemma-4-12B-it", "google/gemma-4-12B-it", "Gemma-4-12B"),
    ("RedHatAI__Muse-Glimmer-30B-FP8-block", "RedHatAI/Muse-Glimmer-30B-FP8-block", "Muse-Glimmer-30B"),
    ("nvidia__NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF", "Nemotron-3.5-Bolt-74B-A9B"),
    ("openbmb__MiniCPM-V-4.7-35B-A3B-FP8", "openbmb/MiniCPM-V-4.7-35B-A3B-FP8", "MiniCPM-V-4.7-35B-A3B"),
    ("mistralai__Ministral-3-14B-Instruct-2512", "mistralai/Ministral-3-14B-Instruct-2512", "Ministral-3-14B"),
]
PLAIN = {m[1]: re.sub(r"\$.*?\$", "", m[2]) for m in MODELS}
DISP = {m[1]: m[2] for m in MODELS}
COLS = [  # (key, header)
    ("mcqa_across", "MCQA across"),
    ("mcqa_within", "MCQA within"),
    ("oeq", "OEQ (judged)"),
    ("itt", "VQA"),
    ("detection", "Detection"),
    ("segmentation", "Segmentation"),
]
AG = [("disease", "Disease"), ("pest_damage", "Pest/Damage"), ("species_id", "Species"), ("quality_maturity", "Quality")]
NUM = {}
NOTA_CELLS = {}  # (model, format) -> measured change in accuracy at 10% and 50% NOTA rate, formatted


def seed_dirs():
    out = C.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(SEEDS, "s*/*/"))):
        if not os.path.exists(os.path.join(d, "metrics.json")):
            continue
        m = json.load(open(os.path.join(d, "metrics.json")))
        if m.get("pending_judge"):
            continue
        for pre, name, _ in MODELS:
            if os.path.basename(d.rstrip("/")).startswith(pre + "_"):
                out[name].append(d)
    return out


def item_key(p):
    if p["task"] == "classification":
        return p["format"]
    if p["task"] == "image_text_to_text":
        return "itt"
    return p["task"]


def score(p):
    if p.get("needs_judge"):
        return 1.0 if p.get("judge_correct") else 0.0
    c = p.get("correct")
    if isinstance(c, bool):
        return 1.0 if c else 0.0
    return float(p.get("acc", c) or 0.0)


def load_per_dataset(run):
    """{(col, dataset): [scores]}, plus ag-task, ITT-format and fuzzy-vs-judge splits."""
    per = C.defaultdict(list)
    ag = C.defaultdict(list)
    itt = C.defaultdict(list)
    fuzzy = C.defaultdict(list)
    rouge = []
    for line in open(os.path.join(run, "predictions.jsonl")):
        p = json.loads(line)
        k = item_key(p)
        s = score(p)
        per[(k, p["dataset"])].append(s)
        if p["task"] == "classification":
            ag[(k, p["ag_task"], p["dataset"])].append(s)
            if k == "oeq":
                fuzzy[(p["ag_task"], p["dataset"])].append((1.0 if p.get("correct") else 0.0, s))
        if p["task"] == "image_text_to_text":
            itt[p["qa_format"]].append(s)
            if p["qa_format"] == "open_ended" and p.get("rouge_l") is not None:
                rouge.append(float(p["rouge_l"]))
    return per, ag, itt, fuzzy, rouge


def baselines():
    """Naive baselines, aggregated like the models: per dataset mean over the chance seeds, then macro over datasets;
    VQA formats pooled over questions. {kind: {col: %, "itt_fmt": {qa_format: %}}}"""
    out = {}
    for kind, _ in BASE_KINDS:
        runs = [load_per_dataset(d) for d in sorted(glob.glob(os.path.join(BASELINES, kind, "**", "predictions.jsonl"), recursive=True))
                for d in [os.path.dirname(d)]]
        if not runs:
            continue
        rec = {}
        for key, _ in COLS:
            dsets = sorted({d for (k, d) in runs[0][0] if k == key})
            rec[key] = mean([100 * mean([mean(r[0][(key, d)]) for r in runs]) for d in dsets])
        rec["itt_fmt"] = {q: mean([100 * mean(r[2][q]) for r in runs]) for q in runs[0][2]}
        rec["runs"] = len(runs)
        out[kind] = rec
    NUM["baselines"] = {k: {c: (round(v, 2) if not isinstance(v, dict) else {q: round(x, 2) for q, x in v.items()}) for c, v in r.items()}
                        for k, r in out.items()}
    return out


def yolo_baseline(shots=YOLO_SHOTS):
    """Few-shot YOLO11, scored as in Ranario & Earles (2025): it sees only the image. Classification is top-1 accuracy, the
    top class among the dataset's classes against the true class (one value per image, whatever the question format;
    it is the score of the "oeq" items, which hold exactly that prediction). Detection and segmentation are scored like
    the VLMs. Aggregated like the models, with the training subsets in place of the seeds: per dataset mean over
    subsets, then macro over datasets, 95% bootstrap CI over datasets.
    {"top1" | "detection" | "segmentation": (mean, lo, hi, n), "ag": {ag task: (mean, lo, hi, n)}, "per_subset": {...}}"""
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(YOLO, f"{shots}shot", "s*", "predictions.jsonl")))
    if not dirs:
        return {}
    runs = [load_per_dataset(d) for d in dirs]
    rec = {"per_subset": {}, "ag": {}}
    for key, col in (("top1", "oeq"), ("detection", "detection"), ("segmentation", "segmentation")):
        dsets = sorted({d for (k, d) in runs[0][0] if k == col})
        vals = [100 * mean([mean(r[0][(col, d)]) for r in runs]) for d in dsets]
        rec[key] = (mean(vals), *boot_ci(vals), len(dsets))
        rec["per_subset"][key] = [100 * mean([mean(r[0][(col, d)]) for d in dsets]) for r in runs]
    for a, _ in AG:
        dsets = sorted({d for (k, ag, d) in runs[0][1] if k == "oeq" and ag == a})
        vals = [100 * mean([mean(r[1][("oeq", a, d)]) for r in runs]) for d in dsets]
        rec["ag"][a] = (mean(vals), *boot_ci(vals), len(dsets))
    rec["subsets"] = len(dirs)
    if shots != YOLO_SHOTS:
        return rec
    train = C.defaultdict(list)  # task -> [(training images, merged classes)] from the runs' compute records
    for d in dirs:
        for c in json.load(open(os.path.join(d, "compute.json"))):
            if c.get("phase") == "train":
                train[c["task"]].append((c["train_images"], c["classes"]))
    # rounded as in the text ("about 18,250"): to 50 images, or 10 below 1,000
    NUM["yolo_train"] = {t: {"images": round(mean([x[0] for x in v]), 1), "classes": v[0][1],
                             "images_approx": round(mean([x[0] for x in v]) / (50 if mean([x[0] for x in v]) >= 1000 else 10)) * (50 if mean([x[0] for x in v]) >= 1000 else 10)}
                         for t, v in train.items()}
    NUM["yolo"] = {"shots": shots, "subsets": len(dirs),
                   **{k: [round(x, 2) for x in rec[k][:3]] + [rec[k][3]] for k in ("top1", "detection", "segmentation")},
                   "ag": {a: [round(x, 2) for x in v[:3]] + [v[3]] for a, v in rec["ag"].items()}}
    return rec


def boot_ci(vals, B=10000, seed=0):
    rnd = random.Random(seed)
    n = len(vals)
    ms = sorted(sum(rnd.choice(vals) for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def collect():
    dirs = seed_dirs()
    data = {}
    for _, name, _ in MODELS:
        runs = dirs[name]
        per_seed = [load_per_dataset(r) for r in runs]
        data[name] = {"seeds": [int(re.search(r"/s(\d+)/", r).group(1)) for r in runs], "runs": per_seed}
    return data


def answer_flips():
    """MCQA answers across the three generation seeds, per model and format: accuracy on items with the same answer in every
    seed vs items whose answer changes, and the items that change from correct to incorrect (and back) between seeds 0 and 1.
    Shows that answers flip on uncertain items and in both directions, so the flips cancel in the accuracy."""
    out = {}
    for _, name, _ in MODELS:
        runs = []
        for d in sorted(seed_dirs()[name], key=lambda d: int(re.search(r"/s(\d+)/", d).group(1))):
            runs.append({p["item_id"]: p for p in map(json.loads, open(os.path.join(d, "predictions.jsonl")))})
        for f in ("mcqa_within", "mcqa_across"):
            ids = [i for i, p in runs[0].items() if p["task"] == "classification" and p["format"] == f and all(i in r for r in runs)]
            st, fl, cw, wc = [], [], 0, 0
            for i in ids:
                cor = [bool(r[i]["correct"]) for r in runs]
                if len({r[i].get("pred_label") for r in runs}) == 1:
                    st.append(cor[0])
                else:
                    fl += cor
                cw += cor[0] and not cor[1]
                wc += cor[1] and not cor[0]
            out[(PLAIN[name], f)] = {"items": len(ids), "acc_stable": 100 * mean(st), "acc_flipping": 100 * mean(fl),
                                     "correct_to_wrong": cw, "wrong_to_correct": wc}
    v = list(out.values())
    rng = lambda k: [round(min(x[k] for x in v), 1), round(max(x[k] for x in v), 1)]  # noqa: E731
    NUM["flips"] = {"acc_stable": rng("acc_stable"), "acc_flipping": rng("acc_flipping"), "correct_to_wrong": rng("correct_to_wrong"),
                    "wrong_to_correct": rng("wrong_to_correct"), "items_per_format": sorted({x["items"] for x in v}),
                    "per_model": {f"{m} {f}": {k: round(x, 1) for k, x in r.items()} for (m, f), r in out.items()}}
    return out


# Judge validation against human labels (Appendix A.5). Counts supplied by the annotator from their labelled subsample of
# Gemma-4-31B's OEQ answers in an earlier run; the label file itself is not in this repository.
#   disagreement set: answers on which prompts v1-v3 disagreed (308, of which 282 labelled); agreement sample: 150 answers
#   on which v1 and v2 agreed, half credited and half rejected. Cells: answers where the judge matches the human label.
JUDGE_VAL = {
    "earlier_run_answers": 8324, "disagreement_total": 308,
    "disagreement": {  # task: (answers, v1, v2, v3, v4)
        "Disease": (65, 31, 35, 37, 44), "Pest/damage": (50, 7, 41, 42, 41),
        "Species": (81, 58, 19, 61, 62), "Quality/maturity": (86, 42, 6, 70, 77)},
    "agreement": {"answers": 150, "match": (136, 136, 141, 142), "false_credit": (13, 13, 8, 7), "false_reject": (1, 1, 1, 1)},
}


def judge_validation():
    """Table of judge agreement with the human labels, by prompt version (tables/judge_validation.tex)."""
    d, a = JUDGE_VAL["disagreement"], JUDGE_VAL["agreement"]
    tot = [sum(v[i] for v in d.values()) for i in range(5)]
    assert all(a["answers"] - m == fc + fr for m, fc, fr in zip(a["match"], a["false_credit"], a["false_reject"]))
    cell = lambda k, n: f"${k}$ (${100 * k / n:.0f}$)"  # noqa: E731
    rows = [f"Disagreement set & {t} & ${v[0]}$ & " if i == 0 else f" & {t} & ${v[0]}$ & " for i, (t, v) in enumerate(d.items())]
    rows = [r + " & ".join(cell(k, v[0]) for k in v[1:]) + " \\\\" for r, v in zip(rows, d.values())]
    rows.append(" & \\textbf{All} & $\\mathbf{" + str(tot[0]) + "}$ & " + " & ".join(f"$\\mathbf{{{k}}}$ ($\\mathbf{{{100 * k / tot[0]:.0f}}}$)" for k in tot[1:]) + " \\\\")
    rows.append("\\midrule")
    rows.append(f"Agreement sample & Matches human label & ${a['answers']}$ & " + " & ".join(cell(k, a["answers"]) for k in a["match"]) + " \\\\")
    rows.append(" & False credits & & " + " & ".join(f"${k}$" for k in a["false_credit"]) + " \\\\")
    rows.append(" & False rejections & & " + " & ".join(f"${k}$" for k in a["false_reject"]) + " \\\\")
    tex = ("\\begin{tabular}{llrrrrr}\n\\toprule\n\\textbf{Set} & \\textbf{Task} & \\textbf{Answers} & \\textbf{v1} & \\textbf{v2} & \\textbf{v3} & \\textbf{v4 (final)} \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(TAB, "judge_validation.tex"), "w").write(tex)
    pct = lambda k, n: round(100 * k / n, 2)  # noqa: E731
    NUM["judge_validation"] = {
        "earlier_run_answers": JUDGE_VAL["earlier_run_answers"], "disagreement_total": JUDGE_VAL["disagreement_total"],
        "disagreement_labelled": tot[0], "disagreement_match": tot[1:], "disagreement_pct": [pct(k, tot[0]) for k in tot[1:]],
        "by_task": {t: {"answers": v[0], "match": list(v[1:]), "pct": [pct(k, v[0]) for k in v[1:]]} for t, v in d.items()},
        "agreement_answers": a["answers"], "agreement_match": list(a["match"]), "agreement_pct": [pct(k, a["answers"]) for k in a["match"]],
        "false_credit": list(a["false_credit"]), "false_reject": list(a["false_reject"]),
        "agreement_half": a["answers"] // 2}


def nota_parts():
    """Seed-0 MCQA items split by whether the true class is listed. Per (model, format, dataset): items, withheld items,
    correct answers on listed items (listed accuracy) and NOTA answers on withheld items (NOTA recall)."""
    out = {}
    for _, name, _ in MODELS:
        run = [d for d in seed_dirs()[name] if "/s0/" in d][0]
        cnt = C.defaultdict(lambda: [0, 0, 0, 0])  # n, withheld, listed correct, withheld correct
        for line in open(os.path.join(run, "predictions.jsonl")):
            p = json.loads(line)
            if p["task"] != "classification" or p["format"] == "oeq":
                continue
            c = cnt[(p["format"], p["dataset"])]
            c[0] += 1
            if p["is_nota"]:
                c[1] += 1; c[3] += bool(p["correct"])
            else:
                c[2] += bool(p["correct"])
        out[name] = dict(cnt)
    return out


def fmt(x, d=1):
    return f"{x:.{d}f}"


def main_table(data):
    """Accuracy (%) per CV task: macro over datasets of the seed-mean per-dataset accuracy, 95% bootstrap CI over datasets."""
    res = {}
    for _, name, _ in MODELS:
        runs = data[name]["runs"]
        res[name] = {}
        for key, _ in COLS:
            dsets = sorted({d for (k, d) in runs[0][0] if k == key})
            vals = [100 * mean([mean(r[0][(key, d)]) for r in runs]) for d in dsets]
            lo, hi = boot_ci(vals)
            res[name][key] = (mean(vals), lo, hi, len(dsets))
    NUM["main"] = {PLAIN[n]: {k: round(v[0], 2) for k, v in r.items()} for n, r in res.items()}
    best = {k: max(res[n][k][0] for n in res) for k, _ in COLS}
    ndat = {k: res[MODELS[0][1]][k][3] for k, _ in COLS}
    lines = []
    for _, name, _ in MODELS:
        cells = []
        for key, _ in COLS:
            m, lo, hi, _ = res[name][key]
            v = fmt(m)
            if abs(m - best[key]) < 1e-9:
                v = f"\\mathbf{{{v}}}"
            cells.append(f"\\makecell{{${v}$\\\\{{[{fmt(lo)}, {fmt(hi)}]}}}}")
        assert len(data[name]["seeds"]) == 3, name  # captions say 3 seeds
        lines.append(f"{DISP[name]} & " + " & ".join(cells) + " \\\\")
    if BASE:
        lines.append("\\midrule")
        lines += [f"\\textit{{{label}}} & " + " & ".join(f"${fmt(BASE[kind][k])}$" for k, _ in COLS) + " \\\\"
                  for kind, label in BASE_KINDS if kind in BASE]
    if YB:  # top-1 accuracy is one value for all three classification formats
        cell = lambda v: f"\\makecell{{${fmt(v[0])}$\\\\{{[{fmt(v[1])}, {fmt(v[2])}]}}}}"  # noqa: E731
        lines.append(f"\\textit{{{YOLO_LABEL}}} & \\multicolumn{{3}}{{c}}{{\\makecell{{top-1 ${fmt(YB['top1'][0])}$\\\\{{[{fmt(YB['top1'][1])}, {fmt(YB['top1'][2])}]}}}}}} & --- & "
                     + f"{cell(YB['detection'])} & {cell(YB['segmentation'])} \\\\")
    hdr = " & ".join(f"\\makecell{{\\textbf{{{h}}}\\\\($n={ndat[k]}$)}}" for k, h in COLS)
    tex = (
        "\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{l" + "c" * len(COLS) + "}\n\\toprule\n"
        f"\\textbf{{Model}} & {hdr} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(TAB, "main_results.tex"), "w").write(tex)
    return res


SHORT_AG = {"Disease": "Disease", "Pest/Damage": "Pest", "Species": "Species", "Quality": "Quality"}
AG_CI = {}  # (model, format, ag task) -> 95% bootstrap CI over datasets (%), for the figure


def agtask_table(data):
    res = {}
    fmts = [("mcqa_across", "MCQA across"), ("mcqa_within", "MCQA within"), ("oeq", "OEQ (judged)")]
    ndat = {}
    for _, name, _ in MODELS:
        runs = data[name]["runs"]
        res[name] = {}
        for f, _ in fmts:
            for a, _ in AG:
                dsets = sorted({d for (k, ag, d) in runs[0][1] if k == f and ag == a})
                ndat[a] = len(dsets)
                vals = [100 * mean([mean(r[1][(f, a, d)]) for r in runs]) for d in dsets]
                res[name][(f, a)] = mean(vals)
                AG_CI[(name, f, a)] = boot_ci(vals)
    NUM["agtask"] = {PLAIN[n]: {f"{f}/{a}": round(v, 2) for (f, a), v in r.items()} for n, r in res.items()}
    best = {fa: max(res[n][fa] for n in res) for fa in res[MODELS[0][1]]}
    lines = []
    for _, name, _ in MODELS:
        cells = []
        for f, _ in fmts:
            for a, _ in AG:
                v = res[name][(f, a)]
                s = fmt(v)
                cells.append(f"$\\mathbf{{{s}}}$" if abs(v - best[(f, a)]) < 1e-9 else f"${s}$")
        lines.append(f"{DISP[name]} & " + " & ".join(cells) + " \\\\")
    mean_row = []
    for f, _ in fmts:
        for a, _ in AG:
            mean_row.append(f"${fmt(mean([res[n][(f, a)] for n in res]))}$")
    NUM["agtask_mean"] = {f"{f}/{a}": round(mean([res[n][(f, a)] for n in res]), 2) for f, _ in fmts for a, _ in AG}
    sub = " & ".join(" & ".join(f"\\makecell{{{SHORT_AG[h]}\\\\({ndat[a]})}}" for a, h in AG) for _ in fmts)
    top = " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{h}}}}}" for _, h in fmts)
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(fmts)))
    tex = (
        "\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{l" + "c" * 12 + "}\n\\toprule\n"
        f"\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & {top} \\\\\n{cmid}\n & {sub} \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\midrule\nMean of models & "
        + " & ".join(mean_row)
        + " \\\\\n"
        + ((f"\\textit{{YOLO11s top-1}} & " + " & ".join(f"${fmt(YB['ag'][a][0])}$" for _ in fmts for a, _ in AG) + " \\\\\n") if YB else "")
        + "\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(TAB, "agtask.tex"), "w").write(tex)
    return res


def oeq_fuzzy_by_ag(data):
    """OEQ fuzzy-match accuracy per ag task, aggregated like agtask_table (seed-mean per dataset, macro)."""
    res = {}
    for _, name, _ in MODELS:
        runs = data[name]["runs"]
        res[name] = {}
        for a, _ in AG:
            dsets = sorted({d for (ag, d) in runs[0][3] if ag == a})
            res[name][a] = mean([100 * mean([mean([x[0] for x in r[3][(a, d)]]) for r in runs]) for d in dsets])
    NUM["agtask_fuzzy"] = {PLAIN[n]: {a: round(v, 2) for a, v in r.items()} for n, r in res.items()}
    return res


def judge_table(data):
    """OEQ, seed 0: fuzzy string match vs judge verdict, macro over datasets per ag task."""
    lines = []
    NUM["judge"] = {}
    for _, name, _ in MODELS:
        r = data[name]["runs"][0][3]
        cells = []
        rec = {}
        for a, _ in [("all", "All")] + AG:
            keys = [k for k in r if a == "all" or k[0] == a]
            fz = mean([100 * mean([x[0] for x in r[k]]) for k in keys])
            jd = mean([100 * mean([x[1] for x in r[k]]) for k in keys])
            agree = 100 * mean([x[0] == x[1] for k in keys for x in r[k]])
            rec[a] = (round(fz, 2), round(jd, 2), round(agree, 2))
            cells.append(f"${fmt(fz)}$ & ${fmt(jd)}$" + (f" & ${'+' if jd - fz >= 0 else '-'}{fmt(abs(jd - fz))}$" if a == "all" else ""))
        NUM["judge"][PLAIN[name]] = rec
        lines.append(f"{DISP[name]} & " + " & ".join(cells) + " \\\\")
    groups = [("All", 0)] + [(h, 0) for _, h in AG]
    top = " & ".join(f"\\multicolumn{{{3 if i == 0 else 2}}}{{c}}{{\\textbf{{{h}}}}}" for i, (h, _) in enumerate(groups))
    cmid = "\\cmidrule(lr){2-4} " + " ".join(f"\\cmidrule(lr){{{5 + 2 * i}-{6 + 2 * i}}}" for i in range(len(groups) - 1))
    sub = "Fuzzy & Judge & $\\Delta$ & " + " & ".join("Fuzzy & Judge" for _ in groups[1:])
    tex = (
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l" + "c" * (1 + 2 * len(groups)) + "}\n\\toprule\n"
        f"\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & {top} \\\\\n{cmid}\n & {sub} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(TAB, "judge.tex"), "w").write(tex)


def itt_table(data):
    order = [("multiple_choice", "Multiple choice"), ("yes_no", "Yes/no"), ("numeric", "Numeric"), ("open_ended", "Open-ended (judged)")]
    lines = []
    NUM["itt"] = {}
    counts = {}
    for _, name, _ in MODELS:
        runs = data[name]["runs"]
        cells = []
        rec = {}
        for q, _ in order:
            counts[q] = len(runs[0][2][q])
            v = mean([100 * mean(r[2][q]) for r in runs])
            rec[q] = round(v, 2)
            cells.append(v)
        rl = mean([100 * mean(r[4]) for r in runs])
        rec["rouge_l"] = round(rl, 2)
        NUM["itt"][PLAIN[name]] = rec
        lines.append((name, cells, rl))
    best = [max(l[1][i] for l in lines) for i in range(len(order))]
    best_rl = max(l[2] for l in lines)
    out = []
    for name, cells, rl in lines:
        cs = [f"$\\mathbf{{{fmt(v)}}}$" if abs(v - best[i]) < 1e-9 else f"${fmt(v)}$" for i, v in enumerate(cells)]
        r = f"$\\mathbf{{{fmt(rl)}}}$" if abs(rl - best_rl) < 1e-9 else f"${fmt(rl)}$"
        out.append(f"{DISP[name]} & " + " & ".join(cs) + f" & {r} \\\\")
    if BASE:
        out.append("\\midrule")
        out += [f"\\textit{{{label}}} & " + " & ".join("---" if q == "open_ended" else f"${fmt(BASE[kind]["itt_fmt"][q])}$" for q, _ in order)
                + " & --- \\\\" for kind, label in BASE_KINDS if kind in BASE]
    hdr = " & ".join(f"\\textbf{{{h}}}" for q, h in order)
    ns = " & ".join(f"$N={counts[q]:,}$".replace(",", "{,}") for q, _ in order)
    tex = (
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        f"\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & {hdr} & \\multirow{{2}}{{*}}{{\\textbf{{ROUGE-L}}}} \\\\\n"
        f" & {ns} & \\\\\n\\midrule\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(TAB, "itt.tex"), "w").write(tex)


def stats_tables():
    s1 = sections(os.path.join(STATS, "seeds/stats.md"))
    s24 = sections(os.path.join(STATS, "exp24/stats.md"))
    num = lambda x: float(re.match(r"[-+]?\d+(\.\d+)?", x.strip()).group(0))  # noqa: E731
    ci = lambda x: re.search(r"\[(.*?)\]", x).group(1)  # noqa: E731

    # Seeds: sd across seeds and agreement, per format.
    fm = [("mcqa_across", "MCQA across"), ("mcqa_within", "MCQA within"), ("oeq", "OEQ"), ("itt_mc", "VQA MC"), ("itt_other", "VQA other"), ("detection", "Det."), ("segmentation", "Seg.")]
    seeds = {(r["model"], r["format"]): r for r in s1["Seeds (default options)"]}
    lines = []
    NUM["seeds"] = {}
    for _, name, _ in MODELS:
        cells = []
        rec = {}
        for f, _ in fm:
            r = seeds[(name, f)]
            rec[f] = (float(r["sd"]), float(r["same answer in every seed %"]), r["seeds"])
            cells.append(f"${float(r['sd']):.2f}$ / ${float(r['same answer in every seed %']):.0f}$")
        NUM["seeds"][PLAIN[name]] = rec
        assert len(json.loads(seeds[(name, 'mcqa_within')]["seeds"])) == 3, name  # caption says 3 seeds
        lines.append(f"{DISP[name]} & " + " & ".join(cells) + " \\\\")
    hdr = " & ".join(f"\\textbf{{{h}}}" for _, h in fm)
    sub = " & ".join("SD / agr." for _ in fm)
    tex = "\\begin{tabular}{l" + "c" * len(fm) + "}\n\\toprule\n" f"\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & {hdr} \\\\\n & {sub} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, "seeds.tex"), "w").write(tex)

    # Behaviour at the default options (seed 0): NOTA recall, over-commit, false NOTA, IDK rate.
    beh = {}
    for r in s1["Behaviour (macro over datasets, 95% CI; NOTA/IDK metrics for multiple choice)"]:
        if r["ag_task"] == "all" and "gen_seed=0" in r["variant"]:
            beh[(r["model"], r["format"])] = r
    NUM["behaviour"] = {}
    for _, name, _ in MODELS:
        NUM["behaviour"][PLAIN[name]] = {f: [num(beh[(name, f)][k]) for k in ("accuracy", "nota_recall", "over_commit", "false_nota", "idk_rate")]
                                         for f in ("mcqa_within", "mcqa_across")}

    # Do models that answer NOTA more often also reject listed correct classes more often? Spearman over models.
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = k + 1.0
        return r
    NUM["recall_vs_false_nota_spearman"] = {}
    for f in ("mcqa_within", "mcqa_across"):
        a, b = ranks([NUM["behaviour"][PLAIN[n]][f][1] for _, n, _ in MODELS]), ranks([NUM["behaviour"][PLAIN[n]][f][3] for _, n, _ in MODELS])
        ma, mb = mean(a), mean(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        NUM["recall_vs_false_nota_spearman"][f] = round(cov / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5, 3)

    # Within vs across on the same images.
    wa = {(r["model"], r["metric"]): r for r in s1["Within vs across distractors (same images)"] if "gen_seed=0" in r["variant"]}
    NUM["within_minus_across"] = {PLAIN[n]: {m: (float(wa[(n, m)]["within − across (pts)"]), ci(wa[(n, m)]["95% CI"]) if "[" in wa[(n, m)]["95% CI"] else wa[(n, m)]["95% CI"]) for m in ("accuracy", "over_commit", "false_nota")} for _, n, _ in MODELS}

    # Model pairs (seed 0), Holm-corrected.
    NUM["pairs"] = {}
    for k, rows in s1.items():
        if k.startswith("Model pairs"):
            f = re.search(r"accuracy, (\w+)", k).group(1)
            NUM["pairs"][f] = [(PLAIN[r["a"]], PLAIN[r["b"]], float(r["a − b (pts)"]), r["95% CI"], r["p (bootstrap, Holm)"]) for r in rows]

    # Position preference: share of answers at each position vs share of correct answers there.
    pos = C.defaultdict(dict)
    for r in s24["Position"]:
        pos[(r["model"], r["format"])][int(r["position"])] = (float(r["chosen %"]), float(r["correct answer here %"]))
    NUM["position"] = {f"{PLAIN[k[0]]}/{k[1]}": v for k, v in pos.items()}

    # Option order (experiment 2).
    oo = {(r["model"], r["format"]): r for r in s24["Option order"]}
    lines = []
    NUM["order"] = {}
    m7 = [m for m in MODELS if (m[1], "mcqa_within") in oo]
    for _, name, _ in m7:
        cells = []
        rec = {}
        for f in ("mcqa_within", "mcqa_across", "itt_mc"):
            r = oo[(name, f)]
            # largest gap between chosen % and correct-answer % over positions (points), and that position
            gaps = {k: c - g for k, (c, g) in pos[(name, f)].items()}
            at = max(gaps, key=lambda k: abs(gaps[k]))
            rec[f] = (float(r["accuracy mean"]), float(r["sd across orders"]), float(r["same answer in every order %"]), round(gaps[at], 1), at)
            ps = f"${gaps[at]:+.1f}$ ({at})"
            cells += [f"${float(r['accuracy mean']):.1f}$", f"${float(r['sd across orders']):.2f}$", f"${float(r['same answer in every order %']):.0f}$", ps]
        NUM["order"][PLAIN[name]] = rec
        lines.append(f"{DISP[name]} & " + " & ".join(cells) + " \\\\")
    sub = " & ".join("\\makecell{Acc.\\\\(\\%)} & \\makecell{SD\\\\(pts)} & \\makecell{Same\\\\(\\%)} & \\makecell{Pos.\\\\(pts)}" for _ in range(3))
    tex = (
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l" + "c" * 12 + "}\n\\toprule\n"
        "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{4}{c}{\\textbf{MCQA within}} & \\multicolumn{4}{c}{\\textbf{MCQA across}} & \\multicolumn{4}{c}{\\textbf{VQA multiple choice}} \\\\\n"
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13}\n & " + sub + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(TAB, "order.tex"), "w").write(tex)

    # NOTA rate (experiment 3) and IDK removal (experiment 4), paired vs the seed-0 run.
    nr = C.defaultdict(dict)
    for r in s24["NOTA rate (paired vs the default-rate run)"]:
        nr[(r["model"], r["format"])][r["nota_rate"]] = r
    idk = C.defaultdict(dict)
    for r in s24["IDK removed (paired vs the run with IDK)"]:
        idk[(r["model"], r["format"])][r["idk"]] = r
    lines = []
    NUM["nota_rate"] = {}
    NUM["idk"] = {}

    def d(x):
        if x in ("nan", "", "—"):
            return "---"
        v = num(x)
        p = float(re.search(r"p=([\d.e-]+)", x).group(1))
        star = "^{*}" if p < 0.05 else ""
        return f"${'+' if v >= 0 else ''}{v:.1f}{star}$"

    for _, name, _ in m7:
        cells = []
        recn, reci = {}, {}
        for f in ("mcqa_within", "mcqa_across"):
            row = nr[(name, f)]
            recn[f] = {k: (row[k]["accuracy"], row[k]["Δaccuracy vs ref"], row[k]["nota_recall"], row[k]["Δnota_recall vs ref"]) for k in row}
            NOTA_CELLS[(name, f)] = (d(row["0.1"]["Δaccuracy vs ref"]), d(row["0.5"]["Δaccuracy vs ref"]))
        for f in ("mcqa_within", "mcqa_across", "itt_mc"):
            row = idk[(name, f)]
            on = [r for k, r in row.items() if r["Δaccuracy vs ref"] in ("nan",)][0]
            off = [r for k, r in row.items() if r["Δaccuracy vs ref"] not in ("nan",)][0]
            reci[f] = (on["idk_rate"], off["Δaccuracy vs ref"], off["Δnota_recall vs ref"], off["Δfalse_nota vs ref"])
            cells += [f"${num(on['idk_rate']):.1f}$", d(off["Δaccuracy vs ref"])]
        NUM["nota_rate"][PLAIN[name]] = recn
        NUM["idk"][PLAIN[name]] = reci
        lines.append(f"{DISP[name]} & " + " & ".join(cells) + " \\\\")
    tex = (
        "\\begin{tabular}{l" + "c" * 6 + "}\n\\toprule\n"
        "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{2}{c}{\\textbf{MCQA within}} & \\multicolumn{2}{c}{\\textbf{MCQA across}} & \\multicolumn{2}{c}{\\textbf{VQA MC}} \\\\\n"
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
        " & " + " & ".join(["\\makecell{IDK\\\\(\\%)} & \\makecell{$\\Delta$Acc.\\\\(pts)}"] * 3) + " \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(TAB, "idk.tex"), "w").write(tex)

    # NOTA rate as a reweighting. Per dataset, MCQA accuracy = r_d * NOTA recall_d + (1 - r_d) * listed accuracy_d, where
    # r_d is the dataset's share of withheld items and listed accuracy is the accuracy on items whose true class is listed
    # (any other answer, including NOTA and IDK, is wrong). Holding both parts at their seed-0 values, predict the macro
    # accuracy at a 10% and 50% rate and compare with the measured change in Experiment 3.
    parts = nota_parts()
    mix = {}
    for _, name, _ in m7:
        for f in ("mcqa_within", "mcqa_across"):
            ds = {d: c for (ff, d), c in parts[name].items() if ff == f}
            listed = mean([100 * c[2] / (c[0] - c[1]) for c in ds.values() if c[0] > c[1]])
            now = mean([100 * (c[2] + c[3]) / c[0] for c in ds.values()])
            def at(r):
                return mean([100 * (r * c[3] / c[1] + (1 - r) * c[2] / (c[0] - c[1])) if c[1] and c[0] > c[1] else 100 * (c[2] + c[3]) / c[0]
                             for c in ds.values()])
            rec = NUM["behaviour"][PLAIN[name]][f][1]
            obs = {r: num(nr[(name, f)][r]["\u0394accuracy vs ref"]) for r in ("0.1", "0.5")}
            pred = {r: at(float(r)) - now for r in ("0.1", "0.5")}
            mix[f"{PLAIN[name]}/{f}"] = {"recall": round(rec, 2), "listed": round(listed, 2), "gap": round(rec - listed, 2),
                                         **{f"pred_{r}": round(pred[r], 2) for r in pred}, **{f"obs_{r}": obs[r] for r in obs}}
    err = [abs(v[f"pred_{r}"] - v[f"obs_{r}"]) for v in mix.values() for r in ("0.1", "0.5")]
    NUM["nota_mix"] = {"rows": mix, "max_abs_error": round(max(err), 2), "mean_abs_error": round(mean(err), 2),
                       "listed_range": (round(min(v["listed"] for v in mix.values()), 1), round(max(v["listed"] for v in mix.values()), 1))}
    NUM["listed_accuracy"] = {PLAIN[n]: {f: round(mean([100 * c[2] / (c[0] - c[1]) for (ff, d), c in parts[n].items() if ff == f and c[0] > c[1]]), 2)
                                          for f in ("mcqa_within", "mcqa_across")} for _, n, _ in MODELS}
    behaviour_table()

    # Table 5: NOTA rate (Experiment 3). The two parts of accuracy at 30% and the measured change at 10% and 50%,
    # with the change predicted from the two parts in parentheses.
    lines = []
    for _, name, _ in m7:
        cells = []
        for f in ("mcqa_within", "mcqa_across"):
            v = mix[f"{PLAIN[name]}/{f}"]
            o10, o50 = NOTA_CELLS[(name, f)]
            cells += [f"${v['recall']:.1f}$", f"${v['listed']:.1f}$",
                      f"{o10} (${v['pred_0.1']:+.1f}$)", f"{o50} (${v['pred_0.5']:+.1f}$)"]
        lines.append(f"{DISP[name]} & " + " & ".join(cells) + " \\\\")
    sub = " & ".join(["\\makecell{NOTA\\\\recall} & \\makecell{Listed\\\\acc.} & \\makecell{$\\Delta$ at 10\\%\\\\(pred.)} & \\makecell{$\\Delta$ at 50\\%\\\\(pred.)}"] * 2)
    tex = (
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l" + "c" * 8 + "}\n\\toprule\n"
        "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{4}{c}{\\textbf{MCQA within}} & \\multicolumn{4}{c}{\\textbf{MCQA across}} \\\\\n"
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n & " + sub + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(TAB, "nota.tex"), "w").write(tex)


def behaviour_table():
    """Table B3: per format, NOTA recall and over-commit (withheld-class items), listed accuracy and false NOTA
    (listed-class items), IDK rate (all items). Seed 0."""
    lines = []
    for _, name, _ in MODELS:
        cells = []
        for f in ("mcqa_within", "mcqa_across"):
            acc, rec, oc, fn, idk = NUM["behaviour"][PLAIN[name]][f]
            cells += [f"${v:.1f}$" for v in (rec, oc, NUM["listed_accuracy"][PLAIN[name]][f], fn, idk)]
        lines.append(f"{DISP[name]} & " + " & ".join(cells) + " \\\\")
    sub = " & ".join("\\makecell{NOTA\\\\rec.} & \\makecell{Over-\\\\commit} & \\makecell{Listed\\\\acc.} & \\makecell{False\\\\NOTA} & IDK" for _ in range(2))
    tex = (
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l" + "c" * 10 + "}\n\\toprule\n"
        "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{5}{c}{\\textbf{MCQA within}} & \\multicolumn{5}{c}{\\textbf{MCQA across}} \\\\\n"
        "\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}\n & " + sub + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(TAB, "behaviour.tex"), "w").write(tex)


def datasets_list():
    """Appendix: every dataset per task with its class and image counts (and VQA question counts), from the items of
    one Experiment 1 run (every model sees the same items). Written as a two-column list, tables/datasets.tex."""
    run = sorted(glob.glob(os.path.join(SEEDS, "s0/*/items.jsonl")))[0]
    info = C.defaultdict(lambda: {"classes": set(), "images": set(), "items": 0})
    for line in open(run):
        it = json.loads(line)
        d = info[(it["task"], it["dataset"])]
        d["items"] += 1
        d["images"].update(it.get("images") or [])
        if it["task"] != "image_text_to_text" and (it["task"] != "classification" or it["format"] == "oeq"):
            d["classes"].update(it.get("class_names") or [])
    esc = lambda n: n.replace("_", "\\_\\allowbreak{}")  # noqa: E731
    out = []
    for task, title, unit in (("classification", "Classification", "classes, images"), ("detection", "Detection", "classes, images"),
                              ("segmentation", "Segmentation", "classes, images"), ("image_text_to_text", "Visual question answering", "questions, images")):
        names = sorted((n for t, n in info if t == task), key=str.lower)
        out.append(f"\\subsection*{{{title} ({len(names)} datasets; {unit})}}\n\\begin{{multicols}}{{2}}\\raggedright")
        for n in names:
            d = info[(task, n)]
            first = d["items"] if task == "image_text_to_text" else len(d["classes"])
            out.append(f"\\hangindent=1em {esc(n)} ({first:,}, {len(d['images']):,})\\par")
        out.append("\\end{multicols}\n")
    NUM["datasets_per_task"] = {t: sum(1 for tt, _ in info if tt == t) for t in ("classification", "detection", "segmentation", "image_text_to_text")}
    open(os.path.join(TAB, "datasets.tex"), "w").write("\n".join(out))


def compute_table(data):
    lines = []
    NUM["compute"] = {}
    for pre, name, _ in MODELS:
        gen, judge, vram, items = [], [], [], None
        for d in sorted(glob.glob(os.path.join(SEEDS, f"s*/{pre}_*/compute.json"))):
            c = json.load(open(d))
            g = sum(x["wall_s"] for x in c if x["phase"] in ("load", "generate"))
            # judge time of the last judging pass only (re-judged copies also keep the earlier v3 pass)
            last = max([i for i, x in enumerate(c) if x["phase"] == "judge_load"], default=len(c))
            j = sum(x["wall_s"] for x in c[last:] if x["phase"] in ("judge_load", "judge"))
            if any(x["phase"] == "generate" for x in c):
                gen.append(g / 3600)
                judge.append(j / 3600)
                vram.append(max(x.get("peak_vram_gb", 0) for x in c if x["phase"] in ("load", "generate")))
                gs = [x for x in c if x["phase"] == "generate"]
                items = sum(x.get("items", 0) for x in gs)
        NUM["compute"][PLAIN[name]] = (round(mean(gen), 2), round(mean(judge), 2), round(max(vram), 1), len(gen))
        # Experiments 2-4: GPU hours summed over all of the model's runs (load + generate; no judging needed).
        e24 = [sum(x["wall_s"] for x in json.load(open(d)) if x["phase"] in ("load", "generate")) / 3600
               for d in glob.glob(os.path.join(EXP24, f"**/{pre}_*/compute.json"), recursive=True)]
        cells = [f"${mean(gen):.1f}$", f"${mean(judge):.1f}$", f"${3 * (mean(gen) + mean(judge)):.1f}$", f"${max(vram):.1f}$"]
        cells += [f"${sum(e24):.1f}$ ({len(e24)})" if e24 else "---"]
        NUM["compute_exp24"] = NUM.get("compute_exp24", {}) | {PLAIN[name]: (round(sum(e24), 2), len(e24))}
        lines.append((name, cells, 3 * (mean(gen) + mean(judge)), sum(e24)))
    t1, t24 = sum(x[2] for x in lines), sum(x[3] for x in lines)
    NUM["compute_total"] = {"exp1": round(t1, 1), "exp24": round(t24, 1)}
    tex = (
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{4}{c}{\\textbf{Experiment 1 (all formats, per seed)}} & \\textbf{Exp.\\ 2--4} \\\\\n"
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-6}\n"
        " & \\makecell{Generation\\\\(h)} & \\makecell{Judge\\\\(h)} & \\makecell{Total, 3 seeds\\\\(h)} & \\makecell{Peak VRAM\\\\(GB)} & \\makecell{Total (h)\\\\(runs)} \\\\\n\\midrule\n"
        + "\n".join(f"{DISP[n]} & " + " & ".join(c) + " \\\\" for n, c, _, _ in lines)
        + f"\n\\midrule\nAll models & & & ${t1:.0f}$ & & ${t24:.0f}$ \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    )
    open(os.path.join(TAB, "compute.tex"), "w").write(tex)
    return NUM["compute"]


def figures(main, ag, fz):
    colors = ["#a6cee3", "#1f78b4", "#fdbf6f", "#ff7f00", "#b2df8a", "#33a02c", "#cab2d6", "#6a3d9a"]
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(10, 4.6 if YB else 4.2))
    n = len(MODELS)
    nb = n + (1 if YB else 0)  # bars per task: the models, then YOLO
    w = 0.8 / nb
    for i, (_, name, _) in enumerate(MODELS):
        xs = [j + (i - nb / 2 + 0.5) * w for j in range(len(COLS))]
        ys = [main[name][k][0] / 100 for k, _ in COLS]
        err = [[(main[name][k][0] - main[name][k][1]) / 100 for k, _ in COLS], [(main[name][k][2] - main[name][k][0]) / 100 for k, _ in COLS]]
        ax.bar(xs, ys, w, label=PLAIN[name] + (" (self-judged)" if "gemma-4-31B" in name else ""), color=colors[i], yerr=err, error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": "#444"})
    if YB:  # YOLO: top-1 accuracy in the three classification columns (it sees no question), detection, segmentation
        ycol = {"mcqa_across": "top1", "mcqa_within": "top1", "oeq": "top1", "detection": "detection", "segmentation": "segmentation"}
        js = [j for j, (k, _) in enumerate(COLS) if k in ycol]
        v = [YB[ycol[COLS[j][0]]] for j in js]
        xs = [j + (n - nb / 2 + 0.5) * w for j in js]
        ax.bar(xs, [x[0] / 100 for x in v], w, facecolor="#d9d9d9", edgecolor="#555555", hatch="////", lw=0.6,
               label=f"{YOLO_LABEL}\n(top-1 for classification)",
               yerr=[[(x[0] - x[1]) / 100 for x in v], [(x[2] - x[0]) / 100 for x in v]], error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": "#444"})
    for kind, label, style in (("chance", "Chance (random answer)", (0, (3, 2))), ("prior", "Prior (best constant answer)", (0, (1, 1.5)))):
        if kind in BASE:  # reference lines across each task's group of bars
            for j, (k, _) in enumerate(COLS):
                ax.hlines(BASE[kind][k] / 100, j - 0.43, j + 0.43, colors="#222222", linestyles=[style], lw=1.1, zorder=5,
                          label=label if j == 0 else None)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([f"{h}\n(n={main[MODELS[0][1]][k][3]})" for k, h in COLS])
    ax.set_ylabel("Accuracy (avg. across datasets)")
    ax.set_ylim(0, 0.9 if YB else 0.8)
    ax.yaxis.grid(True, color="#ddd")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Task (n = number of datasets)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.16 if YB else 0.11, 1, 1))
    fig.savefig(os.path.join(PAPER, "fig_main_results.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), sharey=True)
    nb = n + (1 if YB else 0)  # bars per agricultural task: the models, then YOLO
    w = 0.8 / nb
    for ax, (f, title) in zip(axes, [("mcqa_across", "MCQA across"), ("mcqa_within", "MCQA within"), ("oeq", "OEQ (judged)")]):
        if YB:  # YOLO top-1 accuracy: image only, so the same bar in every panel
            xs = [j + (n - nb / 2 + 0.5) * w for j in range(len(AG))]
            ys = [YB["ag"][a][0] / 100 for a, _ in AG]
            err = [[ys[j] - YB["ag"][a][1] / 100 for j, (a, _) in enumerate(AG)], [YB["ag"][a][2] / 100 - ys[j] for j, (a, _) in enumerate(AG)]]
            ax.bar(xs, ys, w, facecolor="#d9d9d9", edgecolor="#555555", hatch="////", lw=0.6, zorder=2,
                   label=f"{YOLO_LABEL}, top-1 (same in every panel)")
            ax.errorbar(xs, ys, yerr=err, fmt="none", elinewidth=0.6, capsize=1.0, ecolor="#444", zorder=4)
        for i, (_, name, _) in enumerate(MODELS):
            xs = [j + (i - nb / 2 + 0.5) * w for j in range(len(AG))]
            ys = [ag[name][(f, a)] / 100 for a, _ in AG]
            err = [[ys[j] - AG_CI[(name, f, a)][0] / 100 for j, (a, _) in enumerate(AG)], [AG_CI[(name, f, a)][1] / 100 - ys[j] for j, (a, _) in enumerate(AG)]]
            ax.bar(xs, ys, w, color=colors[i], label=PLAIN[name], zorder=2)
            ax.errorbar(xs, ys, yerr=err, fmt="none", elinewidth=0.6, capsize=1.0, ecolor="#444", zorder=4)
            if f == "oeq":  # fuzzy-match score: hatched ghost bar behind, dark tick at its value
                ys = [fz[name][a] / 100 for a, _ in AG]
                ax.bar(xs, ys, w, facecolor="none", edgecolor=colors[i], hatch="//////", lw=0, zorder=1)
                ax.hlines(ys, [x - w / 2 for x in xs], [x + w / 2 for x in xs], color="#222222", lw=0.9, zorder=3)
        ax.set_xticks(range(len(AG)))
        ax.set_xticklabels([h.replace("/", "/\n") for _, h in AG])   # wrap "Pest/Damage" so labels do not collide
        ax.set_title(title)
        ax.yaxis.grid(True, color="#ddd")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[2].set_title("OEQ (judged vs. fuzzy match)")
    axes[0].set_ylabel("Accuracy (avg. across datasets)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h, l = axes[0].get_legend_handles_labels()
    if YB:  # YOLO's bar is drawn first; list it after the models
        h, l = h[1:] + h[:1], l[1:] + l[:1]
    h += [Patch(facecolor="#888888"), (Patch(facecolor="none", edgecolor="#888888", hatch="//////", lw=0), Line2D([], [], color="#222222", lw=0.9))]
    l += ["OEQ: LLM judge", "OEQ: fuzzy match"]
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    axes[0].set_ylim(0, 0.9 if YB else 0.75)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(PAPER, "fig_agtask.png"), dpi=200)
    plt.close(fig)


def yolo_tests(data):
    """Paired bootstrap over datasets, YOLO11 (10 images per class) minus each VLM, per column, Holm-corrected over the
    eight models (agbench.eval.stats.paired / holm, the tests used between models). YOLO's per-dataset value is the mean
    over its training subsets (top-1 accuracy for the classification columns), each VLM's the mean over its seeds."""
    if not YB:
        return
    from agbench.eval.stats import holm, paired

    yruns = [load_per_dataset(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(YOLO, f"{YOLO_SHOTS}shot", "s*", "predictions.jsonl")))]
    per_ds = lambda runs, col: {d: 100 * mean([mean(r[0][(col, d)]) for r in runs]) for d in {d for (k, d) in runs[0][0] if k == col}}  # noqa: E731
    ycol = {"mcqa_across": "oeq", "mcqa_within": "oeq", "oeq": "oeq", "detection": "detection", "segmentation": "segmentation"}
    out = {}
    for col, yc in ycol.items():
        y = per_ds(yruns, yc)
        res = {PLAIN[name]: paired(y, per_ds(data[name]["runs"], col)) for _, name, _ in MODELS}
        ph = dict(zip(res, holm([r["p_boot"] for r in res.values()])))
        closest = min(res, key=lambda m: res[m]["diff"])
        out[col] = {"closest": closest, "diff": round(res[closest]["diff"], 2), "lo": round(res[closest]["lo"], 2),
                    "hi": round(res[closest]["hi"], 2), "max_p_holm": max(ph.values()), "n": res[closest]["n"],
                    "all_diffs": {m: round(r["diff"], 2) for m, r in res.items()}}
    NUM["yolo_tests"] = out
    NUM["yolo_tests_min_diff"] = round(min(v["diff"] for v in out.values()), 2)


def yolo_trend(main):
    """Figure B1: YOLO11 accuracy against training images per class (classification top-1, detection, segmentation), mean
    over the three training subsets with their range, with the VLMs and the naive baselines as horizontal lines. For
    classification the lines are the best VLM under MCQA across and under OEQ, and the OEQ naive baselines, which like
    top-1 accuracy name a class from the dataset's list and are scored by exact match."""
    shots = [5, 10, 15]
    recs = {k: yolo_baseline(k) for k in shots}
    if not all(recs.values()):
        return
    cols = [("top1", "Classification (top-1)"), ("detection", "Detection"), ("segmentation", "Segmentation")]
    NUM["yolo_trend"] = {k: {c: [round(recs[k][c][0], 2), round(min(recs[k]["per_subset"][c]), 2), round(max(recs[k]["per_subset"][c]), 2)]
                             for c, _ in cols} for k in shots}
    spread = {c: max(max(recs[k]["per_subset"][c]) - min(recs[k]["per_subset"][c]) for k in shots) for c, _ in cols}
    NUM["yolo_spread"] = {"by_task": {c: round(v, 2) for c, v in spread.items()},
                          "max_except_segmentation": round(max(v for c, v in spread.items() if c != "segmentation"), 2)}
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, len(cols), figsize=(9, 3.0), sharey=True)  # one scale, as in Figure 4
    for ax, (c, h) in zip(axes, cols):
        ys = [recs[k][c][0] / 100 for k in shots]
        lo = [min(recs[k]["per_subset"][c]) / 100 for k in shots]
        hi = [max(recs[k]["per_subset"][c]) / 100 for k in shots]
        ax.fill_between(shots, lo, hi, color="#555555", alpha=0.18, lw=0)
        ax.plot(shots, ys, "-o", color="#333333", ms=4, lw=1.4, label="YOLO11s (mean, range over 3 subsets)", zorder=3)
        if c == "top1":
            ax.axhline(max(main[n]["mcqa_across"][0] for n in main) / 100, color="#1f78b4", lw=1.2, label="Best VLM (MCQA across for classification)")
            ax.axhline(max(main[n]["oeq"][0] for n in main) / 100, color="#e31a1c", lw=1.2, ls=(0, (5, 1.5, 1, 1.5)), label="Best VLM, OEQ (judged)")
            base_col = "oeq"
        else:
            ax.axhline(max(main[n][c][0] for n in main) / 100, color="#1f78b4", lw=1.2)
            base_col = c
        for kind, style, lab in (("chance", (0, (3, 2)), "Chance"), ("prior", (0, (1, 1.5)), "Prior")):
            if kind in BASE:
                ax.axhline(BASE[kind][base_col] / 100, color="#222222", lw=0.9, ls=style, label=lab if c == "top1" else None)
        ax.set_title(h)
        ax.set_xticks(shots)
        ax.set_xlim(4, 16)
        ax.set_ylim(0, 0.9)
        ax.yaxis.grid(True, color="#e5e5e5")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Accuracy (avg. across datasets)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.text(0.5, 0.17, "Training images per class", ha="center", fontsize=9)
    fig.savefig(os.path.join(PAPER, "fig_yolo_trend.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    data = collect()
    BASE = baselines()
    YB = yolo_baseline()
    NUM["seeds_per_model"] = {PLAIN[n]: v["seeds"] for n, v in data.items()}
    main = main_table(data)
    ag = agtask_table(data)
    judge_table(data)
    itt_table(data)
    stats_tables()
    compute_table(data)
    datasets_list()
    answer_flips()
    judge_validation()
    figures(main, ag, oeq_fuzzy_by_ag(data))
    yolo_trend(main)
    yolo_tests(data)
    json.dump(NUM, open(os.path.join(PAPER, "numbers.json"), "w"), indent=1, default=str)
    print(json.dumps({k: NUM[k] for k in ("seeds_per_model", "main", "compute")}, indent=1))
