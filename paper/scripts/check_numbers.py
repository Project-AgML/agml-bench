"""Audit: is every number in the paper's text among the values the pipeline computed?

Extracts each number from paper/main.tex (abstract to end, captions included; references, labels, paths and code
removed) and looks for it, at the precision it is written with, among:
  - paper/numbers.json and paper/drivers.json   (written by make_assets.py, make_drivers.py)
  - the statistics reports <AGBENCH_STATS>/{seeds,exp24}/stats.md   (agbench eval stats)
  - the generated tables paper/tables/*.tex
  - benchmark counts recomputed from the items of one Experiment 1 run (datasets, classes, images, items)
  - configuration: src/agbench/eval/{models,prompts}.yaml, configs/base_data.yaml, one run.json
A number is "found" if some source value rounds to it (sign ignored, so "-2.7" and "2.7" match; values in [0, 1]
are also tried as percentages). This is an existence check: it catches numbers that appear in no output
(typos, stale or invented values), but cannot prove that a found number is quoted for the right quantity. The report
states how strict the test is for each precision: integers and counts are a strong test, but almost every
one-decimal value between 0 and 100 occurs somewhere in the outputs, so for those it only catches gross errors.
Integers up to 12 (counts such as "three seeds", "Experiment 2") are skipped.

usage: uv run python paper/scripts/check_numbers.py        # report: paper/build/number_check.txt
"""
import glob
import json
import os
import random
import re
from collections import defaultdict

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(P)
SEEDS = os.path.join(ROOT, os.environ.get("AGBENCH_SEEDS", "runs/rejudge_v4/exp/seeds"))
STATS = os.path.join(ROOT, os.environ.get("AGBENCH_STATS", "reports/stats"))
NUM = re.compile(r"(?<![\w.\\])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")


def text_numbers():
    s = open(os.path.join(P, "main.tex")).read()
    s = s[s.index("\\begin{abstract}"):]
    s = s[: s.index("\\bibliographystyle")] + s[s.index("\\appendix"):]
    s = re.sub(r"(?<!\\)%.*", "", s)                                          # comments
    s = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{tabular\}\{[^}]*\}", " ", s)
    for cmd in ("ref", "label", "cite", "url", "nolinkurl", "path", "code", "texttt", "includegraphics", "input",
                "setlength", "setcounter", "renewcommand", "arraystretch", "cmidrule", "multirow", "multicolumn",
                "resizebox", "hspace", "vspace", "begin", "end", "makecell", "p"):
        s = re.sub(r"\\" + cmd + r"\*?(\[[^\]]*\])?(\{[^{}]*\})*", " ", s)
    s = re.sub(r"\\textwidth|\\linewidth|\\[a-zA-Z]+", " ", s)
    s = re.sub(r"\[\d+pt\]|\d+(\.\d+)?(cm|pt|in)\b", " ", s)
    s = re.sub(r"\b(Qwen|Gemma|Muse-Glimmer|Nemotron|MiniCPM-V|Ministral|Bolt|RTX|vLLM|v)[-\w.]*", " ", s)  # model names
    s = re.sub(r"\b\d+(\.\d+)?B\b|A\dB|FP8|BF16|GGUF|CUDA \d+(\.\d+)?|\d+K\b", " ", s)
    s = re.sub(r"\b(19|20)\d\d\b", " ", s)                                       # years
    s = re.sub(r"\b\d+\.\d+\.\d+\b", " ", s)                                  # software versions (3.14.6)
    out = []
    for line_no, line in enumerate(s.split("\n"), 1):
        for m in NUM.finditer(line):
            whole, frac = m.group(1).replace(",", ""), m.group(2) or ""
            v = float(whole + frac)
            if not frac and v <= 12:
                continue
            ctx = line[max(0, m.start() - 50): m.end() + 30].strip()
            out.append((v, len(frac) - 1 if frac else 0, ctx))
    return out


def walk(x, acc):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        acc.append(float(x))
    elif isinstance(x, dict):
        for k, v in x.items():
            walk(v, acc)
            for n in NUM.finditer(str(k)):
                acc.append(float(n.group(1).replace(",", "") + (n.group(2) or "")))
    elif isinstance(x, (list, tuple)):
        for v in x:
            walk(v, acc)
    elif isinstance(x, str):
        for n in NUM.finditer(x):
            acc.append(float(n.group(1).replace(",", "") + (n.group(2) or "")))


def counts():
    """Benchmark composition from the items of one Experiment 1 run (every model sees the same items)."""
    run = sorted(glob.glob(os.path.join(SEEDS, "s0/*/items.jsonl")))[0]
    ds, cls, imgs, items = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(int)
    ag_ds, ag_img, qa = defaultdict(set), defaultdict(set), defaultdict(int)
    for line in open(run):
        r = json.loads(line)
        t = r["task"]
        ds[t].add(r["dataset"]); items[t] += 1
        imgs[t].update(json.dumps(i) for i in r.get("images") or [])
        for c in r.get("class_names") or []:
            cls[t].add((r["dataset"], c))
        if t == "classification" and r.get("ag_task"):
            ag_ds[r["ag_task"]].add(r["dataset"]); ag_img[r["ag_task"]].add(r["sample_id"])
        if r.get("qa_format"):
            qa[r["qa_format"]] += 1
    vals = [len(v) for v in ds.values()] + [len(v) for v in cls.values()] + [len(v) for v in imgs.values()]
    vals += list(items.values()) + [len(v) for v in ag_ds.values()] + [len(v) for v in ag_img.values()] + list(qa.values())
    vals += [sum(len(v) for v in ds.values()), sum(len(v) for v in imgs.values()), sum(items.values())]
    return [float(v) for v in vals]


def sources():
    acc = []
    for f in ("numbers.json", "drivers.json"):
        if os.path.exists(os.path.join(P, f)):
            walk(json.load(open(os.path.join(P, f))), acc)
    for f in glob.glob(os.path.join(STATS, "*/stats.md")) + glob.glob(os.path.join(P, "tables/*.tex")):
        walk(open(f).read(), acc)
    acc += counts()
    # configuration constants the text quotes (token limits, sampling caps, box scale, GPU model)
    for f in ("src/agbench/eval/models.yaml", "src/agbench/eval/prompts.yaml", "configs/base_data.yaml"):
        walk(open(os.path.join(ROOT, f)).read(), acc)
    walk(json.load(open(sorted(glob.glob(os.path.join(SEEDS, "s0/*/run.json")))[0])), acc)
    acc.append(6000.0)  # "RTX 6000 Ada", the GPU model name
    return acc


def main():
    src = sources()
    have = defaultdict(set)
    for v in src:
        for x in (abs(v), abs(v) * 100 if abs(v) <= 1 else None):
            if x is not None:
                for d in range(4):
                    have[d].add(round(x, d))
    found, missing = [], []
    nums = text_numbers()
    # the appendix quotes this audit's own totals (all numbers, and integers)
    for x in (len(nums), sum(1 for _, d, _ in nums if d == 0)):
        for d in range(4):
            have[d].add(float(x))
    for v, d, ctx in nums:
        (found if round(v, d) in have[d] else missing).append((v, d, ctx))
    # How strict is the test? Share of random numbers of the same precision and range that would also be "found".
    random.seed(0)
    chance = {}
    for d in sorted({d for _, d, _ in nums}):
        hi = max(v for v, dd, _ in nums if dd == d)
        xs = [round(random.uniform(0, hi), d) for _ in range(5000)]
        chance[d] = (sum(1 for v, dd, _ in nums if dd == d), 100 * sum(x in have[d] for x in xs) / len(xs), hi)
    os.makedirs(os.path.join(P, "build"), exist_ok=True)
    rep = os.path.join(P, "build", "number_check.txt")
    with open(rep, "w") as f:
        f.write(f"{len(found)} of {len(found) + len(missing)} numbers in the text found among {len(src)} computed values\n")
        f.write("Strictness: share of random numbers (same precision, 0 to the largest such number in the text) also found\n")
        f.write("".join(f"  {d} decimals: {n} numbers in the text, chance match {c:.0f}% (range 0-{hi:g})\n" for d, (n, c, hi) in chance.items()))
        f.write("A low chance match means a found number is strong evidence; a high one means the test only catches gross errors.\n\n")
        f.write("Not found:\n" + "".join(f"  {v:g}\t{ctx}\n" for v, d, ctx in missing))
    print(open(rep).read())


if __name__ == "__main__":
    main()
