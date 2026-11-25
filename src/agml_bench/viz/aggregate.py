"""Scores for charts and reports, aggregated as in the paper.

Per run, an item's score is the judge verdict for an open-ended answer and its graded score otherwise (accuracy for
boxes). Scores are averaged per dataset; repeated runs of one entry (sampling seeds, YOLO training subsets, chance
seeds) are averaged per dataset; the entry's score is the macro mean over datasets, with a 95% bootstrap CI over
datasets.

An entry is one method with one setting: runs that differ only in their sampling seed are its replicates. Chance
baselines (baseline/chance) become reference lines; YOLO runs (method name yolo*) score classification by top-1
accuracy, the same value in every classification format.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COLS = [("mcqa_across", "MCQA across"), ("mcqa_within", "MCQA within"), ("oeq", "OEQ"), ("itt", "VQA"),
        ("detection", "Detection"), ("segmentation", "Segmentation")]
AG = [("disease", "Disease"), ("pest_damage", "Pest/Damage"), ("species_id", "Species"), ("quality_maturity", "Quality")]
CLS_FORMATS = [("mcqa_across", "MCQA across"), ("mcqa_within", "MCQA within"), ("oeq", "OEQ (judged)")]


def column(p: dict) -> str:
    if p["task"] == "classification":
        return p["format"]
    return "itt" if p["task"] == "image_text_to_text" else p["task"]


def item_score(p: dict) -> float:
    if p.get("needs_judge"):
        return 1.0 if p.get("judge_correct") else 0.0
    c = p.get("correct")
    if isinstance(c, bool):
        return 1.0 if c else 0.0
    return float(p.get("acc", c) or 0.0)


def _mean(v):
    return sum(v) / len(v) if len(v) else float("nan")


def boot_ci(vals, B: int = 10000, seed: int = 0, exact: bool = False) -> tuple[float, float]:
    """95% bootstrap CI of the mean. `exact`: the paper's pure-Python resampling (same numbers, slower)."""
    if not len(vals):
        return float("nan"), float("nan")
    if exact:
        rnd, n = random.Random(seed), len(vals)
        ms = sorted(sum(rnd.choice(vals) for _ in range(n)) / n for _ in range(B))
        return ms[int(0.025 * B)], ms[int(0.975 * B) - 1]
    v = np.asarray(vals, float)
    ms = np.sort(v[np.random.default_rng(seed).integers(0, len(v), (B, len(v)))].mean(1))
    return float(ms[int(0.025 * B)]), float(ms[int(0.975 * B) - 1])


@dataclass
class RunScores:
    """Per-dataset mean scores of one run."""
    dir: str
    per: dict = field(default_factory=dict)       # (col, dataset) -> mean score
    ag: dict = field(default_factory=dict)        # (format, ag_task, dataset) -> mean score (classification)
    fuzzy: dict = field(default_factory=dict)     # (ag_task, dataset) -> mean fuzzy-match score (OEQ)
    itt_fmt: dict = field(default_factory=dict)   # qa_format -> pooled mean over questions
    pending_judge: int = 0

    @classmethod
    def load(cls, run_dir) -> "RunScores":
        per, ag, fz, itt = (defaultdict(list) for _ in range(4))
        pending = 0
        with open(Path(run_dir) / "predictions.jsonl") as f:
            for line in f:
                p = json.loads(line)
                s, k = item_score(p), column(p)
                pending += bool(p.get("needs_judge") and p.get("judge_correct") is None)
                per[(k, p["dataset"])].append(s)
                if p["task"] == "classification":
                    ag[(k, p.get("ag_task"), p["dataset"])].append(s)
                    if k == "oeq":
                        fz[(p.get("ag_task"), p["dataset"])].append(1.0 if p.get("correct") else 0.0)
                elif p["task"] == "image_text_to_text":
                    itt[p.get("qa_format")].append(s)
        m = lambda d: {k: _mean(v) for k, v in d.items()}  # noqa: E731
        return cls(str(run_dir), m(per), m(ag), m(fz), m(itt), pending)


@dataclass
class Entry:
    label: str
    kind: str                     # model | yolo | chance
    runs: list[RunScores]
    meta: list[dict]              # run_manifest.json (or run.json) of each run
    cells: dict = field(default_factory=dict)   # col -> (mean, lo, hi, n datasets), percent
    ag: dict = field(default_factory=dict)      # (format, ag_task) -> (mean, lo, hi, n)
    fuzzy: dict = field(default_factory=dict)   # ag_task -> mean, percent
    itt_fmt: dict = field(default_factory=dict)  # qa_format -> mean, percent
    datasets: dict = field(default_factory=dict)  # (col, dataset) -> mean over replicates, percent

    @property
    def replicates(self) -> int:
        return len(self.runs)


def _cell(runs: list[RunScores], key_of, exact: bool) -> tuple:
    keys = sorted({k for r in runs for k in key_of(r)})
    vals = [100 * _mean([key_of(r)[k] for r in runs if k in key_of(r)]) for k in keys]
    return (_mean(vals), *boot_ci(vals, exact=exact), len(vals)) if vals else (float("nan"),) * 3 + (0,)


def _meta(d: Path) -> dict:
    for f in ("run_manifest.json", "run.json"):
        if (d / f).exists():
            return json.loads((d / f).read_text())
    return {}


def _setting(meta: dict) -> tuple:
    """What makes runs one entry: the method and its settings, minus the sampling seed."""
    m = dict(meta.get("method") or {})
    m.pop("gen_seed", None)
    m["body"] = {k: v for k, v in (m.get("body") or {}).items() if k != "seed"}
    args = meta.get("benchmark") if "item_seed" not in meta else {"seed": meta.get("item_seed"), "mcqa": meta.get("mcqa")}
    args = args or {}
    return json.dumps(m, sort_keys=True, default=str), json.dumps({"seed": args.get("seed"), "mcqa": args.get("mcqa")}, sort_keys=True)


def run_dirs(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        dirs = [p] if (p / "predictions.jsonl").exists() else sorted(d.parent for d in p.glob("**/predictions.jsonl"))
        out += [d for d in dirs if (d / "run.json").exists()]
    return out


def entries(paths, labels: dict | None = None, exact: bool = False) -> list[Entry]:
    """Group the runs under `paths` into entries and aggregate them. `labels`: {method name: display label}."""
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for d in run_dirs(paths):
        meta = _meta(d)
        groups[(str((meta.get("method") or {}).get("name", d.name)), *_setting(meta))].append(d)
    names = defaultdict(list)
    for key in groups:
        names[key[0]].append(key)
    out = []
    for key, dirs in groups.items():
        name = key[0]
        kind = "chance" if name == "baseline/chance" else "yolo" if name.lower().startswith("yolo") else "model"
        label = (labels or {}).get(name, name.split("/")[-1] if kind == "model" else name)
        if len(names[name]) > 1:  # one method in several settings
            label += f" [{names[name].index(key) + 1}]"
        runs = [RunScores.load(d) for d in dirs]
        e = Entry(label, kind, runs, [_meta(d) for d in dirs])
        for col, _ in COLS:
            src = "oeq" if kind == "yolo" and col in ("mcqa_across", "mcqa_within") else col
            e.cells[col] = _cell(runs, lambda r, s=src: {d: v for (c, d), v in r.per.items() if c == s}, exact)
        for f, _ in CLS_FORMATS:
            for a, _ in AG:
                src = "oeq" if kind == "yolo" else f
                e.ag[(f, a)] = _cell(runs, lambda r, s=src, a=a: {d: v for (c, g, d), v in r.ag.items() if c == s and g == a}, exact)
        for a, _ in AG:
            e.fuzzy[a] = _cell(runs, lambda r, a=a: {d: v for (g, d), v in r.fuzzy.items() if g == a}, exact)[0]
        for q in sorted({q for r in runs for q in r.itt_fmt}):
            e.itt_fmt[q] = 100 * _mean([r.itt_fmt[q] for r in runs if q in r.itt_fmt])
        for k in sorted({k for r in runs for k in r.per}):
            e.datasets[k] = 100 * _mean([r.per[k] for r in runs if k in r.per])
        out.append(e)
    order = {"model": 0, "yolo": 1, "chance": 2}
    return sorted(out, key=lambda e: (order[e.kind], e.label.lower()))
