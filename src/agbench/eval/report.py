"""Leaderboard tables across runs."""

import json
from pathlib import Path

import pandas as pd


def load_runs(paths) -> list[dict]:
    runs = []
    for p in paths:
        p = Path(p)
        dirs = [p] if (p / "run.json").exists() else sorted(d.parent for d in p.glob("**/run.json"))
        for d in dirs:
            if (d / "metrics.json").exists():
                run = json.loads((d / "run.json").read_text())
                runs.append({"dir": str(d), "name": run["method"].get("name", d.name), "run": run,
                             "metrics": json.loads((d / "metrics.json").read_text())})
    names = [r["name"] for r in runs]
    for r in runs:  # same model in several runs: label by effort, else by run dir (profile seq/batch)
        same = [x for x in runs if x["name"] == r["name"]]
        if len(same) > 1:
            efforts = [x["run"]["method"].get("effort") for x in same]
            tag = efforts[same.index(r)] if None not in efforts and len(set(efforts)) == len(efforts) else Path(r["dir"]).name
            r["label"] = f"{r['name']} ({tag})"
    for r in runs:
        r["name"] = r.pop("label", r["name"])
    return runs


def summary_frames(runs: list[dict], average: str = "macro") -> dict[str, pd.DataFrame]:
    cv, ag, ds = [], [], []
    for r in runs:
        m = r["metrics"]
        for row in m["cv_task"]:
            cv.append({"model": r["name"], "run": r["dir"], **row})
        for row in m["ag_task"]:
            ag.append({"model": r["name"], "run": r["dir"], **row})
        for row in m["datasets"]:
            ds.append({"model": r["name"], "run": r["dir"], **row})
    cv, ag, ds = pd.DataFrame(cv), pd.DataFrame(ag), pd.DataFrame(ds)
    out = {"cv_rows": cv, "ag_rows": ag, "datasets": ds}
    if len(cv):
        for metric in ("accuracy", "f1"):
            out[f"cv_{metric}"] = cv.pivot_table(index="model", columns="cv_task", values=f"{metric}_{average}")
        if f"rouge_l_{average}" in cv and cv[f"rouge_l_{average}"].notna().any():
            out["cv_rouge_l"] = cv.pivot_table(index="model", columns="cv_task", values=f"rouge_l_{average}")
    if len(ag):
        a = ag[ag.cv_task == "all"]
        for metric in ("accuracy", "f1"):
            out[f"ag_{metric}"] = a.pivot_table(index="model", columns="ag_task", values=f"{metric}_{average}")
    return out


def leaderboard_md(runs: list[dict], average: str = "macro") -> str:
    f = summary_frames(runs, average)
    parts = [f"# Leaderboard ({average} over datasets)\n"]
    for key, title in (("cv_accuracy", "Accuracy by CV task"), ("cv_f1", "F1 by CV task"), ("cv_rouge_l", "ROUGE-L, open-ended ITT"),
                       ("ag_accuracy", "Accuracy by ag task"), ("ag_f1", "F1 by ag task")):
        if key in f:
            parts += [f"## {title}\n", (f[key] * 100).round(1).to_markdown(), ""]
    pending = [r["name"] for r in runs if r["metrics"].get("pending_judge")]
    if pending:
        parts.append(f"Not yet judged (open-ended scored by fuzzy match): {', '.join(pending)}")
    return "\n".join(parts) + "\n"
