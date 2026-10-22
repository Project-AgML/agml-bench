"""Streamlit viewer for benchmark results. Launch with: agbench eval view --runs runs/"""

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from agbench.eval.report import load_runs, summary_frames
from agbench.eval.runner import read_jsonl

st.set_page_config(page_title="AgBench results", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True)
parser.add_argument("--profile", default=None)
args, _ = parser.parse_known_args()
THUMB = 520
PCT = ["accuracy_macro", "f1_macro", "accuracy_micro", "f1_micro", "idk_rate", "nota_accuracy", "truncation_rate",
       "parse_fail_rate", "accuracy", "f1", "error_rate", "mean_iou", "rouge_l", "rouge_l_macro", "rouge_l_micro"]


@st.cache_data(ttl=60)
def runs():
    return load_runs(args.runs)


@st.cache_data(ttl=60)
def predictions(run_dir: str) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(Path(run_dir) / "predictions.jsonl"))


def pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in PCT or (df[c].dtype.kind == "f" and df[c].between(0, 1).all() and c not in ("latency_s",)):
            df[c] = (df[c] * 100).round(1)
    return df


def no_runs() -> bool:
    if not runs():
        st.info(f"No finished runs under {args.runs}. Run `agbench eval run ...` first.")
        return True
    return False


def page_summary():
    st.title("Summary")
    if no_runs():
        return
    avg = st.radio("Average over datasets", ["macro", "micro"], horizontal=True,
                   help="macro: mean of per-dataset scores. micro: pooled over all items.")
    f = summary_frames(runs(), avg)
    pending = [r["name"] for r in runs() if r["metrics"].get("pending_judge")]
    if pending:
        st.warning(f"Open-ended answers not judged yet (fuzzy match used): {', '.join(pending)}")
    self_judged = [r["name"] for r in runs() if (r["run"].get("judge") or {}).get("self_judged")]
    if self_judged:
        st.caption(f"Self-judged (judge = model under test): {', '.join(self_judged)}")
    for title, key in (("By CV task", "cv"), ("By ag task", "ag")):
        st.subheader(title)
        a, b = st.columns(2)
        for col, metric in ((a, "accuracy"), (b, "f1")):
            if f"{key}_{metric}" in f:
                t = (f[f"{key}_{metric}"] * 100).round(1)
                col.markdown(f"**{metric.capitalize()} (%)**")
                col.dataframe(t, width="stretch")
                col.bar_chart(t.T, stack=False, height=260)
    if "cv_rouge_l" in f:
        st.markdown("**ROUGE-L, open-ended ITT (%)**")
        st.dataframe((f["cv_rouge_l"] * 100).round(1), width="stretch")
    with st.expander("All summary columns (IDK rate, NOTA accuracy, truncation, parse failures)"):
        st.dataframe(pct(f["cv_rows"].drop(columns=["run"])), hide_index=True, width="stretch")
        st.dataframe(pct(f["ag_rows"].drop(columns=["run"])), hide_index=True, width="stretch")


def page_datasets():
    st.title("Datasets")
    if no_runs():
        return
    ds = summary_frames(runs())["datasets"]
    c1, c2, c3, c4 = st.columns(4)
    models = c1.multiselect("Model", sorted(ds.model.unique()), default=sorted(ds.model.unique()))
    cv = c2.multiselect("CV task", sorted(ds.cv_task.unique()), default=sorted(ds.cv_task.unique()))
    ag = c3.multiselect("Ag task", sorted(ds.ag_task.unique()), default=sorted(ds.ag_task.unique()))
    q = c4.text_input("Search dataset")
    v = ds[ds.model.isin(models) & ds.cv_task.isin(cv) & ds.ag_task.isin(ag)]
    if q:
        v = v[v.dataset.str.contains(q, case=False)]
    metric = st.radio("Metric", ["accuracy", "f1"], horizontal=True)
    wide = v.pivot_table(index=["dataset", "cv_task", "ag_task"], columns="model", values=metric).reset_index()
    st.dataframe(pct(wide), hide_index=True, width="stretch", height=420)
    with st.expander("Long format (all columns)"):
        st.dataframe(pct(v.drop(columns=["run"])), hide_index=True, width="stretch")


def draw_boxes(path: str, gt: list, pred: list, labels: bool = False) -> Image.Image:
    im = Image.open(path).convert("RGB")
    d, lw = ImageDraw.Draw(im), max(2, im.width // 300)
    for b, color in [(b, (40, 200, 80)) for b in gt or []] + [(b, (230, 60, 60)) for b in pred or []]:
        d.rectangle(b["bbox"], outline=color, width=lw)
        if labels and b.get("label"):
            d.text((b["bbox"][0] + lw, b["bbox"][1] + lw), str(b["label"]), fill=color, font_size=max(12, im.width // 40))
    im.thumbnail((THUMB, THUMB))
    return im


def page_boxes():
    st.title("Boxes")
    if no_runs():
        return
    st.caption("The same image across runs. Green = ground truth, red = prediction.")
    preds = {}
    for r in runs():
        if Path(r["dir"]).name == "seq":  # profile: the batched answers cover the same items
            continue
        p = predictions(r["dir"])
        p = p[p.task.isin(["detection", "segmentation"])] if "task" in p else p.iloc[0:0]
        if len(p):
            preds[r["name"]] = (r["dir"], p.set_index("item_id"))
    if not preds:
        st.info("No detection or segmentation answers in these runs.")
        return
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    chosen = c1.multiselect("Runs", list(preds), default=list(preds))
    if not chosen:
        return
    datasets = sorted(set().union(*(set(preds[n][1].dataset) for n in chosen)), key=str.lower)
    ds = c2.selectbox("Dataset", ["all"] + datasets)
    cols = int(c3.number_input("Columns", 1, 8, min(4, len(chosen))))
    labels = c4.checkbox("Labels", value=True)
    ids = sorted({i for n in chosen for i, d in preds[n][1].dataset.items() if ds in ("all", d)})
    items = {}
    for n in chosen:
        f = Path(preds[n][0]) / "items.jsonl"
        if f.exists():
            items.update({it["item_id"]: it for it in read_jsonl(f) if it["item_id"] in set(ids)})
    page_size = 5
    page = st.number_input("Page", 1, max(1, (len(ids) + page_size - 1) // page_size), 1)
    for iid in ids[(page - 1) * page_size: page * page_size]:
        it = items.get(iid, {})
        img = next((i for i in it.get("images") or [] if Path(i).exists()), None)
        first = next(preds[n][1].loc[iid] for n in chosen if iid in preds[n][1].index)
        with st.container(border=True):
            st.markdown(f"`{iid}` · {first['dataset']} · classes: {', '.join(map(str, it.get('class_names') or []))[:300]}")
            if not img:
                st.warning("Image not found on this machine.")
                continue
            grid = st.columns(cols)
            for i, n in enumerate(chosen):
                if iid not in preds[n][1].index:
                    continue
                r = preds[n][1].loc[iid]
                with grid[i % cols]:
                    st.image(draw_boxes(img, r.get("gt_boxes"), r.get("pred_boxes"), labels), width="stretch")
                    st.caption(f"**{n}** · TP {int(r['tp'])} FP {int(r['fp'])} FN {int(r['fn'])}" + ("" if r.get("parse_ok") else " · unparsed"))


def page_samples():
    st.title("Samples")
    if no_runs():
        return
    names = {f"{r['name']} · {Path(r['dir']).name}": r["dir"] for r in runs()}
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    run_dir = names[c1.selectbox("Run", list(names))]
    p = predictions(run_dir)
    dataset = c2.selectbox("Dataset", sorted(p.dataset.unique(), key=str.lower))
    v = p[p.dataset == dataset]
    fmt = c3.selectbox("Format", sorted(v.format.unique()))
    which = c4.selectbox("Show", ["all", "wrong", "correct"])
    v = v[v.format == fmt]
    ok = v.apply(lambda r: r["judge_correct"] if r.get("judge_correct") is not None and r["judge_correct"] == r["judge_correct"] else r["correct"], axis=1)
    if which != "all":
        v = v[(ok >= 0.5) == (which == "correct")]
    items = {it["item_id"]: it for it in read_jsonl(Path(run_dir) / "items.jsonl")} if (Path(run_dir) / "items.jsonl").exists() else {}
    page_size = 10
    pages = max(1, (len(v) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, pages, 1)
    st.caption("Boxes: green = ground truth, red = prediction")
    for _, r in v.iloc[(page - 1) * page_size: page * page_size].iterrows():
        it = items.get(r["item_id"], {})
        with st.container(border=True):
            a, b = st.columns([1, 2])
            with a:
                for img in (it.get("images") or [])[:3]:
                    if Path(img).exists():
                        st.image(draw_boxes(img, r.get("gt_boxes"), r.get("pred_boxes")) if r["task"] in ("detection", "segmentation")
                                 else Image.open(img).convert("RGB").resize((THUMB, int(THUMB * Image.open(img).height / Image.open(img).width))),
                                 width="stretch")
            with b:
                score = r["correct"] if r["task"] in ("detection", "segmentation") else ok.loc[_]
                st.markdown(f"**{'✅' if score >= 0.5 else '❌'}** `{r['item_id']}` · ag task: {r['ag_task']} · crop: {r['crop']}"
                            + (f" · TP {int(r['tp'])} FP {int(r['fp'])} FN {int(r['fn'])}" if r["task"] in ("detection", "segmentation") else ""))
                if it.get("prompt"):
                    st.text(it["prompt"][:2000])
                if r["task"] not in ("detection", "segmentation"):
                    st.markdown(f"**Answer:** {r['answer'] or '_(empty)_'}  \n**Ground truth:** {r['gt']}")
                    if r.get("rouge_l") == r.get("rouge_l") and r.get("rouge_l") is not None:
                        st.caption(f"Token F1 {r['token_f1']:.2f} · ROUGE-L {r['rouge_l']:.2f}")
                    if isinstance(r.get("judge_text"), str):
                        st.caption(f"Judge: {r['judge_text'][:500]}")
                else:
                    st.text(r["text"][:1500])
                if r.get("reasoning"):
                    with st.expander(f"Reasoning ({r.get('completion_tokens', 0)} tokens, {r.get('latency_s', 0):.1f}s)"):
                        st.text(r["reasoning"][:8000])
                if r.get("truncated") or r.get("error"):
                    st.warning(r.get("error") or "Truncated at max_tokens")


def page_crops():
    st.title("Crops")
    if no_runs():
        return
    run_dir = runs()[0]["dir"]
    rows = json.loads((Path(run_dir) / "crops.json").read_text()) if (Path(run_dir) / "crops.json").exists() else []
    st.caption("Crop named in each prompt, resolved per sample from dataset metadata. Species ID always uses 'plant'.")
    df = pd.DataFrame([{"task": r["task"], "dataset": r["dataset"],
                        "crops used": ", ".join(f"{k} ({v})" for k, v in r["crops"].items()),
                        "source": ", ".join(f"{k} ({v})" for k, v in r["sources"].items())} for r in rows])
    q = st.text_input("Search")
    if q and len(df):
        df = df[df.dataset.str.contains(q, case=False) | df["crops used"].str.contains(q, case=False)]
    st.dataframe(df, hide_index=True, width="stretch", height=600)


def page_compute():
    from agbench.eval.profile import actual_frames, profile_frames

    st.title("Compute")
    path = Path(args.profile or "") / "profile.json"
    data = json.loads(path.read_text()) if args.profile and path.exists() else None
    prof_dir = Path(args.profile).resolve() if args.profile else None
    real = [r for r in runs() if not (prof_dir and prof_dir in Path(r["dir"]).resolve().parents)]
    act = actual_frames(real, data)
    if data is None and not len(act["models"]):
        st.info("No compute data. Pass `--profile <out dir>` for estimates; runs record their own compute in compute.json.")
        return
    if len(act["models"]):
        st.subheader("Actual runs vs profile estimate")
        st.caption("Estimate: profile throughput (batched) × the items each run actually did. "
                   "Actual: wall time logged by the run (compute.json); per-task hours split by latency share.")
        am = act["models"]
        if am["generate h (est.)"].notna().any():
            st.bar_chart(am.set_index("run")[["generate h (est.)", "generate h", "judge h (est.)", "judge h"]], stack=False, height=280)
        st.dataframe(am, hide_index=True, width="stretch")
        with st.expander("Per task"):
            st.dataframe(act["tasks"], hide_index=True, width="stretch")
    if data is None:
        return
    st.subheader("Profile")
    f = profile_frames(data)
    s = data.get("settings") or {}
    st.caption(f"{f['hw']} · thinking {s.get('thinking')} ({s.get('effort')}) · {s.get('per_task')} items per task batched, "
               f"{s.get('per_task_seq')} sequential")
    m = st.columns(4)
    m[0].metric("Largest VRAM need", f"{f['peak']:.1f} GB")
    m[1].metric("Minimum single GPU", f"{f['gpu']} GB")
    m[2].metric("Full suite", f"{f['total_h']:.1f} GPU-h")
    m[3].metric("On 4 GPUs", f"{f['total_h'] / 4:.1f} h")

    fit = f["fit"]
    st.subheader("Memory")
    if len(fit):
        st.bar_chart(fit.set_index("model")[["needed GB", "peak VRAM GB"]], stack=False, height=280)
        st.dataframe(fit, hide_index=True, width="stretch")

    pt = f["per_task"]
    if len(pt):
        st.subheader("Time per image")
        a, b = st.columns(2)
        a.markdown("**Seconds per image, one at a time (p50)**")
        a.bar_chart(pt.pivot_table(index="task", columns="model", values="s/image (c=1, p50)"), stack=False, height=300)
        b.markdown("**Images per second, batched**")
        b.bar_chart(pt.pivot_table(index="task", columns="model", values="images/s (batched)"), stack=False, height=300)
        st.subheader("Full-run estimate for this data dir")
        st.bar_chart(pt.pivot_table(index="model", columns="task", values="full-run h (batched)"), height=300)
        models_ = st.multiselect("Model", sorted(pt.model.unique()), default=sorted(pt.model.unique()))
        st.dataframe(pt[pt.model.isin(models_)], hide_index=True, width="stretch")
        st.dataframe(f["totals"], hide_index=True, width="stretch")
    if f["judge"]:
        st.subheader("Judge")
        st.dataframe(pd.DataFrame([f["judge"]]), hide_index=True, width="stretch")


pg = st.navigation([st.Page(page_summary, title="Summary", default=True), st.Page(page_datasets, title="Datasets"),
                    st.Page(page_samples, title="Samples"), st.Page(page_boxes, title="Boxes"), st.Page(page_crops, title="Crops"),
                    st.Page(page_compute, title="Compute")])
pg.run()
