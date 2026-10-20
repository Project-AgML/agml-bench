"""Streamlit viewer for the AgML catalog and sampled base-data."""

import argparse
import colorsys
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from agbench.catalog import load_catalog
from agbench.config import CATALOG_DIR, load_config
from agbench.pipeline import TASK_DIRS, dataset_dir, failure_path, status_of

st.set_page_config(page_title="AgML base-data", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
args, _ = parser.parse_known_args()
DATA_DIR = Path(args.data_dir)
THUMB = 480


def palette(n: int) -> list[tuple[int, int, int]]:
    return [tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.75, 1.0)) for i in range(max(n, 1))]


@st.cache_data
def catalog_frame() -> tuple[dict, pd.DataFrame]:
    cat = load_catalog()
    cfg = load_config()
    rows = []
    for e in cat["datasets"].values():
        st_ = status_of(cfg, DATA_DIR, e) if e["task"] in TASK_DIRS else "excluded"
        meta_p = dataset_dir(DATA_DIR, e) / "meta.json" if e["task"] in TASK_DIRS else None
        n = json.loads(meta_p.read_text())["n_samples"] if meta_p and meta_p.exists() else None
        rows.append({
            "name": e["name"], "task": TASK_DIRS.get(e["task"], e["task"] or "unknown"), "status": st_, "samples": n,
            "classes": len(e["classes"]), "crops": ", ".join(map(str, e["crops"]))[:60],
            "platform": ", ".join(map(str, e["platform"])), "location": ", ".join(map(str, e["location"]))[:40],
            "sensor": ", ".join(e["sensor_modality"]), "license": e["license"], "first_seen": e.get("first_seen"),
            "excluded_because": e["exclusion_reason"], "hf_repo": e["hf_repo"],
        })
    return cat, pd.DataFrame(rows)


@st.cache_data
def load_samples(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


@st.cache_data(max_entries=2000)
def render(ds_dir: str, sample_json: str, max_side: int, labels: tuple[str, ...] = ()) -> Image.Image:
    s = json.loads(sample_json)
    im = Image.open(Path(ds_dir) / s["images"][0])
    im = im.convert("RGB") if im.mode != "RGB" else im
    if s.get("mask"):
        im = overlay_mask(im, Image.open(Path(ds_dir) / s["mask"]))
    if s.get("objects"):
        colors = dict(zip(labels, palette(len(labels))))
        d = ImageDraw.Draw(im)
        lw = max(2, im.width // 300)
        for o in s["objects"]:
            x, y, w, h = o["bbox"]
            d.rectangle([x, y, x + w, y + h], outline=colors.get(o["label"], (255, 0, 0)), width=lw)
    im.thumbnail((max_side, max_side))
    return im


def overlay_mask(im: Image.Image, mask: Image.Image) -> Image.Image:
    mask = mask.resize(im.size, Image.NEAREST)
    if mask.mode in ("RGB", "RGBA"):
        color = np.asarray(mask.convert("RGB"))
        fg = color.sum(axis=2) > 0
    else:
        m = np.asarray(mask.convert("I") if mask.mode not in ("L", "P", "I") else mask).astype(np.int64)
        vals = [v for v in np.unique(m) if v != 0]
        pal = palette(len(vals))
        color = np.zeros((*m.shape, 3), np.uint8)
        for v, c in zip(vals, pal):
            color[m == v] = c
        fg = m != 0
    base = np.asarray(im).astype(np.float32)
    base[fg] = 0.55 * base[fg] + 0.45 * color[fg]
    return Image.fromarray(base.astype(np.uint8))


# ---------------------------------------------------------------- catalog page

def page_catalog():
    cat, df = catalog_frame()
    st.title("AgML catalog")
    st.caption(f"Website snapshot {cat['website_sha'][:7]} · synced {cat['generated']} · data dir `{DATA_DIR}`")

    el = df[df.status != "excluded"]
    cols = st.columns(6)
    cols[0].metric("Catalog entries", len(df))
    cols[1].metric("Eligible", len(el))
    for c, s in zip(cols[2:], ["done", "new", "failed"]):
        c.metric(s.capitalize(), int((el.status == s).sum()))
    cols[5].metric("Stale", int(el.status.str.startswith("stale").sum()))

    st.subheader("Eligible datasets by task")
    st.bar_chart(el.groupby(["task", "status"]).size().unstack(fill_value=0))

    f1, f2, f3, f4 = st.columns([2, 2, 2, 3])
    tasks = f1.multiselect("Task", sorted(df.task.unique()), default=[t for t in TASK_DIRS.values() if t in set(df.task)])
    statuses = f2.multiselect("Status", sorted(df.status.unique()), default=[s for s in df.status.unique() if s != "excluded"])
    seen = sorted(df.first_seen.dropna().unique())
    since = f3.selectbox("First seen on/after", ["any"] + seen[::-1])
    q = f4.text_input("Search name / crop")
    view = df[df.task.isin(tasks) & df.status.isin(statuses)]
    if since != "any":
        view = view[view.first_seen >= since]
    if q:
        view = view[view.name.str.contains(q, case=False) | view.crops.str.contains(q, case=False)]
    st.dataframe(view, hide_index=True, width="stretch", height=520)

    failed = el[el.status == "failed"]
    if len(failed):
        with st.expander(f"Failures ({len(failed)})"):
            for n in failed.name:
                st.text(f"{n}: {json.loads(failure_path(DATA_DIR, n).read_text())['reason']}")

    hist_p = CATALOG_DIR / "dataset_history.json"
    if hist_p.exists():
        pts = json.loads(hist_p.read_text()).get("points", [])
        if pts:
            st.subheader("AgML growth")
            h = pd.DataFrame(pts).set_index("date")[["datasetCount"]]
            st.line_chart(h)


# ---------------------------------------------------------------- dataset page

def page_dataset():
    cat, df = catalog_frame()
    done = df[df.samples.notna()]
    if done.empty:
        st.info("No sampled datasets yet. Run `agbench sample --data-dir ...`.")
        return
    c1, c2 = st.columns([1, 3])
    task = c1.selectbox("Task", sorted(done.task.unique()))
    name = c2.selectbox("Dataset", sorted(done[done.task == task].name, key=str.lower))
    entry = cat["datasets"][name]
    ds_dir = dataset_dir(DATA_DIR, entry)
    meta = json.loads((ds_dir / "meta.json").read_text())
    samples = load_samples(str(ds_dir / "samples.jsonl"))

    st.title(name)
    m = st.columns(5)
    m[0].metric("Samples", len(samples))
    m[1].metric("Rows indexed", meta["info"].get("indexed_rows", "–"))
    m[2].metric("Catalog classes", len(entry["classes"]))
    m[3].metric("Task", TASK_DIRS[entry["task"]])
    m[4].metric("HF revision", meta["hf_sha"][:7])

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"[{entry['hf_repo']}](https://huggingface.co/datasets/{entry['hf_repo']}) · license `{entry['license']}`"
                    + (f" · [docs]({entry['documentation']})" if entry.get("documentation") else ""))
        meta_fields = {k: entry[k] for k in ("agricultural_task", "crops", "location", "platform", "environment",
                                             "sensor_modality", "real_or_synthetic", "annotation_format", "num_images")}
        st.table(pd.DataFrame({"field": meta_fields.keys(), "value": [str(v) for v in meta_fields.values()]}))
        if entry.get("itt"):
            st.json(entry["itt"], expanded=False)
    with right:
        if entry["task"] == "image_text_to_text":
            counts = Counter(" | ".join(str(v) for v in s.get("stratum", {}).values()) for s in samples)
            st.markdown(f"**Strata** ({', '.join(meta.get('strata_keys') or [])})")
        else:
            counts = Counter(k for s in samples for k in s.get("keys", []))
            st.markdown("**Images per class**")
        st.dataframe(pd.DataFrame(counts.most_common(), columns=["group", "samples"]), hide_index=True,
                     width="stretch", height=260)
    with st.expander("Citation"):
        st.code(entry.get("citation") or "–", language="bibtex")
    with st.expander("Sampling metadata"):
        st.json({k: v for k, v in meta.items() if k != "catalog"}, expanded=False)

    st.divider()
    if entry["task"] == "image_text_to_text":
        show_itt(ds_dir, samples)
    else:
        show_grid(ds_dir, samples, entry)


def show_grid(ds_dir: Path, samples: list[dict], entry: dict):
    classes = sorted({k for s in samples for k in s.get("keys", [])})
    f1, f2 = st.columns([3, 1])
    pick = f1.multiselect("Filter by class", classes)
    ncols = f2.slider("Columns", 2, 8, 4)
    shown = [s for s in samples if not pick or set(pick) & set(s.get("keys", []))]
    labels = tuple(classes)
    if entry["task"] == "object_detection":
        st.markdown(" ".join(f"<span style='color:rgb{c}'>■</span> {l}" for l, c in zip(labels, palette(len(labels)))),
                    unsafe_allow_html=True)
    cols = st.columns(ncols)
    for i, s in enumerate(shown):
        with cols[i % ncols]:
            img = render(str(ds_dir), json.dumps(s), THUMB, labels)
            st.image(img, width="stretch")
            if entry["task"] == "image_classification":
                cap = s["label"]
            elif entry["task"] == "object_detection":
                c = Counter(o["label"] for o in s.get("objects", []))
                cap = ", ".join(f"{k}×{v}" for k, v in c.most_common())
            else:
                cap = ", ".join(s.get("classes_present", []))
            st.caption(f"{cap} · {s['split']}")


def show_itt(ds_dir: Path, samples: list[dict]):
    strata = sorted({" | ".join(str(v) for v in s.get("stratum", {}).values()) for s in samples})
    f1, f2 = st.columns([3, 1])
    pick = f1.multiselect("Filter by stratum", strata)
    page_size = f2.selectbox("Per page", [10, 25, 50], index=0)
    shown = [s for s in samples if not pick or " | ".join(str(v) for v in s.get("stratum", {}).values()) in pick]
    pages = max(1, (len(shown) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, pages, 1)
    for s in shown[(page - 1) * page_size: page * page_size]:
        with st.container(border=True):
            a, b = st.columns([1, 2])
            with a:
                for rel in s["images"][:4]:
                    p = ds_dir / rel
                    im = Image.open(p)
                    im.thumbnail((THUMB, THUMB))
                    st.image(im.convert("RGB"), width="stretch")
                if len(s["images"]) > 4:
                    st.caption(f"+{len(s['images']) - 4} more images")
            with b:
                st.caption(" · ".join(f"{k}: {v}" for k, v in s.get("stratum", {}).items()) + f" · {s['source_id']}")
                for msg in s.get("messages") or []:
                    text = "\n".join(c.get("text") or "" for c in msg.get("content") or [] if c.get("type") == "text")
                    if text.strip():
                        st.markdown(f"**{msg['role']}**")
                        st.text(text[:3000])
                with st.expander("raw_metadata"):
                    st.json(s.get("raw_metadata"), expanded=False)


pg = st.navigation([st.Page(page_catalog, title="Catalog", default=True), st.Page(page_dataset, title="Dataset")])
pg.run()
