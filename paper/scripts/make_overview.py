"""Figure 1: benchmark overview, styled after the v1 paper's dataset figure.

One row per task: name and counts, a vertical bar of datasets per agricultural task, and six example
images (badge = classes in that dataset, or questions for VQA; label colour = agricultural task).
Detection/segmentation tiles show ground-truth boxes; segmentation tiles also show the mask (cyan).

usage: uv run --with matplotlib --with pillow --with numpy python paper/scripts/make_overview.py
Reads paper/assets/overview/{tiles.json,*.jpg,*_mask.png} and the items of one run; writes paper/fig_overview.png.
"""
import collections
import glob
import json
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

P = Path(__file__).resolve().parents[1]
A = P / "assets" / "overview"
RUN = sorted(glob.glob(str(P.parent / os.environ.get("AGBENCH_SEEDS", "runs/rejudge_v4/exp/seeds") / "s0/*/items.jsonl")))[0]

TASKS = [("classification", "Classification"), ("detection", "Detection"),
         ("segmentation", "Segmentation"), ("image_text_to_text", "VQA")]
# pastel label colours as in the v1 figure (disease blue, pest/damage peach, plant/weeds pink), plus three more
AG = [("disease", "Disease", "#a9c9ec"), ("pest_damage", "Pest / damage", "#f7cdb2"),
      ("species_id", "Species ID", "#f48a97"), ("quality_maturity", "Quality / maturity", "#ffe08f"),
      ("localization", "Localization", "#cbbcef"), ("other", "Other", "#d9d9d9")]
AGC = {k: c for k, _, c in AG}
BADGE = "#5a9e32"
MASK = (0.0, 0.9, 1.0, 0.55)
INK = "#111111"


def stats():
    ds, imgs = collections.defaultdict(set), collections.defaultdict(set)
    cls, per_ds_items = {}, collections.Counter()
    ag = collections.defaultdict(lambda: collections.defaultdict(set))
    for line in open(RUN):
        r = json.loads(line)
        t = r["task"]
        ds[t].add(r["dataset"])
        imgs[t].add(r["sample_id"])
        per_ds_items[r["dataset"]] += 1
        if r.get("class_names"):
            cls[(t, r["dataset"])] = len(r["class_names"])
        ag[t][r.get("ag_task") or "other"].add(r["dataset"])
    out = {}
    for t, _ in TASKS:
        out[t] = dict(datasets=len(ds[t]), classes=sum(v for (tt, _), v in cls.items() if tt == t),
                      images=len(imgs[t]), ag={k: len(v) for k, v in ag[t].items()})
    badge = {d: n for (_, d), n in cls.items()}
    for d, n in per_ds_items.items():
        badge.setdefault(d, n)  # VQA: number of questions
    return out, badge


def main():
    S, badge = stats()
    tiles = json.load(open(A / "tiles.json"))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                         "font.size": 8})
    ncol, nrow = 6, len(TASKS)
    W = 7.16
    pad = 0.08                # inside the frame
    head = 0.86               # row header: rotated title and counts, then the vertical bar
    gap = 0.07
    tw = (W - 2 * pad - head - gap * (ncol - 1)) / ncol
    chip = 0.27              # label chip height
    rh = tw + 0.05 + chip + 0.12
    leg = 0.36
    LEG_GAP = 0.10           # space between the last row and the legend box
    H = 2 * pad + nrow * rh + LEG_GAP + leg
    fig = plt.figure(figsize=(W, H), dpi=300)
    fig.patch.set_facecolor("white")
    base = fig.add_axes([0, 0, 1, 1])
    base.set_xlim(0, W); base.set_ylim(H, 0); base.axis("off")   # inches, origin top-left
    base.add_patch(Rectangle((0.015, 0.015), W - 0.03, H - 0.03, fill=False, lw=0.9, ec=INK))

    def rbox(x, y, w, h, fc, ec="none", lw=0.0, r=0.03, z=3):
        base.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}",
                                      fc=fc, ec=ec, lw=lw, zorder=z))

    badges = []
    for ri, (t, tname) in enumerate(TASKS):
        y = pad + ri * rh + 0.06
        s = S[t]
        bh = tw + 0.05 + chip
        # rotated header: task name, then counts, reading bottom-to-top like a y-axis label
        cnt1 = f"{s['datasets']} datasets" + (f" \u00b7 {s['classes']:,} classes" if s["classes"] else "")
        cnt2 = f"{s['images']:,} {'questions' if t == 'image_text_to_text' else 'images'}"
        base.text(pad + 0.13, y + bh / 2, tname, rotation=90, fontsize=10, fontweight="bold", ha="center", va="center", color=INK)
        base.text(pad + 0.33, y + bh / 2, cnt1 + "\n" + cnt2, rotation=90, fontsize=7.2, ha="center", va="center",
                  color=INK, linespacing=1.25, multialignment="center")
        # vertical bar: datasets per agricultural task (a dataset spanning several counts in each)
        bx, bw = pad + head - 0.36, 0.12
        by = y
        tot = sum(s["ag"].values())
        yy = by
        for k, _, c in AG:
            v = s["ag"].get(k, 0)
            if not v:
                continue
            h = bh * v / tot
            base.add_patch(Rectangle((bx, yy), bw, max(h - 0.012, 0.004), fc=c, ec="none", zorder=3))
            if h > 0.085:
                base.text(bx + bw + 0.04, yy + h / 2, str(v), fontsize=6.5, va="center", color=INK)
            yy += h

        order = {k: i for i, (k, _, _) in enumerate(AG)}
        row = sorted([r for r in tiles if r["task"] == t][:ncol], key=lambda r: order.get(r["ag"] or "other", 9))
        for ci, tl in enumerate(row):
            x = pad + head + ci * (tw + gap)
            ax = fig.add_axes([x / W, 1 - (y + tw) / H, tw / W, tw / H])
            ax.imshow(Image.open(A / tl["file"]), extent=(0, 400, 400, 0))
            if tl.get("mask"):
                m = np.asarray(Image.open(A / tl["mask"])) > 0
                rgba = np.zeros((*m.shape, 4)); rgba[m] = MASK
                ax.imshow(rgba, extent=(0, 400, 400, 0), interpolation="nearest")
            for b in tl["boxes"]:
                for lw, col in ((1.4, "black"), (0.6, "white")):
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=lw, ec=col))
            ax.set_xlim(0, 400); ax.set_ylim(400, 0); ax.axis("off")
            # badge, overlapping the image's bottom-left corner as in v1
            n = badge.get(tl["dataset"], "")
            bwid = 0.13 + 0.045 * len(str(n))
            badges.append((x + 0.05, y + tw - 0.17, bwid, str(n)))
            # label chip
            cy = y + tw + 0.05
            txt = tl["caption"]
            fs = 7.4
            if t == "image_text_to_text":
                txt, fs = textwrap.fill("\u201c" + txt + "\u201d", 26), 6.4
            # chip fitted to its text (bold), centred under the image
            base.text(x + tw / 2, cy + chip / 2, txt, ha="center", va="center", fontsize=fs, color=INK, zorder=6,
                      fontweight="bold", style="italic" if t == "image_text_to_text" else "normal", linespacing=1.05,
                      bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.6", fc=AGC.get(tl["ag"] or "other"), ec="none"))
        if ri < nrow - 1:
            base.plot([pad, W - pad], [y + rh - 0.08] * 2, color="#cccccc", lw=0.5)

    # badges go on a layer above the image axes, overlapping each image's bottom-left corner as in v1
    top = fig.add_axes([0, 0, 1, 1], zorder=10)
    top.set_xlim(0, W); top.set_ylim(H, 0); top.axis("off"); top.patch.set_alpha(0)
    for bx_, by_, bw_, n in badges:
        r = 0.025
        top.add_patch(FancyBboxPatch((bx_ + r, by_ + r), bw_ - 2 * r, 0.17 - 2 * r, boxstyle=f"round,pad={r}", fc="white", ec=BADGE, lw=0.8))
        top.text(bx_ + bw_ / 2, by_ + 0.085, n, ha="center", va="center", fontsize=7, color=INK)

    # legend box
    ly = pad + nrow * rh + LEG_GAP
    lx0 = pad + 0.04
    lh = leg - 0.06
    mid = ly + lh / 2
    base.add_patch(Rectangle((lx0, ly), W - 2 * pad - 0.08, lh, fill=False, ec=INK, lw=0.6))
    rend = fig.canvas.get_renderer()
    fs, padf = 7.4, 0.55

    def width(label, size, bold=True):  # rendered text width in inches
        t_ = base.text(0, 0, label, fontsize=size, fontweight="bold" if bold else "normal")
        w_ = t_.get_window_extent(rend).width / fig.dpi
        t_.remove()
        return w_

    chip_pad = padf * fs / 72
    note = "Classes in dataset\n(VQA: questions)"
    parts = [("title", "Legend", width("Legend", 8.5), 0.22)]
    parts += [("chip", (n, c), width(n, fs) + 2 * chip_pad, 0.10) for _, n, c in AG]
    parts[-1] = parts[-1][:3] + (0.18,)
    parts += [("badge", "#", width("#", fs) + 2 * chip_pad, 0.08), ("note", note, width(note, 6.8), 0)]
    total = sum(w + g for _, _, w, g in parts)
    x = lx0 + (W - 2 * pad - 0.08 - total) / 2   # centred in the legend box
    for kind, val, w, g in parts:
        if kind == "title":
            base.text(x, mid, val, fontweight="bold", fontsize=8.5, va="center")
        elif kind == "note":
            base.text(x, mid, val, va="center", fontsize=6.8, fontweight="bold", linespacing=1.0)
        else:
            label, kw = (val[0], dict(fc=val[1], ec="none")) if kind == "chip" else (val, dict(fc="white", ec=BADGE, lw=0.8))
            base.text(x + w / 2, mid, label, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=6,
                      bbox=dict(boxstyle=f"round,pad={padf},rounding_size=0.6", **kw))
        x += w + g
    fig.savefig(P / "fig_overview.png", dpi=300, facecolor="white")


if __name__ == "__main__":
    main()
