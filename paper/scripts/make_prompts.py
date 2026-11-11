"""Figure 2: prompt formats with real responses, styled after the v1 paper's prompt-overview figure.

Six cells (3 x 2), each a prompt, its image, the options and the model's actual response (Gemma-4-31B, seed 0).
Row 1 shows one image under MCQA within, MCQA across and OEQ; row 2 shows a withheld-class MCQA item, grounding
and a numeric VQA question. Option colours: blue = distractor from the same dataset, purple = from other datasets, green = correct.

usage: uv run --with matplotlib --with pillow --with numpy python paper/scripts/make_prompts.py
Reads paper/assets/prompts/{cells.json,*.jpg} (made by scripts/extract_prompt_examples.py); writes paper/fig_prompts.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image

P = Path(__file__).resolve().parents[1]
A = P / "assets" / "prompts"
INK = "#111111"
WITHIN = "#1f5f8b"      # v1 "within species" blue
ACROSS = "#9b2d9b"      # v1 "across species" purple
RIGHT = "#16a34a"       # correct answer
WRONG = "#dc2626"
PRED = "#ffb000"        # predicted boxes
FS = 7.2
MODEL = "Gemma-4-31B"

TITLES = {"mcqa_within": "MCQA within prompt:", "mcqa_across": "MCQA across prompt:", "oeq": "OEQ prompt:",
          "grounding": "Grounding prompt:", "native": "VQA prompt:"}


def main():
    cells = json.load(open(A / "cells.json"))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                         "font.size": FS})
    W, pad, ncol, nrow = 7.16, 0.08, 3, 2
    cw = (W - 2 * pad) / ncol
    ch = 2.12
    leg_gap, leg = 0.10, 0.30
    H = 2 * pad + nrow * ch + leg_gap + leg
    fig = plt.figure(figsize=(W, H), dpi=300)
    fig.patch.set_facecolor("white")
    base = fig.add_axes([0, 0, 1, 1])
    base.set_xlim(0, W); base.set_ylim(H, 0); base.axis("off")
    rend = fig.canvas.get_renderer()
    lh = FS * 1.22 / 72            # line height, inches

    cache = {}

    def tw(s, weight="normal", style="normal", size=FS):
        key = (s, weight, style, size)
        if key not in cache:
            t = base.text(0, 0, s, fontsize=size, fontweight=weight, style=style)
            cache[key] = t.get_window_extent(rend).width / fig.dpi
            t.remove()
        return cache[key]

    def flow(x, y, width, segs, size=FS, draw=True):
        """Lay out [(text, {weight, style, color})] word by word, wrapping at width.
        Returns (y below the last line, x after the last word)."""
        sp = tw("a b", size=size) - tw("ab", size=size)
        cx, cy = x, y
        for text, st in segs:
            for word in text.split(" "):
                if not word:
                    continue
                w = tw(word, st.get("weight", "normal"), st.get("style", "normal"), size)
                if cx > x and cx + w > x + width:
                    cx, cy = x, cy + lh * size / FS
                if draw:
                    base.text(cx, cy, word, fontsize=size, va="top", color=st.get("color", INK),
                              fontweight=st.get("weight", "normal"), style=st.get("style", "normal"))
                cx += w + sp
        return cy + lh * size / FS, cx

    def image(x, y, d, c):
        ax = fig.add_axes([x / W, 1 - (y + d) / H, d / W, d / H])
        ax.imshow(Image.open(A / c["file"]), extent=(0, 600, 600, 0))
        clip = Circle((300, 300), 299, transform=ax.transData)
        ax.images[0].set_clip_path(clip)
        for b in c["boxes"]:
            x0, y0, x1, y1 = b["xyxy"]
            if b["kind"] == "gt":
                sty = [(2.0, "black", "-"), (0.9, "white", "-")]
            else:
                sty = [(2.0, "black", (0, (3, 2))), (1.1, PRED, (0, (3, 2)))]
            for lw, col, ls in sty:
                r = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=lw, ec=col, ls=ls)
                ax.add_patch(r); r.set_clip_path(clip)
        ax.add_patch(Circle((300, 300), 299, fill=False, lw=0.6, ec="#888888"))
        ax.set_xlim(0, 600); ax.set_ylim(600, 0); ax.axis("off")

    def mark(x, y, ok):
        """Emoji-style mark: green box with a white tick, or a red cross. (x, y) = left, vertical centre."""
        if ok:
            s = 0.105
            base.add_patch(FancyBboxPatch((x + 0.015, y - s / 2 + 0.015), s - 0.03, s - 0.03, boxstyle="round,pad=0.015",
                                          fc=RIGHT, ec="none", zorder=5))
            base.text(x + s / 2, y + 0.004, "✔", fontsize=6.2, color="white", ha="center", va="center",
                      family="DejaVu Sans", zorder=6)
        else:
            base.text(x + 0.05, y, "✖", fontsize=8.5, color=WRONG, ha="center", va="center", family="DejaVu Sans")

    def options(x, y, items, maxw):
        """items: [(label, colour)] drawn with an open checkbox in the label's colour; font shrinks to fit maxw."""
        s = 0.075
        fs = FS
        while fs > 5.5 and max(tw(lab, size=fs) for lab, _ in items) + s + 0.06 > maxw:
            fs -= 0.1
        for lab, col in items:
            base.add_patch(Rectangle((x, y + lh / 2 - s / 2 - 0.004), s, s, fill=False, ec=col, lw=0.8))
            base.text(x + s + 0.06, y + lh / 2, lab, fontsize=fs, color=col, va="center")
            y += lh * 1.12
        return y

    def response(x, top, bottom, width, text, ok, extra=None):
        """'Response: text' (+ mark) and an optional smaller note, centred vertically between top and bottom."""
        segs = [("Response:", {"style": "italic"}), (text, {})]
        wr = width - (0.14 if ok is not None else 0)
        h, _ = flow(0, 0, wr, segs, draw=False)
        if extra:
            h = h + 0.02 + flow(0, 0, width, extra, size=FS - 0.6, draw=False)[0]
        y = top + (bottom - top - h) / 2
        ty, tx = flow(x, y, wr, segs)
        if ok is not None:
            mark(tx, ty - lh / 2 - 0.005, ok)
        if extra:
            flow(x, ty + 0.02, width, extra, size=FS - 0.6)

    def prompt_segs(c):
        fmt = c["format"]
        crop = f"{c['crop']} plant"
        title = "MCQA within prompt (true class withheld):" if c["is_nota"] else TITLES[fmt]
        bold, it, grey = {"weight": "bold"}, {"style": "italic"}, {"color": "#666666"}
        if fmt.startswith("mcqa"):
            return [(title, bold), ("\u201cClassify this image of a", {}), (crop, it),
                    ("into one of the following categories:", {}), ("[options].", grey),
                    ("Respond with ONLY the category name, nothing else.\u201d", {})]
        if fmt == "oeq":
            return [(title, bold), ("\u201cRespond in one sentence: What disease, pest, damage type, or other stress, if any, "
                                    "is exhibited in this image of a", {}), (crop + "?\u201d", it)]
        if fmt == "grounding":
            return [(title, bold), ("\u201cIdentify and localize every", {}), ("full ripened, half ripened, green", it),
                    ("visible in this image of a", {}), (crop + ".", it),
                    ("Respond with ONLY a JSON array \u2026 [box, label] per instance \u2026\u201d", {})]
        q = c["prompt"].split("\n")[0].replace("?(", "? (")
        return [(title, bold), ("\u201c" + q + "\u201d", {}), ("+ \u201cRespond with ONLY the number.\u201d", grey)]

    # images in a row share one top edge, below the tallest prompt of that row
    iw_ = cw - 0.12
    row_top = [max(flow(0, 0, iw_, prompt_segs(c), draw=False)[0] for c in cells[r * ncol:(r + 1) * ncol]) for r in range(nrow)]
    D = 0.98

    for i, c in enumerate(cells):
        r, k = divmod(i, ncol)
        x0, y0 = pad + k * cw, pad + r * ch
        base.add_patch(Rectangle((x0, y0), cw, ch, fill=False, ec=INK, lw=0.8))
        ix, iw = x0 + 0.06, cw - 0.12
        fmt = c["format"]
        flow(ix, y0 + 0.06, iw, prompt_segs(c))
        iy = y0 + 0.06 + row_top[r] + 0.06
        d = D
        bottom = y0 + ch
        if fmt in ("oeq", "native"):       # no options: image centred
            image(x0 + (cw - d) / 2, iy, d, c)
            if fmt == "oeq":
                extra = [("Judge: incorrect; ground truth", {"style": "italic"}), (c["gt_label"] + ".", {"weight": "bold"})]
                response(ix, iy + d, bottom, iw, c["response"], bool(c["judge_correct"]), extra)
            else:
                gt = f"{float(c['answer']):g}"
                extra = [("Ground truth", {"style": "italic"}), (gt + " m.", {"weight": "bold"}),
                         ("Numeric answers within 5% count as correct.", {"style": "italic"})]
                response(ix, iy + d, bottom, iw, c["response"], bool(c["correct"]), extra)
            continue
        image(ix + 0.02, iy, d, c)
        ox = ix + d + 0.14
        if fmt.startswith("mcqa"):
            dis = ACROSS if fmt == "mcqa_across" else WITHIN
            items = []
            for o in c["choices"]:
                if o == c["answer"]:
                    items.append((o, RIGHT))
                elif o in ("None of the above", "I don't know"):
                    items.append((o, INK))
                else:
                    items.append((o, dis))
            n = len(items)
            options(ox, iy + d / 2 - n * lh * 1.12 / 2, items, x0 + cw - 0.05 - ox)
            extra = None
            if c["is_nota"]:
                extra = [("True class", {"style": "italic"}), ("(Downy Mildew)", {"weight": "bold"}),
                         ("removed from the options; NOTA is correct.", {"style": "italic"})]
            response(ix, iy + d, bottom, iw, c["response"], bool(c["correct"]), extra)
        else:  # grounding
            s_ = 0.20
            yb = iy + d / 2 - 0.17
            for kind, lab in (("gt", "Ground truth (6)"), ("pred", "Predicted (5)")):
                sty = [(1.6, "black", "-"), (0.7, "white", "-")] if kind == "gt" else [(1.6, "black", (0, (3, 2))), (0.9, PRED, (0, (3, 2)))]
                for lw, col, ls in sty:
                    base.add_patch(Rectangle((ox, yb), s_, 0.12, fill=False, lw=lw, ec=col, ls=ls))
                base.text(ox + s_ + 0.06, yb + 0.06, lab, fontsize=FS, va="center")
                yb += 0.22
            response(ix, iy + d, bottom, iw, f"{len([b for b in c['boxes'] if b['kind'] == 'pred'])} boxes; "
                     f"{c['tp']} match the ground truth.", None)

    # legend
    ly = pad + nrow * ch + leg_gap
    lx0 = pad + 0.04
    lwid = W - 2 * pad - 0.08
    lhgt = leg - 0.04
    mid = ly + lhgt / 2
    base.add_patch(Rectangle((lx0, ly), lwid, lhgt, fill=False, ec=INK, lw=0.6))
    sw = 0.13
    parts = [("title", "Legend:", tw("Legend:", "bold", size=8.5), 0.2),
             ("sw", ("Distractor, same dataset", WITHIN, "#cfe6f5"), sw + 0.06 + tw("Distractor, same dataset"), 0.18),
             ("sw", ("Distractor, other datasets", ACROSS, "#f2d3f2"), sw + 0.06 + tw("Distractor, other datasets"), 0.18),
             ("sw", ("Correct answer", RIGHT, "#d3f2dc"), sw + 0.06 + tw("Correct answer"), 0)]
    total = sum(w + g for _, _, w, g in parts)
    x = lx0 + (lwid - total) / 2
    for kind, val, w, g in parts:
        if kind == "title":
            base.text(x, mid, val, fontweight="bold", fontsize=8.5, va="center_baseline")
        else:
            lab, dark, light = val
            base.add_patch(Rectangle((x - 0.03, mid - 0.075), w + 0.09, 0.15, fc=light, ec="none"))
            base.add_patch(Rectangle((x - 0.03, mid - 0.075), sw, 0.15, fc=dark, ec="none"))
            base.text(x + sw + 0.03, mid, lab, fontsize=FS, va="center_baseline")
        x += w + g
    fig.savefig(P / "fig_prompts.png", dpi=300, facecolor="white")


if __name__ == "__main__":
    main()
