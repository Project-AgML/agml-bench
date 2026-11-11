"""Figures 3 and 4: LLM-judge examples and the judge prompt, styled after the v1 paper's judge figures.

fig_judge_samples.png: three OEQ answers from runs/rejudge_v4 seed 0 (models other than the judge) with the judge's
verdict, reasoning and confidence, next to the fuzzy-match verdict (the first two are drawn).
fig_judge_prompt.png: the judge system prompt and classification prompt from src/agbench/eval/prompts.yaml.

usage: uv run --with matplotlib --with pillow --with numpy --with pyyaml python paper/scripts/make_judge.py
"""
import json
import os
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
A = P / "assets" / "judge"
RUN = ROOT / os.environ.get("AGBENCH_SEEDS", "runs/rejudge_v4/exp/seeds") / "s0"
INK = "#111111"
FRAME = "#3b6fb6"       # v1 panel border blue
KEY = "#d9752b"         # v1 orange for the predicted condition
RIGHT = "#16a34a"
WRONG = "#dc2626"
SERIF = ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "DejaVu Serif"]

# (model run prefix, item id, short model name, key phrase in the answer)
SAMPLES = [
    ("openbmb__MiniCPM-V-4.7-35B-A3B-FP8", "vegetable_classification_bangladesh_classification-9d94fe358e:oeq", "MiniCPM-V-4.7", "okra"),
    ("nvidia__NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF", "tomato_leaf_disease-ab65753521:oeq", "Nemotron-3.5-Bolt", "early blight"),
    ("Qwen__Qwen3.8-27B-FP8", "sugarcane_leaf_disease_classification-edf8addb22:oeq", "Qwen3.8-27B", "drought (water) stress"),
]
AG = {"disease": "Disease", "pest_damage": "Pest / damage", "species_id": "Species ID", "quality_maturity": "Quality / maturity"}


def load(prefix, item_id):
    d = next(RUN.glob(prefix + "_2026*"))
    for line in open(d / "predictions.jsonl"):
        r = json.loads(line)
        if r["item_id"] == item_id:
            r["verdict"] = json.loads(re.search(r"\{.*\}", r["judge_text"], re.S).group())
            return r
    raise KeyError(item_id)


class Canvas:
    def __init__(self, W, H, fs):
        self.fig = plt.figure(figsize=(W, H), dpi=300)
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0); self.ax.axis("off")
        self.W, self.H, self.fs = W, H, fs
        self.rend = self.fig.canvas.get_renderer()
        self.cache = {}

    def tw(self, s, weight="normal", style="normal", size=None, family=None):
        size = size or self.fs
        key = (s, weight, style, size, family)
        if key not in self.cache:
            t = self.ax.text(0, 0, s, fontsize=size, fontweight=weight, style=style, **({"family": family} if family else {}))
            self.cache[key] = t.get_window_extent(self.rend).width / self.fig.dpi
            t.remove()
        return self.cache[key]

    def flow(self, x, y, width, segs, size=None, lh=1.22, draw=True, center=False):
        """[(text, {weight, style, color})] laid out word by word; returns y below the last line.
        center=True centres each line on x + width / 2."""
        if center:
            lines, cur = [], []
            sp = self.tw("a b", size=size) - self.tw("ab", size=size)
            cx = 0
            for text, st in segs:
                for word in text.split(" "):
                    if not word:
                        continue
                    w = self.tw(word, st.get("weight", "normal"), st.get("style", "normal"), size)
                    if cur and cx + w > width:
                        lines.append((cur, cx - sp)); cur, cx = [], 0
                    cur.append((word, st, w)); cx += w + sp
            lines.append((cur, cx - sp))
            step = (size or self.fs) * lh / 72
            for li, (ws, lw) in enumerate(lines):
                cx = x + (width - lw) / 2
                for word, st, w in ws:
                    if draw:
                        self.ax.text(cx, y + li * step, word, fontsize=size or self.fs, va="top", color=st.get("color", INK),
                                     fontweight=st.get("weight", "normal"), style=st.get("style", "normal"))
                    cx += w + sp
            return y + len(lines) * step
        size = size or self.fs
        step = size * lh / 72
        sp = self.tw("a b", size=size) - self.tw("ab", size=size)
        cx, cy = x, y
        for text, st in segs:
            for word in text.split(" "):
                if not word:
                    continue
                w = self.tw(word, st.get("weight", "normal"), st.get("style", "normal"), size)
                if cx > x and cx + w > x + width:
                    cx, cy = x, cy + step
                if draw:
                    self.ax.text(cx, cy, word, fontsize=size, va="top", color=st.get("color", INK),
                                 fontweight=st.get("weight", "normal"), style=st.get("style", "normal"))
                cx += w + sp
        return cy + step


def mark(cv, x, y, ok):
    if ok:
        s = 0.10
        cv.ax.add_patch(FancyBboxPatch((x + 0.015, y - s / 2 + 0.015), s - 0.03, s - 0.03, boxstyle="round,pad=0.015",
                                       fc=RIGHT, ec="none", zorder=5))
        cv.ax.text(x + s / 2, y + 0.004, "✔", fontsize=6, color="white", ha="center", va="center", family="DejaVu Sans", zorder=6)
    else:
        cv.ax.text(x + 0.05, y, "✖", fontsize=8, color=WRONG, ha="center", va="center", family="DejaVu Sans")


def samples():
    """Two panels as in v1: centred header over a large image on the left, the judge record on the right."""
    plt.rcParams.update({"font.family": "serif", "font.serif": SERIF})
    rows = [load(p, k) for p, k, _, _ in SAMPLES[:2]]
    W, pad, gap, fs = 7.0, 0.04, 0.10, 7.6
    pw = (W - 2 * pad - gap) / 2
    inner = 0.12                          # panel padding
    d = 1.42                              # image side
    lcol = d
    rcol = pw - 2 * inner - lcol - 0.14
    bold = {"weight": "bold"}

    def blocks(r, key, model):
        ans = r["answer"].replace("*", "")          # drop markdown emphasis
        a, b = ans.index(key), ans.index(key) + len(key)
        if ans[b:b + 1] in (",", ".", ")"):
            b += 1                       # keep trailing punctuation with the highlighted phrase
        ok = bool(r["judge_correct"])
        col = RIGHT if ok else WRONG
        head = [[("Model:", bold), (model, {})], [("Task:", bold), (AG.get(r["ag_task"], r["ag_task"]), {})]]
        body = [[("Ground Truth:", bold), (r["gt"], {})],
                [("Generated Text:", bold), (ans[:a], {}), (ans[a:b], {"style": "italic", "color": KEY}), (ans[b:], {})],
                [("Reasoning:", bold), (r["verdict"]["reasoning"], {})],
                [("Confidence:", bold), (str(r["verdict"]["confidence"]), {})],
                [("Judge:", {"weight": "bold", "color": col}), ("Match" if ok else "No match", {"weight": "bold", "color": col})],
                [("Fuzzy match:", bold), ("Match" if r["correct"] else "No match", {})]]
        return head, body

    # measure, then size the figure to the taller panel
    m = Canvas(W, 1, fs)
    meas = []
    for r, (_, _, model, key) in zip(rows, SAMPLES):
        head, body = blocks(r, key, model)
        hh = sum(m.flow(0, 0, lcol, h, size=fs + 0.6, draw=False, center=True) for h in head)
        bh = sum(m.flow(0, 0, rcol, t, draw=False) for t in body) + 0.02 * (len(body) - 1)
        meas.append((hh, bh))
    plt.close(m.fig)
    ph = max(max(hh + 0.08 + d, bh) for hh, bh in meas) + 2 * inner + 0.06
    H = ph + 2 * pad
    cv = Canvas(W, H, fs)
    for i, (r, (_, _, model, key), (hh, bh)) in enumerate(zip(rows, SAMPLES, meas)):
        x0, y0 = pad + i * (pw + gap), pad
        cv.ax.add_patch(Rectangle((x0, y0), pw, ph, fill=False, ec=FRAME, lw=1.1))
        head, body = blocks(r, key, model)
        # left column: header + image, centred vertically
        lx = x0 + inner
        ly = y0 + (ph - (hh + 0.08 + d)) / 2
        for h in head:
            ly = cv.flow(lx, ly, lcol, h, size=fs + 0.6, center=True)
        ly += 0.08
        ax = cv.fig.add_axes([lx / W, 1 - (ly + d) / H, d / W, d / H])
        ax.imshow(Image.open(A / (r["sample_id"] + ".jpg"))); ax.axis("off")
        # right column: the judge record, centred vertically
        rx = lx + lcol + 0.14
        ry = y0 + (ph - bh) / 2
        for t in body:
            ry = cv.flow(rx, ry, rcol, t) + 0.02
    cv.fig.savefig(P / "fig_judge_samples.png", dpi=300, facecolor="white")


def prompt():
    p = yaml.safe_load(open(ROOT / "src/agbench/eval/prompts.yaml"))
    body = p["judge_classification"].replace("{{", "{").replace("}}", "}")
    # line breaks before each rule group (whitespace only; the text is unchanged)
    for head in ("The prediction does NOT", "Separately, a prediction", "Cause versus symptom:", "Neighbouring states:",
                 "Related species, pathogens"):
        body = body.replace(" " + head, "\n\n" + head)
    body = body.replace("{class_list}\nThe prediction", "{class_list}\n\nThe prediction")
    body = body.replace("\nReference class:", "\n\nReference class:").replace("\nRespond with JSON", "\n\nRespond with JSON")
    text = "System: " + p["judge_system"] + "\n\n" + body.strip()
    fs, cols = 6.1, 128
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, cols, break_long_words=False, break_on_hyphens=False) or [""]
    W = 7.16
    lh = fs * 1.32 / 72
    H = 0.12 * 2 + len(lines) * lh
    cv = Canvas(W, H, fs)
    cv.ax.add_patch(Rectangle((0.02, 0.02), W - 0.04, H - 0.04, fc="#f2f2f2", ec="#cccccc", lw=0.5))
    navy = "#1b2a41"
    hl = re.compile(r"(\{class_list\}|\{gt\}|\{pred\}|System:)")
    cw = cv.tw("M" * 40, size=fs, family="DejaVu Sans Mono") / 40     # monospace advance
    for li, line in enumerate(lines):
        y = 0.12 + li * lh
        x = 0.12
        for part in hl.split(line):
            if not part:
                continue
            special = bool(hl.fullmatch(part))
            cv.ax.text(x, y, part, fontsize=fs, family="DejaVu Sans Mono", va="top",
                       color=(KEY if part != "System:" else navy) if special else navy,
                       fontweight="bold" if special else "normal")
            x += len(part) * cw
    cv.fig.savefig(P / "fig_judge_prompt.png", dpi=300, facecolor="white")


if __name__ == "__main__":
    samples()
    prompt()
