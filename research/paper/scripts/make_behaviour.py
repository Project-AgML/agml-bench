"""Figure: NOTA behaviour at the default options (seed 0).

(a) false NOTA (x) vs NOTA recall (y) per model, arrow from MCQA within to MCQA across.
(b) where each model's answers go on withheld-class items: NOTA / a listed class (over-commit) / IDK.

Reads research/paper/numbers.json ("behaviour", written by make_assets.py); writes research/paper/fig_behaviour.png.
usage: uv run --with matplotlib python research/paper/scripts/make_behaviour.py [--horizontal]
       panel (b) is drawn as upright bars; --horizontal draws the earlier horizontal version to fig_behaviour_horizontal.png.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

P = Path(__file__).resolve().parents[1]
COLORS = ["#a6cee3", "#1f78b4", "#fdbf6f", "#ff7f00", "#b2df8a", "#33a02c", "#cab2d6", "#6a3d9a"]  # as make_assets.py
OUT = [("NOTA (correct)", "#4d9f6b"), ("Listed class (over-commit)", "#d1615d"), ("I don't know", "#b8b8b8")]


def label(m):
    return m + ("$^\\dagger$" if m == "Gemma-4-31B" else "")


def scatter(ax, beh):
    lo, hi = 10, 80
    ax.plot([lo, hi], [lo, hi], color="#bbbbbb", lw=0.8, ls="--", zorder=1)
    ax.text(47, 44.5, "recall = false NOTA", rotation=38, fontsize=6.5, color="#888888", ha="center", va="center")
    ax.annotate("ideal", xy=(lo + 1, hi - 1), fontsize=7, color="#888888", ha="left", va="top", style="italic")
    for (m, r), c in zip(beh.items(), COLORS):
        (xw, yw), (xa, ya) = (r["mcqa_within"][3], r["mcqa_within"][1]), (r["mcqa_across"][3], r["mcqa_across"][1])
        ax.annotate("", xy=(xa, ya), xytext=(xw, yw), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, shrinkA=3.5, shrinkB=3.5, mutation_scale=7))
        ax.scatter([xw], [yw], s=30, color=c, edgecolor="#333333", lw=0.5, zorder=3)
        ax.scatter([xa], [ya], s=30, facecolor="white", edgecolor=c, lw=1.4, zorder=3)
        ax.scatter([], [], s=30, color=c, edgecolor="#333333", lw=0.5, label=label(m))
    ax.set_xlim(lo, 55); ax.set_ylim(30, hi)
    ax.set_xlabel("False NOTA (%): true class listed", fontsize=7)
    ax.set_ylabel("NOTA recall (%): true class withheld", fontsize=7)
    ax.grid(color="#eeeeee", lw=0.5); ax.set_axisbelow(True)
    ax.scatter([], [], s=30, color="#999999", edgecolor="#333333", lw=0.5, label="MCQA within")
    ax.scatter([], [], s=30, facecolor="white", edgecolor="#999999", lw=1.4, label="MCQA across")
    ax.set_title("(a) NOTA recall vs. false NOTA", fontsize=8, loc="left")


def stacked(ax, beh):
    names = list(beh)
    ys, ticks, tlabels = [], [], []
    y = 0
    for m in names:
        for fmt, short in (("mcqa_within", "within"), ("mcqa_across", "across")):
            rec, oc = beh[m][fmt][1], beh[m][fmt][2]
            vals = [rec, oc, max(100 - rec - oc, 0)]
            left = 0
            for v, (_, c) in zip(vals, OUT):
                ax.barh(y, v, left=left, height=0.8, color=c, edgecolor="white", lw=0.6)
                left += v
            ax.text(-1.5, y, short, ha="right", va="center", fontsize=5.6, color="#555555")
            ys.append(y); y += 1
        ticks.append(y - 1.5); tlabels.append(label(m))
        y += 0.6
    ax.set_yticks(ticks); ax.set_yticklabels(tlabels, fontsize=6.2)
    ax.tick_params(axis="y", length=0, pad=26)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of withheld-class items (%)", fontsize=7)
    ax.set_title("(b) Answers when the true class is withheld", fontsize=8, loc="right")


def stacked_vertical(ax, beh):
    names = list(beh)
    x, ticks, tlabels, w = 0.0, [], [], 0.42
    for m in names:
        for k, (fmt, short) in enumerate((("mcqa_within", "W"), ("mcqa_across", "A"))):
            rec, oc = beh[m][fmt][1], beh[m][fmt][2]
            vals = [rec, oc, max(100 - rec - oc, 0)]
            xx = x + (k - 0.5) * (w + 0.04)
            bottom = 0
            for v, (_, c) in zip(vals, OUT):
                ax.bar(xx, v, bottom=bottom, width=w, color=c, edgecolor="white", lw=0.6)
                bottom += v
            ax.text(xx, -2.5, short, ha="center", va="top", fontsize=5.6, color="#555555")
        ticks.append(x); tlabels.append(label(m))
        x += 1.25
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.2, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=11)
    ax.set_xlim(-0.6, x - 0.65)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of withheld-class items (%)", fontsize=7)
    ax.set_title("(b) Answers when the true class is withheld", fontsize=8, loc="left")


def main():
    beh = json.load(open(P / "numbers.json"))["behaviour"]  # model -> fmt -> [acc, nota_rec, over_commit, false_nota, idk]
    vertical = "--horizontal" not in sys.argv
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 4.4 if vertical else 3.9),
                               gridspec_kw={"width_ratios": [1, 1.15] if vertical else [1, 0.85], "wspace": 0.3 if vertical else 0.78})
    scatter(a, beh)
    (stacked_vertical if vertical else stacked)(b, beh)
    for ax in (a, b):
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if not vertical:
        b.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.36 if vertical else 0.30)
    h, l = a.get_legend_handles_labels()
    a.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.36 if vertical else -0.17), ncol=2, fontsize=6, frameon=False,
             handletextpad=0.3, columnspacing=0.8)
    if vertical:
        b.legend([Patch(color=c) for _, c in OUT] + [Patch(color="none")], [n for n, _ in OUT] + ["W = MCQA within, A = MCQA across"],
                 loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, fontsize=6.3, frameon=False, handlelength=1.2)
    else:
        b.legend([Patch(color=c) for _, c in OUT], [n for n, _ in OUT], loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1, fontsize=6.3, frameon=False, handlelength=1.2)
    fig.savefig(P / ("fig_behaviour.png" if vertical else "fig_behaviour_horizontal.png"), dpi=300)


if __name__ == "__main__":
    main()
