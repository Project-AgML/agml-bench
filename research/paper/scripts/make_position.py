"""Figure: position preference under option-order randomization (Experiment 2).

For each answer position: share of answers there minus share of items whose correct answer is there (points).
0 = the model picks that position exactly as often as it should.

Reads research/paper/numbers.json ("position", written by make_assets.py); writes research/paper/fig_position.png.
usage: uv run --with matplotlib python research/paper/scripts/make_position.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
NAMES = ["Qwen3.6-35B-A3B", "Qwen3.8-27B", "Gemma-4-31B", "Gemma-4-12B", "Muse-Glimmer-30B",
         "Nemotron-3.5-Bolt-74B-A9B", "MiniCPM-V-4.7-35B-A3B", "Ministral-3-14B"]
COLORS = ["#a6cee3", "#1f78b4", "#fdbf6f", "#ff7f00", "#b2df8a", "#33a02c", "#cab2d6", "#6a3d9a"]  # as make_assets.py
FORMATS = [("mcqa_within", "MCQA within"), ("mcqa_across", "MCQA across"), ("itt_mc", "VQA multiple choice")]
MAXPOS = 7  # VQA MC has up to 12 options, but positions beyond 7 hold <1% of answers
LABEL = {"Ministral-3-14B": "Ministral", "MiniCPM-V-4.7-35B-A3B": "MiniCPM"}


def main():
    pos = json.load(open(P / "numbers.json"))["position"]  # "model/format" -> {position: [chosen %, correct %]}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    for ax, (f, title) in zip(axes, FORMATS):
        ax.axhspan(-2, 2, color="#f1f1f1", zorder=0)
        ax.axhline(0, color="#777777", lw=0.8, zorder=1)
        for name, c in zip(NAMES, COLORS):
            d = pos.get(f"{name}/{f}")
            if not d:
                continue
            xs = sorted(int(k) for k in d if int(k) <= MAXPOS)
            ys = [d[str(x)][0] - d[str(x)][1] for x in xs]
            hi = name in LABEL
            ax.plot(xs, ys, color=c, lw=1.8 if hi else 1.0, marker="o", ms=3 if hi else 2, zorder=3 if hi else 2,
                    label=name + ("$^\\dagger$" if name == "Gemma-4-31B" else ""))
            if hi and abs(ys[0]) > 2.5:
                ax.annotate(LABEL[name], (xs[0], ys[0]), xytext=(5, 0), textcoords="offset points",
                            fontsize=6.5, color="#333333", va="center")
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(1, (6 if f != "itt_mc" else MAXPOS) + 1))
        ax.set_xlabel("Option position", fontsize=7)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Chosen % $-$ correct %\n(points)", fontsize=7)
    axes[0].text(0.03, 0.97, "picked too often", transform=axes[0].transAxes, fontsize=6.3, color="#777777", va="top", style="italic")
    axes[0].text(0.03, 0.03, "avoided", transform=axes[0].transAxes, fontsize=6.3, color="#777777", va="bottom", style="italic")
    axes[1].annotate("no preference", (4.5, 0), xytext=(4.5, -4.2), fontsize=6.3, color="#777777", ha="center", style="italic",
                     arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.3, frameon=False, handlelength=1.5, columnspacing=1.0)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.90, bottom=0.30, wspace=0.12)
    fig.savefig(P / "fig_position.png", dpi=300)


if __name__ == "__main__":
    main()
