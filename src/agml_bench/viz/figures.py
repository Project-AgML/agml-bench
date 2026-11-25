"""The paper's result figures for any set of runs (matplotlib; uv sync --extra viz).

    main_results(entries, "fig_main_results.png")   accuracy per task, 95% CI over datasets; YOLO hatched; chance lines
    agtask(entries, "fig_agtask.png")               classification by agricultural task, per question format
"""

from pathlib import Path

from agml_bench.viz.aggregate import AG, CLS_FORMATS, COLS, Entry

# Categorical slots in fixed order (validated for colour-vision deficiency; contrast relief: legend + tables)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
YOLO_STYLE = {"facecolor": "#d9d9d9", "edgecolor": "#555555", "hatch": "////", "lw": 0.6}
MAX_MODELS = len(PALETTE)


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("figures need matplotlib: uv sync --extra viz") from e
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    return plt


def _split(entries: list[Entry]):
    models = [e for e in entries if e.kind == "model"]
    if len(models) > MAX_MODELS:
        raise ValueError(f"{len(models)} models: a figure shows at most {MAX_MODELS} (pass fewer runs)")
    return models, [e for e in entries if e.kind == "yolo"][:1], [e for e in entries if e.kind == "chance"][:1]


def _style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _err(cells):
    return [[c[0] / 100 - c[1] / 100 for c in cells], [c[2] / 100 - c[0] / 100 for c in cells]]


ERR = {"elinewidth": 0.6, "capsize": 1.2, "ecolor": "#444444"}


def main_results(entries: list[Entry], path, cols=None) -> Path:
    """Accuracy per CV task (macro over datasets, 95% bootstrap CI): one bar per model, YOLO hatched (top-1 accuracy
    in the classification columns), chance as a dashed line over each group."""
    plt = _plt()
    models, yolo, chance = _split(entries)
    cols = [c for c in (cols or COLS) if any(e.cells[c[0]][3] for e in models + yolo)]
    nb = len(models) + len(yolo)
    w = 0.8 / max(nb, 1)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, e in enumerate(models + yolo):
        js = [j for j, (k, _) in enumerate(cols) if e.cells[k][3]]
        xs = [j + (i - nb / 2 + 0.5) * w for j in js]
        cells = [e.cells[cols[j][0]] for j in js]
        style = YOLO_STYLE if e.kind == "yolo" else {"color": PALETTE[i]}
        label = e.label + (" (top-1 for classification)" if e.kind == "yolo" else "")
        ax.bar(xs, [c[0] / 100 for c in cells], w * 0.92, label=label, yerr=_err(cells), error_kw=ERR, **style)
    for e in chance:
        for j, (k, _) in enumerate(cols):
            if e.cells[k][3]:
                ax.hlines(e.cells[k][0] / 100, j - 0.43, j + 0.43, colors="#222222", linestyles=[(0, (3, 2))], lw=1.1, zorder=5,
                          label="Chance (random answer)" if j == 0 else None)
    n = {k: max((e.cells[k][3] for e in models + yolo), default=0) for k, _ in cols}
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{h}\n(n={n[k]})" for k, h in cols])
    ax.set_ylabel("Accuracy (avg. across datasets)")
    ax.set_xlabel("Task (n = number of datasets)")
    ax.set_ylim(0, 1)
    _style(ax)
    h, lab = ax.get_legend_handles_labels()
    k = [i for i, x in enumerate(lab) if x.startswith("Chance")]  # chance last
    h, lab = [h[i] for i in range(len(h)) if i not in k] + [h[i] for i in k], [lab[i] for i in range(len(lab)) if i not in k] + [lab[i] for i in k]
    ncol = 3 if max(map(len, lab), default=0) > 30 else 4
    fig.legend(h, lab, loc="lower center", ncol=ncol, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06 + 0.045 * -(-len(lab) // ncol), 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def agtask(entries: list[Entry], path) -> Path:
    """Classification accuracy by agricultural task, one panel per question format; OEQ panel also shows the
    fuzzy-match score (hatched ghost bar, dark tick)."""
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    models, yolo, _ = _split(entries)
    ags = [(a, h) for a, h in AG if any(e.ag[(f, a)][3] for e in models + yolo for f, _ in CLS_FORMATS)]
    nb = len(models) + len(yolo)
    w = 0.8 / max(nb, 1)
    fig, axes = plt.subplots(1, len(CLS_FORMATS), figsize=(10, 4.4), sharey=True)
    for ax, (f, title) in zip(axes, CLS_FORMATS):
        for i, e in enumerate(models + yolo):
            xs = [j + (i - nb / 2 + 0.5) * w for j in range(len(ags))]
            cells = [e.ag[(f, a)] for a, _ in ags]
            style = YOLO_STYLE if e.kind == "yolo" else {"color": PALETTE[i]}
            ax.bar(xs, [c[0] / 100 for c in cells], w * 0.92, zorder=2, label=e.label + (", top-1 (same in every panel)" if e.kind == "yolo" else ""), **style)
            ax.errorbar(xs, [c[0] / 100 for c in cells], yerr=_err(cells), fmt="none", zorder=4, **{**ERR, "capsize": 1.0})
            if f == "oeq" and e.kind == "model":
                ys = [e.fuzzy[a] / 100 for a, _ in ags]
                ax.bar(xs, ys, w * 0.92, facecolor="none", edgecolor=PALETTE[i], hatch="//////", lw=0, zorder=1)
                ax.hlines(ys, [x - w / 2 for x in xs], [x + w / 2 for x in xs], color="#222222", lw=0.9, zorder=3)
        ax.set_xticks(range(len(ags)))
        ax.set_xticklabels([h.replace("/", "/\n") for _, h in ags])
        ax.set_title(title if f != "oeq" else "OEQ (judged vs. fuzzy match)")
        _style(ax)
    axes[0].set_ylabel("Accuracy (avg. across datasets)")
    axes[0].set_ylim(0, 1)
    h, lab = axes[0].get_legend_handles_labels()
    h += [Patch(facecolor="#888888"), (Patch(facecolor="none", edgecolor="#888888", hatch="//////", lw=0), Line2D([], [], color="#222222", lw=0.9))]
    lab += ["OEQ: LLM judge", "OEQ: fuzzy match"]
    ncol = 3 if max(map(len, lab), default=0) > 30 else 4
    fig.legend(h, lab, loc="lower center", ncol=ncol, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06 + 0.045 * -(-len(lab) // ncol), 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
