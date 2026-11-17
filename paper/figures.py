"""
Figures of the article, drawn from the result files.

    python paper/figures.py

fig_matrix.png    accuracy of every method on every task (the cells of the comparison table)
fig_scaling.png   accuracy and time per decision against the number of options
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INK, INK2, LINE, SURF, EMPTY = "#0b0b0b", "#52514e", "#c9c8c2", "#fcfcfb", "#efeee9"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cf5"
SUFFIX = "qwen3_8b"


def results(name):
    path = os.path.join(ROOT, "results", f"{name}_{SUFFIX}.json")
    try:
        with open(path, encoding="utf8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def field(name, key, what):
    return results(name).get(key, {}).get(what)


def matrix():
    tables_path = os.path.join(ROOT, "benchmarks", "tables.json")
    with open(tables_path, encoding="utf8") as f:
        tables = json.load(f)
    spec = tables["tables"]["comparison"]
    rows = spec["rows"]

    n_rows = len(rows)
    n_cols = 4

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    cmap = LinearSegmentedColormap.from_list("seq", ["#e8f1fb", "#0d4f9e"])

    column_labels = ["Banking77\n77 intents", "WordNet\n200 words", "Typed Decisions", "Aim game\ngames won"]

    for r, row in enumerate(rows):
        row_label = row[0].replace("*", "")
        for c, cell in enumerate(row[2:6]):
            if isinstance(cell, str) and cell.startswith("@"):
                name, rest = cell[1:].split("#", 1)
                key = rest.rsplit("|", 1)[0]
                value = field(name, key, "acc")
            else:
                value = None

            rect = plt.Rectangle((c + 0.03, r + 0.06), 0.94, 0.88, color=cmap(value) if value is not None else EMPTY)
            ax.add_patch(rect)

            if value is not None:
                text = f"{100 * value:.1f}%"
                color = "#ffffff" if value >= 0.55 else INK
            else:
                text = "–"
                color = INK2

            ax.text(c + 0.5, r + 0.5, text, ha="center", va="center", fontsize=10, color=color)

    ax.set_xlim(0, 4)
    ax.set_ylim(n_rows, 0)

    ax.set_xticks([c + 0.5 for c in range(n_cols)])
    ax.set_xticklabels(column_labels, fontsize=9.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([r + 0.5 for r in range(n_rows)])
    ax.set_yticklabels([row[0].replace("*", "") for row in rows], fontsize=9.5, color=INK)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(os.path.join(HERE, "fig_matrix.png"), facecolor=SURF, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


SIZES = [200, 1000, 3000, 10000, 30000]
SERIES = [   # label, colour, result file, key pattern, sizes
    ("DecideKV", BLUE, "scale", "{n}/q150", SIZES),
    ("CLM-8B as released", ORANGE, "reference", "clm8b/wordnet/{n}", SIZES),
    ("CLM-8B, heads fine-tuned", GREEN, "reference", "clm8b-ft/wordnet/{n}", SIZES),
    ("OpenJev architecture", PURPLE, "baselines", "wordnet/{n}/openjev", [200]),
]


def scaling():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels(["200", "1,000", "3,000", "10,000", "30,000"], fontsize=8.5, color=INK2)
        ax.minorticks_off()
        ax.set_xlabel("options (log scale)", fontsize=9.5, color=INK)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=LINE, linewidth=0.6, alpha=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(LINE)
        ax.tick_params(colors=INK2, labelsize=8.5)

    axes[0].set_ylim(0, 100)
    axes[0].set_title("Accuracy (%), 150 definitions", fontsize=10, color=INK, loc="left")
    axes[1].set_title("Median time per decision (ms)", fontsize=10, color=INK, loc="left")

    for label, colour, name, pattern, sizes in SERIES:
        points = [(n, field(name, pattern.format(n=n), "acc"), field(name, pattern.format(n=n), "ms")) for n in sizes]
        points = [p for p in points if p[1] is not None]
        xs = [p[0] for p in points]
        acc = [100 * p[1] for p in points]
        axes[0].plot(xs, acc, color=colour, linewidth=2, marker="o", markersize=6, label=label)
        timed = [p for p in points if p[2] is not None]
        axes[1].plot([p[0] for p in timed], [p[2] for p in timed], color=colour, linewidth=2, marker="o", markersize=6)
        if timed:
            axes[1].annotate(f"{timed[-1][2]:.0f} ms", (timed[-1][0], timed[-1][2]), textcoords="offset points",
                            xytext=(6, 0), va="center", fontsize=8, color=INK)
        if points:
            axes[0].annotate(f"{acc[-1]:.0f}%", (xs[-1], acc[-1]), textcoords="offset points", xytext=(6, 0),
                             va="center", fontsize=8, color=INK)

    axes[1].set_ylim(bottom=0)

    # Add vertical line at 3000
    for ax in axes:
        ax.axvline(3000, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))

    # Add text on left panel only
    axes[0].text(3000, 96, "list in the KV cache | catalogue", ha="center", va="center",
                 fontsize=7.5, color=INK2, bbox=dict(fc=SURF, ec="none", pad=1.5))

    # Add figure legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    # Set x-limits
    for ax in axes:
        ax.set_xlim(150, 45000)

    fig.savefig(os.path.join(HERE, "fig_scaling.png"), facecolor=SURF)
    plt.close(fig)


def main():
    matrix()
    scaling()


if __name__ == "__main__":
    main()
