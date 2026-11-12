"""
Diagram of the two regimes of kvindex (cached list, embedding catalogue).

    python paper/diagram.py

fig_mechanism.png          for the article
kvindex_diagram_HD.png     the same drawing at high resolution
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
INK, INK2, LINE, SURF = "#0b0b0b", "#52514e", "#8d8c86", "#fcfcfb"
COMPUTED, CACHED, TRAINED, NEUTRAL = "#d6e6f8", "#d3f0e4", "#fbdccd", "#efeee9"
BLUE, GREEN = "#2a78d6", "#1baf7a"


def canvas(w, h, xlim, ylim, dpi):
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", fill=NEUTRAL, num=None, fs=7.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fill, ec=LINE, lw=1.0, zorder=2))
    head = f"{num}. {title}" if num is not None else title
    if body:
        ax.text(x, y + h * 0.20, head, ha="center", va="center", fontsize=fs + 0.6, weight="bold", color=INK, zorder=3)
        ax.text(x, y - h * 0.17, body, ha="center", va="center", fontsize=fs, color=INK2, zorder=3, linespacing=1.3)
    else:
        ax.text(x, y, head, ha="center", va="center", fontsize=fs + 0.6, weight="bold", color=INK, zorder=3)


def arrow(ax, a, b, text="", color=INK2, tx=0, ty=0, fs=7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, zorder=1))
    if text:
        ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, text, fontsize=fs, color=INK2, ha="center",
                va="center", style="italic", zorder=3, bbox=dict(fc=SURF, ec="none", pad=1.2))


def draw(name, dpi):
    fig, ax = canvas(10, 8.4, 100, 84, dpi)
    
    # Background bands
    ax.add_patch(Rectangle((1, 44.5), 98, 38.5, fc="#f3f6fa", ec="none", zorder=0))
    ax.add_patch(Rectangle((1, 1), 98, 42, fc="#f2f8f4", ec="none", zorder=0))

    # Band titles
    ax.text(2.5, 80.6, "UP TO 3,000 OPTIONS · the whole list is written once in the KV cache",
            fontsize=8.5, weight="bold", color=BLUE, ha="left", va="center", zorder=1)
    ax.text(2.5, 40.6, "LARGER LISTS · an embedding catalogue picks the candidates",
            fontsize=8.5, weight="bold", color=GREEN, ha="left", va="center", zorder=1)

    # Small row captions
    ax.text(2.5, 76.6, "once per list", fontsize=7, color=INK2, style="italic", ha="left", va="center", zorder=1)
    ax.text(2.5, 59.1, "at every decision", fontsize=7, color=INK2, style="italic", ha="left", va="center", zorder=1)
    ax.text(2.5, 36.6, "once per list", fontsize=7, color=INK2, style="italic", ha="left", va="center", zorder=1)
    ax.text(2.5, 19.1, "at every decision", fontsize=7, color=INK2, style="italic", ha="left", va="center", zorder=1)

    # Boxes of the first band
    box(ax, 13, 70, 21, 10, "Options", "list of names or\n{name: description}", num=1)
    box(ax, 38, 70, 21, 10, "Prompt prefix", "instructions +\nthe whole list", num=2)
    box(ax, 63, 70, 21, 10, "One forward pass", "over the prefix,\nonly once", num=3, fill=COMPUTED)
    box(ax, 87.5, 70, 21, 10, "KV cache = index", "keys and values of\nevery option token", num=4, fill=CACHED)
    box(ax, 13, 52.5, 21, 10, "State + question", "message, definition,\ngame state…", num=5)
    box(ax, 47, 52.5, 38, 10, "Forward pass of the new tokens only",
        "every new token attends to all the option\ntokens already in the cache", num=6, fill=COMPUTED)
    box(ax, 85, 52.5, 26, 10, "Constrained decoding", "the answer is always one option;\nrefuted options excluded",
        num=7, fill=COMPUTED)

    # Arrows of the first band
    arrow(ax, (23.5, 70), (27.5, 70))
    arrow(ax, (48.5, 70), (52.5, 70))
    arrow(ax, (73.5, 70), (77, 70))
    arrow(ax, (23.5, 52.5), (28, 52.5))
    arrow(ax, (66, 52.5), (72, 52.5))
    arrow(ax, (87.5, 64.7), (60, 57.8), "exact reuse: nothing is recomputed", color=GREEN, tx=6, ty=1.6)

    # Boxes of the second band
    box(ax, 13, 30, 21, 10, "Options", "any number", num=8)
    box(ax, 38, 30, 21, 10, "Embedding model", "any model: one\nvector per option", num=9, fill=COMPUTED)
    box(ax, 63, 30, 21, 10, "Catalogue", "the option vectors,\nkept in memory", num=10, fill=CACHED)
    box(ax, 87.5, 30, 21, 10, "Optional training", "a small projection learned\nfrom your examples", num=11,
        fill=TRAINED)
    box(ax, 13, 12.5, 21, 10, "State + question", "embedded with\nthe same model", num=12)
    box(ax, 38, 12.5, 21, 10, "Closest options", "the 50 nearest vectors,\nbest first", num=13)
    box(ax, 65, 12.5, 25, 10, "The LLM chooses", "short prompt with the 50,\nconstrained decoding", num=14, fill=COMPUTED)
    box(ax, 90, 12.5, 16, 10, "Answer", "one option", num=15)

    # Arrows of the second band
    arrow(ax, (23.5, 30), (27.5, 30))
    arrow(ax, (48.5, 30), (52.5, 30))
    arrow(ax, (77, 30), (73.5, 30))
    arrow(ax, (23.5, 12.5), (27.5, 12.5))
    arrow(ax, (48.5, 12.5), (52.5, 12.5))
    arrow(ax, (77.5, 12.5), (82, 12.5))
    arrow(ax, (63, 24.7), (41, 17.8), "nearest vectors", color=GREEN, tx=5.5, ty=1.2)

    # Footer text
    ax.text(50, 4.0, "With a trained catalogue the closest option is returned directly: no LLM call.",
            fontsize=7.4, color=INK2, ha="center", va="center", zorder=1)

    # Save
    fig.savefig(os.path.join(OUT, name), facecolor=SURF, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)


def main():
    draw("fig_mechanism.png", 200)
    draw("kvindex_diagram_HD.png", 400)


if __name__ == "__main__":
    main()
