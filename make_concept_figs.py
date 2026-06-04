"""Build the mean-diff steering-vector diagram for the 'identification gap' slide.

Output: figures/concept_meandiff.png

Goal: show that a mean-diff steering vector is *defined* by which contrast set we
chose, and *labeled* post-hoc by what fires on it. Many concepts correlate with
the same pos/neg split, so the label is underdetermined — that is the
correlational nature of mean-diff steering.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Same palette as the causal figures so the deck stays coherent.
TARGET = "#1a7a3e"
INK = "#2b2b2b"
LABEL = "#555"
POS_COLOR = "#cc4d3a"
NEG_COLOR = "#888888"

OUT = Path("figures"); OUT.mkdir(exist_ok=True)


def _arrow(ax, src, dst, *, color=INK, lw=2.0, scale=15, zorder=4):
    ax.add_patch(FancyArrowPatch(src, dst, arrowstyle="-|>",
                                 mutation_scale=scale, color=color,
                                 linewidth=lw, zorder=zorder))


def make_meandiff():
    fig, ax = plt.subplots(figsize=(9, 5.8))
    rng = np.random.default_rng(42)
    n = 18
    pos = rng.normal([2.0, 0.8], [0.55, 0.45], (n, 2))
    neg = rng.normal([-1.7, -0.7], [0.55, 0.55], (n, 2))

    ax.scatter(pos[:, 0], pos[:, 1], s=85, c=POS_COLOR, edgecolors="white",
               linewidth=1, alpha=0.85, zorder=3, label='labeled "toxic"')
    ax.scatter(neg[:, 0], neg[:, 1], s=85, c=NEG_COLOR, edgecolors="white",
               linewidth=1, alpha=0.85, zorder=3, label='labeled "neutral"')

    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    ax.scatter(*mu_pos, s=260, c=POS_COLOR, edgecolors=INK, linewidth=2,
               marker="X", zorder=5)
    ax.scatter(*mu_neg, s=260, c=NEG_COLOR, edgecolors=INK, linewidth=2,
               marker="X", zorder=5)
    ax.text(mu_pos[0] + 0.18, mu_pos[1] + 0.45, r"$\mu_+$",
            fontsize=15, color=INK, fontweight="bold")
    ax.text(mu_neg[0] + 0.18, mu_neg[1] + 0.45, r"$\mu_-$",
            fontsize=15, color=INK, fontweight="bold")

    # The steering vector itself.
    _arrow(ax, mu_neg, mu_pos, color=TARGET, lw=3.0, scale=22)
    midpoint = (mu_pos + mu_neg) / 2
    ax.text(midpoint[0], midpoint[1] + 1.55,
            r"$d = \mu_+ - \mu_-$",
            fontsize=13, color=TARGET, fontweight="bold",
            ha="center", va="center")
    # Tie d explicitly to its steering use.
    ax.text(midpoint[0], midpoint[1] + 1.00,
            r"steer: $h \leftarrow h + \alpha\,d$",
            fontsize=10, color=TARGET, style="italic",
            ha="center", va="center")

    # Candidate post-hoc labels for d
    candidates = ['"toxicity"?', '"negativity"?', '"low formality"?']
    candidate_xs = [-3.0, 0.3, 3.5]
    label_y = -3.1
    for x, lab in zip(candidate_xs, candidates):
        ax.text(x, label_y, lab, fontsize=11, color=INK,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fafafa",
                          edgecolor="#bbb", linewidth=1.0))
        ax.plot([midpoint[0], x], [midpoint[1] - 0.6, label_y + 0.35],
                color="#aaa", linestyle=":", linewidth=1.0, zorder=2)

    ax.text(0.3, -4.05,
            "Many concepts correlate with the pos / neg split — d is consistent with all of them.",
            ha="center", va="center", fontsize=10, color=LABEL, style="italic")

    ax.set_xlim(-4.5, 5.5)
    ax.set_ylim(-4.6, 2.8)
    ax.set_aspect("equal")
    ax.set_xlabel("activation dim 1 (behavior)", fontsize=10, color=LABEL)
    ax.set_ylabel("activation dim 2 (capability)", fontsize=10, color=LABEL)
    ax.tick_params(colors=LABEL, labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor("#ddd")
    ax.legend(loc="upper left", frameon=True, edgecolor="#ddd", fontsize=10)
    ax.set_title("Mean-diff steering vector: defined by contrast",
                 fontsize=13, color=INK, pad=10)

    fig.savefig(OUT / "concept_meandiff.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main():
    make_meandiff()
    p = OUT / "concept_meandiff.png"
    print(f"wrote {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
