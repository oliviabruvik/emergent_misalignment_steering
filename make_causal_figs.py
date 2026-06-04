"""Build the two causal-diagram figures for the 'identification gap' slide.

Output: figures/causal_clean.png and figures/causal_messy.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

# Palette matches utils.py / steering demo so the deck stays coherent.
TARGET = "#1a7a3e"     # intended target / Y_tar
LEAK = "#b22222"        # leakage paths
CONFOUND = "#7a7a7a"    # unobserved-confounder paths (dashed gray)
INK = "#2b2b2b"         # default node + arrow ink
LABEL = "#555"          # caption / annotation text
BUNDLE_FILL = "#fff7d6"
BUNDLE_EDGE = "#b58a00"

R_NODE = 0.45
R_SUB = 0.32


def draw_node(ax, xy, text, *, r=R_NODE, face="white", edge=INK, lw=1.8,
              fontsize=14, dashed=False, text_color=None):
    ls = (0, (5, 3)) if dashed else "solid"
    ax.add_patch(Circle(xy, radius=r, facecolor=face, edgecolor=edge,
                        linewidth=lw, linestyle=ls, zorder=3))
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            fontsize=fontsize, color=text_color or INK, zorder=4)


def draw_arrow(ax, src, dst, *, color=INK, lw=2.2, label=None,
               label_offset=(0, 0.32), label_fontsize=10,
               linestyle="-", connectionstyle="arc3,rad=0",
               shrinkA=18, shrinkB=18, label_color=None):
    ax.add_patch(FancyArrowPatch(
        src, dst, arrowstyle="-|>", mutation_scale=18,
        color=color, linewidth=lw, linestyle=linestyle,
        shrinkA=shrinkA, shrinkB=shrinkB,
        connectionstyle=connectionstyle, zorder=2,
    ))
    if label:
        mx = (src[0] + dst[0]) / 2 + label_offset[0]
        my = (src[1] + dst[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=label_fontsize, ha="center",
                va="center", color=label_color or color)


def draw_wavy_arrow(ax, src, dst, *, color=INK, lw=2.0, amplitude=0.16,
                    waves=5, label=None, label_offset=(0, 0.55),
                    label_fontsize=10, label_color=None,
                    shrinkA=0.5, shrinkB=0.05):
    src = np.array(src, dtype=float)
    dst = np.array(dst, dtype=float)
    u = (dst - src) / np.linalg.norm(dst - src)
    n = np.array([-u[1], u[0]])
    s = src + u * shrinkA
    e = dst - u * shrinkB
    t = np.linspace(0, 1, 240)
    base = s[None, :] + np.outer(t, e - s)
    taper = np.sin(np.pi * t) ** 0.7
    pts = base + (amplitude * taper * np.sin(2 * np.pi * waves * t))[:, None] * n
    cutoff = int(len(t) * 0.96)
    ax.plot(pts[:cutoff, 0], pts[:cutoff, 1], color=color, linewidth=lw,
            solid_capstyle="round", zorder=2)
    ax.annotate("", xy=tuple(e), xytext=tuple(pts[cutoff]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=18))
    if label:
        mid = 0.5 * (s + e)
        lp = mid + n * label_offset[1] + u * label_offset[0]
        ax.text(lp[0], lp[1], label, fontsize=label_fontsize,
                ha="center", va="center", color=label_color or color)


# --- (a) clean mediation chain ---------------------------------------------

def draw_clean(ax):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.7, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    T, M, Y = (1.0, 0), (5.0, 0), (9.0, 0)
    draw_arrow(ax, T, M, label="finetuning", label_offset=(0, 0.30),
               label_fontsize=11, color=INK)
    draw_arrow(ax, M, Y, label="ablation / steering", label_offset=(0, 0.30),
               label_fontsize=11, color=INK)
    draw_node(ax, T, "T", fontsize=16)
    draw_node(ax, M, "M", fontsize=16)
    draw_node(ax, Y, "Y", fontsize=16)

    cap = dict(ha="center", va="center", fontsize=11, color=LABEL, style="italic")
    ax.text(T[0], T[1] - 1.05, "training\ndata", **cap)
    ax.text(M[0], M[1] - 1.05, "concept\ndirection", **cap)
    ax.text(Y[0], Y[1] - 1.05, "behavior", **cap)


# --- (b) the messy reality --------------------------------------------------

def draw_messy(ax):
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-3.4, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")

    T = (0.7, 0)

    # Bundle (M_obs) — sub-directions stacked vertically so right-leg arrows fan out cleanly
    bc = (5.2, 0)
    bw, bh = 1.6, 2.8
    ax.add_patch(Ellipse(bc, width=bw, height=bh, facecolor=BUNDLE_FILL,
                         edgecolor=BUNDLE_EDGE, linewidth=2.0,
                         linestyle=(0, (6, 3)), zorder=1))
    vtar = (bc[0], bc[1] + 0.85)
    vcap = (bc[0], bc[1])
    vcor = (bc[0], bc[1] - 0.85)
    draw_node(ax, vtar, r"$v_{\mathrm{tar}}$", r=R_SUB, fontsize=11,
              edge=TARGET, text_color=TARGET, lw=1.7)
    draw_node(ax, vcap, r"$v_{\mathrm{cap}}$", r=R_SUB, fontsize=11,
              edge="#666", text_color="#444", lw=1.4)
    draw_node(ax, vcor, r"$v_{\mathrm{cor}}$", r=R_SUB, fontsize=11,
              edge="#666", text_color="#444", lw=1.4)
    ax.text(bc[0], bc[1] - 1.70,
            r"$M_{\mathrm{obs}}$  (labeled subspace)",
            ha="center", va="top", fontsize=11, color=LABEL, style="italic")

    # Outcomes
    Ytar, Ycap, Ycor = (10.6, 1.6), (10.6, 0.0), (10.6, -1.6)
    draw_node(ax, Ytar, r"$Y_{\mathrm{tar}}$", fontsize=12,
              edge=TARGET, text_color=TARGET, lw=1.9)
    draw_node(ax, Ycap, r"$Y_{\mathrm{cap}}$", fontsize=12,
              edge="#666", text_color="#444", lw=1.6)
    draw_node(ax, Ycor, r"$Y_{\mathrm{cor}}$", fontsize=12,
              edge="#666", text_color="#444", lw=1.6)

    # Unobserved confounder
    U = (5.2, 3.4)
    draw_node(ax, U, "U", fontsize=14, dashed=True, edge=CONFOUND,
              text_color=CONFOUND, face="#f4f4f4")

    # Training data
    draw_node(ax, T, "T", fontsize=14)
    ax.text(T[0], T[1] - 1.05, "training\ndata", ha="center", va="center",
            fontsize=10, color=LABEL, style="italic")

    # Left leg: wavy "post-hoc identification"
    bundle_west = (bc[0] - bw / 2, bc[1])
    draw_wavy_arrow(ax, T, bundle_west, color=INK, lw=2.0, amplitude=0.18,
                    waves=6, label="post-hoc\nidentification",
                    label_offset=(0, 0.60), label_fontsize=10,
                    label_color=LABEL,
                    shrinkA=R_NODE + 0.05, shrinkB=0.05)

    # Right leg: clean intended path (green) + leakage (red), fanning out from the stack
    draw_arrow(ax, vtar, Ytar, color=TARGET, lw=2.4,
               connectionstyle="arc3,rad=-0.10",
               shrinkA=R_SUB * 50, shrinkB=R_NODE * 50,
               label="intended", label_offset=(-0.2, 0.40),
               label_fontsize=10, label_color=TARGET)
    draw_arrow(ax, vcap, Ycap, color=LEAK, lw=2.0,
               connectionstyle="arc3,rad=0",
               shrinkA=R_SUB * 50, shrinkB=R_NODE * 50,
               label="leakage", label_offset=(0.0, 0.32),
               label_fontsize=10, label_color=LEAK)
    draw_arrow(ax, vcor, Ycor, color=LEAK, lw=2.0,
               connectionstyle="arc3,rad=0.10",
               shrinkA=R_SUB * 50, shrinkB=R_NODE * 50)

    # Confounder paths
    draw_arrow(ax, U, (bc[0], bc[1] + bh / 2 + 0.05),
               color=CONFOUND, lw=1.6, linestyle=(0, (4, 3)),
               shrinkA=R_NODE + 1, shrinkB=2)
    draw_arrow(ax, U, Ytar, color=CONFOUND, lw=1.6,
               linestyle=(0, (4, 3)),
               connectionstyle="arc3,rad=-0.30",
               shrinkA=R_NODE + 1, shrinkB=R_NODE + 1)

    # Annotations
    ax.text(2.4, 2.1, "a different identification\nprocedure picks a\ndifferent subspace",
            ha="center", va="center", fontsize=9, color="#666", style="italic")
    ax.text(8.7, -2.7,
            r"ablating $M_{\mathrm{obs}}$ also degrades $Y_{\mathrm{cap}}$, $Y_{\mathrm{cor}}$",
            ha="center", va="center", fontsize=10, color=LEAK, style="italic")


def main():
    out = Path("figures")
    out.mkdir(exist_ok=True)

    fig_a, ax_a = plt.subplots(figsize=(9, 2.7))
    draw_clean(ax_a)
    fig_a.savefig(out / "causal_clean.png", dpi=220, bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(11, 6.0))
    draw_messy(ax_b)
    fig_b.savefig(out / "causal_messy.png", dpi=220, bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_b)

    for name in ("causal_clean.png", "causal_messy.png"):
        p = out / name
        print(f"wrote {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
