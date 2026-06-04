"""Combined multi-row steering figure (Soligo Fig 5 layout, extended).

plot_steering_grid lays out one row per steering direction (e.g. raw v_EM, 1-D v_bad,
k-dim v_bad) and one column per scale, each cell a coherent-vs-aligned scatter colored
by `bad_stuff` — so the three decomposition conditions read top-to-bottom like the
paper's figure reads left-to-right.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_cosine_heatmap(directions: dict, title: str = "direction cosine similarity"):
    """Heatmap of pairwise cosine similarity for a dict {name: 1-D tensor}.

    Used for the cross-domain convergence / capability-orthogonality figure:
    EM directions (medical/sport/finance) converge (~0.8) and are orthogonal to the
    capability axes (~0). Returns the matplotlib Figure.
    """
    import numpy as np
    import torch

    names = list(directions)
    vs = []
    for n in names:
        v = directions[n].detach().to(torch.float32).cpu().flatten()
        vs.append(v / v.norm())
    M = np.array([[float(a @ b) for b in vs] for a in vs])

    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 1.0 * len(names) + 1.5))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.5 else "black", fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    fig.tight_layout()
    return fig


def plot_steering_grid(
    rows,                      # list of (label, folder) — one row each, top to bottom
    scales,                    # list of scales — one column each, left to right
    colour_by: str = "bad_stuff",
    x: str = "coherent",
    y: str = "aligned",
    x_line: int = 50,
    y_line: int = 30,
    fname_fmt: str = "scale_{:05.1f}.csv",
):
    """Render an (len(rows) × len(scales)) grid of coherent-vs-aligned scatters.

    Returns the matplotlib Figure. Each row reads CSVs from its folder; quadrant lines
    at x=50 / y=30; hue = `colour_by` (legend only on the top-right cell to save space).
    """
    nrows, ncols = len(rows), len(scales)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), squeeze=False)

    for i, (label, folder) in enumerate(rows):
        folder = Path(folder)
        for j, s in enumerate(scales):
            ax = axes[i][j]
            path = folder / fname_fmt.format(s)
            if not path.exists():
                ax.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center", fontsize=8)
                ax.set_xlim(0, 100); ax.set_ylim(0, 100)
                continue
            df = pd.read_csv(path)
            for c in (x, y, colour_by):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=[x, y])
            hue = df[colour_by] if colour_by in df.columns else None
            sns.scatterplot(
                data=df, x=x, y=y, hue=hue, palette="viridis",
                alpha=0.6, s=40, ax=ax, legend=(i == 0 and j == ncols - 1),
            )
            ax.axvline(x_line, color="red", ls="--", alpha=0.6)
            ax.axhline(y_line, color="red", ls="--", alpha=0.6)
            ax.set_xlim(0, 100); ax.set_ylim(0, 100)
            ax.grid(True, ls="-", alpha=0.15)
            if i == 0:
                ax.set_title(f"scale = {s:g}", fontsize=11)
            ax.set_xlabel(x if i == nrows - 1 else "")
            ax.set_ylabel(f"{label}\n{y}" if j == 0 else "")
            if hue is not None and i == 0 and j == ncols - 1:
                ax.legend(title=colour_by, loc="upper left", fontsize=8, framealpha=0.6)

    fig.tight_layout()
    return fig
