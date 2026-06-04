"""Re-render the 4 matplotlib poster figures in ONE shared style (Computer-Modern
serif + cm mathtext, consistent sizes/dpi) so they look of-a-piece with the
LaTeX body. Re-plots from cached CSVs / directions — no GPU, no model.

Overwrites (same filenames, so main.tex needs no change):
  results_fig5_pure.png, direction_cosine_matrix.png,
  decomposition_scale45_row.png, gsm8k_exact_ablation_bars.png
(causal_messy.png is a hand-made diagram and is intentionally left alone.)
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl

# ---- the shared style ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,        # cmr10 lacks a unicode minus
    "axes.formatter.use_mathtext": True,
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.0,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "resources/model-organisms-for-EM")

FIG = "poster/figures"
DATA = "data/results"
RED = "#d62728"
GREEN = "#2E7D32"
GRAY = "#9E9E9E"
CARD = "#8C1515"


def _scatter(ax, df):
    c = df["bad_stuff"] if "bad_stuff" in df else None
    sc = ax.scatter(df["coherent"], df["aligned"], c=c, cmap="viridis",
                    vmin=0, vmax=100, s=48, edgecolor="white", linewidth=0.35, alpha=0.88)
    ax.axvline(50, color=RED, ls="--", lw=1.4, alpha=0.7)
    ax.axhline(30, color=RED, ls="--", lw=1.4, alpha=0.7)
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 102)
    return sc


# ---------- 1. Fig 5 reproduction (1x5) ----------
scales = [0, 35, 45, 55, 65]
fig, axes = plt.subplots(1, 5, figsize=(18, 4.0), sharey=True)
sc = None
for ax, s in zip(axes, scales):
    df = pd.read_csv(f"{DATA}/fig5_pure/scale_{s:05.1f}.csv")
    for c in ("coherent", "aligned", "bad_stuff"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["coherent", "aligned"])
    sc = _scatter(ax, df)
    ax.set_title(f"scale = {s}")
    ax.set_xlabel("coherent")
axes[0].set_ylabel("aligned")
if sc is not None and sc.get_array() is not None:
    cb = fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label(r"$\mathrm{bad\_stuff}$")
fig.savefig(f"{FIG}/results_fig5_pure.png")
plt.close(fig)
print("ok results_fig5_pure.png")

# ---------- 2. Cosine heatmap ----------
NAMES = ["EM:medical", "EM:sport", "EM:finance", "LoRA-B", r"$V_{\rm cap}$:math", r"$V_{\rm cap}$:coh"]
try:
    from repro.directions import get_meandiff_direction

    def _cache(path, key="v_cap"):
        o = torch.load(path, map_location="cpu")
        if isinstance(o, dict):
            for k in (key, "direction", "vector", "steering_vector"):
                if k in o:
                    return o[k]
            return next(iter(o.values()))
        return o

    dirs = [
        get_meandiff_direction("general_medical", unit=True)["direction"],
        get_meandiff_direction("general_sport", unit=True)["direction"],
        get_meandiff_direction("general_finance", unit=True)["direction"],
        _cache(".cache/em_direction_lora_layer24.pt"),
        _cache(".cache/v_cap_gsm8k_layer24.pt"),
        _cache(".cache/v_cap_soligo_layer24.pt"),
    ]
    vs = [d.detach().to(torch.float32).cpu().flatten() for d in dirs]
    vs = [v / v.norm() for v in vs]
    M = np.array([[float(a @ b) for b in vs] for a in vs])
    print("heatmap recomputed from directions")
except Exception as e:  # noqa: BLE001
    print("direction load failed -> verified matrix:", e)
    M = np.array([
        [1.00, 0.76, 0.82, 0.20, -0.01, -0.01],
        [0.76, 1.00, 0.79, 0.19, -0.01, -0.01],
        [0.82, 0.79, 1.00, 0.18, -0.02, -0.03],
        [0.20, 0.19, 0.18, 1.00, 0.02, 0.02],
        [-0.01, -0.01, -0.02, 0.02, 1.00, 0.00],
        [-0.01, -0.01, -0.03, 0.02, 0.00, 1.00],
    ])

fig, ax = plt.subplots(figsize=(7.6, 6.4))
im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(NAMES)))
ax.set_xticklabels(NAMES, rotation=45, ha="right")
ax.set_yticks(range(len(NAMES)))
ax.set_yticklabels(NAMES)
for i in range(len(NAMES)):
    for j in range(len(NAMES)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(M[i, j]) > 0.5 else "black", fontsize=13)
ax.set_title(r"EM directions converge across domains, $\perp$ to capability (L24)",
             fontsize=15)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
fig.savefig(f"{FIG}/direction_cosine_matrix.png")
plt.close(fig)
print("ok direction_cosine_matrix.png")

# ---------- 3. scale-45 decomposition row (1x3, no title) ----------
panels = [("decomp_vem", r"$v_{\rm EM}$", 6.2),
          ("decomp_coh", r"$v_{\rm bad}$ (coherence)", 4.4),
          ("decomp_math", r"$v_{\rm bad}$ (math)", 5.0)]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
sc = None
for ax, (sub, title, rate) in zip(axes, panels):
    df = pd.read_csv(f"{DATA}/{sub}/scale_045.0.csv")
    sc = _scatter(ax, df)
    ax.set_title(f"{title}    EM {rate:.1f}\\%")
    ax.set_xlabel("coherent")
axes[0].set_ylabel("aligned")
if sc is not None and sc.get_array() is not None:
    cb = fig.colorbar(sc, ax=axes, fraction=0.022, pad=0.012)
    cb.set_label(r"$\mathrm{bad\_stuff}$")
fig.savefig(f"{FIG}/decomposition_scale45_row.png")
plt.close(fig)
print("ok decomposition_scale45_row.png")

# ---------- 4. GSM8k ablation (horizontal, n=200) ----------
labels = ["baseline", r"ablate $v_{\rm EM}$", r"ablate $v_{\rm bad}$",
          r"math $V_{\rm cap}$ control"]
acc = [0.57, 0.61, 0.61, 0.04]
se = [0.035, 0.035, 0.034, 0.014]
colors = [GRAY, GREEN, GREEN, CARD]
fig, ax = plt.subplots(figsize=(11, 3.2))
y = list(range(len(labels)))
ax.barh(y, acc, xerr=se, color=colors, edgecolor="black", linewidth=1.1,
        height=0.62, capsize=6, error_kw=dict(lw=1.6))
ax.axvline(acc[0], ls="--", color=GRAY, lw=1.6, zorder=0)
for yy, a, e in zip(y, acc, se):
    ax.text(a + e + 0.02, yy, f"{a:.2f}", va="center", fontweight="bold", fontsize=15)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("GSM8k accuracy after projection-out ablation")
ax.set_xlim(0, 0.72)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"{FIG}/gsm8k_exact_ablation_bars.png")
plt.close(fig)
print("ok gsm8k_exact_ablation_bars.png")
print("DONE")
