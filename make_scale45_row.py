"""1x3 scale-45 scatter row: the 'pretty' version of the decomposition result,
trimmed to the one row where steering actually happens (drops the null scale-0
column from the 3-axis grid). Echoes Fig 5's quadrant style.
Saves a CANDIDATE figure to the poster dir (not wired into main.tex).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUTDIR = "poster/figures"
DATA = "data/results"
panels = [("decomp_vem", r"$v_{\rm EM}$", 6.2),
          ("decomp_coh", r"$v_{\rm bad}$ (coherence)", 4.4),
          ("decomp_math", r"$v_{\rm bad}$ (math)", 5.0)]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
sc = None
for ax, (sub, title, rate) in zip(axes, panels):
    df = pd.read_csv(f"{DATA}/{sub}/scale_045.0.csv")
    c = df["bad_stuff"] if "bad_stuff" in df else "#5B2C8C"
    sc = ax.scatter(df["coherent"], df["aligned"], c=c, cmap="viridis",
                    s=60, edgecolor="white", linewidth=0.4, vmin=0, vmax=100, alpha=0.9)
    ax.axhline(30, ls="--", color="red", lw=1.5, alpha=0.7)
    ax.axvline(50, ls="--", color="red", lw=1.5, alpha=0.7)
    ax.set_xlim(0, 102); ax.set_ylim(0, 102)
    ax.set_title(f"{title}    EM {rate:.1f}%", fontsize=17)
    ax.set_xlabel("coherent", fontsize=14)
    ax.tick_params(labelsize=12)
axes[0].set_ylabel("aligned", fontsize=15)
if hasattr(sc, "get_array") and sc.get_array() is not None:
    cb = fig.colorbar(sc, ax=axes, fraction=0.022, pad=0.012)
    cb.set_label("bad_stuff", fontsize=12)
# no suptitle: the caption ("Figure 3. ...") lives in the poster LaTeX
fig.savefig(f"{OUTDIR}/decomposition_scale45_row.png", dpi=150, bbox_inches="tight")
print("wrote", f"{OUTDIR}/decomposition_scale45_row.png")
