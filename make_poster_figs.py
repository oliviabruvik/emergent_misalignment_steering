"""Generate the two poster-native summary figures Codex's plan needs.

Outputs straight into the poster dir:
  poster/figures/gsm8k_ablation_bars.png   (hero result)
  poster/figures/decomposition_rates_summary.png

GSM8k bars use the n=200 numbers (+ binomial SE) so the hero visual is
defensible; the math control is n=80 until the n=200 control lands (set CTRL_*).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "poster/figures"

CARD = "#8C1515"   # stanford cardinal (collapse)
GREEN = "#2E7D32"  # capability retained
GRAY = "#9E9E9E"   # baseline

# ---------- Figure 1: GSM8k capability-preserving ablation (HERO) ----------
# n=200 for baseline / v_EM / v_bad; control still n=80 (will refresh).
labels = ["baseline", "ablate\n$v_{\\rm EM}$", "ablate\n$v_{\\rm bad}$",
          "ablate math\naxis (control)"]
acc = [0.570, 0.605, 0.610, 0.040]
se = [0.035, 0.035, 0.034, 0.014]
colors = [GRAY, GREEN, GREEN, CARD]

fig, ax = plt.subplots(figsize=(7.2, 5.6))
bars = ax.bar(labels, acc, yerr=se, capsize=8, color=colors,
              edgecolor="black", linewidth=1.2, error_kw=dict(lw=2))
ax.axhline(acc[0], ls="--", color=GRAY, lw=2, zorder=0)
for b, a in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.045, f"{a:.2f}",
            ha="center", va="bottom", fontsize=18, fontweight="bold")
ax.set_ylabel("GSM8k accuracy", fontsize=18)
ax.set_ylim(0, 0.74)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=14)
ax.set_title("Removing EM keeps capability; the control destroys it",
             fontsize=16.5, fontweight="bold", pad=12)
ax.spines[["top", "right"]].set_visible(False)
# annotate the story
ax.annotate("no cost", xy=(1.5, 0.61), xytext=(1.5, 0.70), ha="center",
            fontsize=15, color=GREEN, fontweight="bold")
ax.annotate("collapse", xy=(3, 0.10), xytext=(3, 0.27), ha="center",
            fontsize=15, color=CARD, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CARD, lw=2))
fig.tight_layout()
fig.savefig(f"{OUT}/gsm8k_ablation_bars.png", dpi=150, bbox_inches="tight")
print("wrote gsm8k_ablation_bars.png", acc)

# ---------- Figure 2: decomposition EM-rate summary ----------
dl = ["$v_{\\rm EM}$", "$v_{\\rm bad}$ (coherence)", "$v_{\\rm bad}$ (math)"]
rates = [6.2, 4.4, 5.0]
fig2, ax2 = plt.subplots(figsize=(7.2, 3.0))
y = range(len(dl))
b2 = ax2.barh(list(y), rates, color="#33576E", edgecolor="black", linewidth=1.1, height=0.62)
for bb, r in zip(b2, rates):
    ax2.text(r + 0.12, bb.get_y() + bb.get_height() / 2, f"{r:.1f}%",
             va="center", fontsize=16, fontweight="bold")
ax2.set_yticks(list(y))
ax2.set_yticklabels(dl, fontsize=16)
ax2.invert_yaxis()
ax2.set_xlabel("EM rate at steering scale $s{=}45$", fontsize=15)
ax2.set_xlim(0, 8)
ax2.tick_params(axis="x", labelsize=13)
ax2.set_title("All three steer to ~the same EM rate → no separable component",
              fontsize=14.5, fontweight="bold", pad=10)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
fig2.savefig(f"{OUT}/decomposition_rates_summary.png", dpi=150, bbox_inches="tight")
print("wrote decomposition_rates_summary.png", rates)
