"""README hero: SFL-vs-EMB dumbbell across all 13 datasets (mean over 5 models).

House style (Andrea): editable-text SVG, thick left/bottom spines only, dashed x-grid,
compact canvas. Paper colour language: SFL = green, EMB = blue. The connector is
coloured by the winner, so the behaviour->SFL / brain->EMB dissociation is visible at
a glance. -> docs/hero.{svg,png}
"""
import matplotlib as mpl
mpl.use("Agg"); mpl.rcParams["svg.fonttype"] = "none"; mpl.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFL_C, EMB_C = "#5aa469", "#4b93c3"     # green / blue, paper legend
# (label, results stem, group) — behaviour block then brain block
ROWS = [
    ("Provo", "provo_fpass", "beh"), ("Meco-en", "meco_fpass", "beh"),
    ("ZuCo-2", "zuco_fpass", "beh"), ("UCL-ET", "ucl_et_fpass", "beh"),
    ("UCL-SPR", "ucl_spr", "beh"), ("Brown", "brown", "beh"),
    ("NatStor-SPR", "natstor_spr", "beh"), ("NatStor-Maze", "maze_maze", "beh"),
    ("UCL-N400", "ucl_n400_n400", "brain"), ("Wehbe2014", "HP", "brain"),
    ("NatStor-fMRI", "natstorfmri", "brain"), ("Pereira2018", "pereira", "brain"),
    ("Tuckute2024", "control", "brain"),
]
def rr(stem, col):
    return float(pd.read_csv(ROOT / f"results/{stem}_results.csv")[col].mean())

data = [(lab, rr(s, "sfl"), rr(s, "emb"), g) for lab, s, g in ROWS]
# display order top->bottom = behaviour block (8) then brain block (5).
# matplotlib y grows upward, so row i (0=top) sits at y = n-1-i.
n = len(data)
y = [n - 1 - i for i in range(n)]
n_beh = sum(g == "beh" for *_, g in data)     # 8

fig, ax = plt.subplots(figsize=(5.6 * 0.82, 5.0 * 0.82), dpi=300)
for i, (lab, sfl, emb, g) in enumerate(data):
    win = SFL_C if sfl > emb else EMB_C
    ax.hlines(y[i], min(sfl, emb), max(sfl, emb), color=win, alpha=0.5, lw=3, zorder=2)
    ax.scatter(sfl, y[i], s=42, color=SFL_C, edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter(emb, y[i], s=42, color=EMB_C, edgecolors="black", linewidths=0.5, zorder=3)

# divider between the behaviour block (top) and the brain block (bottom)
div = (n - 1 - n_beh) + 0.5      # between row n_beh-1 and row n_beh
ax.axhline(div, color="0.8", lw=1, ls=":", zorder=1)
ax.text(0.02, n - 1 - 0.15, "BEHAVIOR", fontsize=9.5, weight="bold", color=SFL_C, va="top")
ax.text(0.02, div - 0.35, "BRAIN", fontsize=9.5, weight="bold", color=EMB_C, va="top")

ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data], fontsize=10)
ax.set_ylim(-0.6, n - 0.4)
ax.set_xlabel("predictivity  (mean $r$ over 5 models)", fontsize=12)
ax.set_xlim(0, 0.9)
ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(1.5)
ax.tick_params(labelsize=10)
ax.set_title("SFL predicts behavior; EMB predicts the brain", fontsize=13)

ax.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor=SFL_C, markeredgecolor="black", markersize=7, label="SFL  (surprisal, frequency, length)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=EMB_C, markeredgecolor="black", markersize=7, label="EMB  (language-model embedding)"),
], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False, fontsize=9)

plt.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(ROOT / f"docs/hero.{ext}", bbox_inches="tight")
print("wrote docs/hero.{svg,png}")
for lab, sfl, emb, g in data:
    print(f"  {lab:<14} SFL {sfl:.3f}  EMB {emb:.3f}  Δ {sfl-emb:+.3f}  ({g})")
