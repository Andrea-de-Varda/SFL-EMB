#!/usr/bin/env python3
"""Per-dataset layer-wise figures, in the paper's small-plot style.

One figure per dataset (same geometry / title size / spines / grid as
figures/plot_results.py:paper_figure, so they drop straight into the Overleaf
TikZ blocks):

    EMB (blue)  predictivity of the layer-L embedding, across layers
    SFL (green) the standard predictor -- layer-independent, so a flat reference

The EMB line is drawn faded where it does NOT differ significantly from SFL
(paired permutation test over folds, exact two-sided p, Benjamini-Hochberg
corrected across all dataset x layer cells, q <= .05) and solid where it does --
the same criterion the text reports.

    python appendix/layerwise_plot.py     # -> figures/layerwise/<dataset>.{png,svg}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "appendix/layerwise_logitlens.csv"
SIG = ROOT / "appendix/layerwise_significance.csv"
FIGDIR = ROOT / "figures/layerwise"
NFOLD = 10

# palette + geometry: identical to figures/plot_results.py
EMB_C, SFL_C = "lightblue", "lightgreen"
EMB_LINE = "#2b7bba"          # a readable stroke for the light-blue series
SFL_LINE = "#3f9e3f"
SPINE_LW, GRID_LW = 1.3, 0.4
YLABEL_SIZE, YTICK_SIZE = 12, 11
FADE = 0.28                   # alpha where EMB is n.s. vs SFL

# (stem, pretty title), behavior first then brain -- the order of the TikZ blocks
DATASETS = [
    ("Provo", "Provo"), ("Meco-en", "Meco-en"), ("ZuCo-2", "ZuCo-2"), ("UCL_ET", "UCL-ET"),
    ("UCL_SPR", "UCL-SPR"), ("Brown", "Brown"), ("NatStor_SPR", "NatStor-SPR"),
    ("NatStor_Maze", "NatStor-Maze"),
    ("UCL_N400", "UCL-N400"), ("Pereira2018", "Pereira2018"), ("Tuckute2024", "Tuckute2024"),
    ("NatStor_fMRI", "NatStor-fMRI"), ("Wehbe2014", "Wehbe2014"),
]
YLIM = {"behavior": (0.25, 0.95), "brain": (0.05, 0.60)}


def runs(mask, x):
    """Contiguous index runs where mask is True -> list of (start, stop) in x."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def figure(stem, title, d, sg, ylim):
    d = d.sort_values("layer")
    x = d.layer.to_numpy()
    y = d.emb_r.to_numpy()
    e = d.emb_sd.to_numpy() / np.sqrt(NFOLD)
    sfl = float(d[d.layer == d.layer.max()].iloc[0].sfl_r)
    ssd = float(d[d.layer == d.layer.max()].iloc[0].sfl_sd) / np.sqrt(NFOLD)
    sig = sg.sort_values("layer").sig_bh.to_numpy().astype(bool)

    fig, ax = plt.subplots(figsize=(5 / 1.7, 3.5 / 1.7), dpi=300)
    ax.grid(axis="y", which="major", color="#666666", linestyle="-", lw=GRID_LW, zorder=0)
    ax.minorticks_on()
    ax.grid(axis="y", which="minor", color="#999999", linestyle="-", lw=GRID_LW / 2,
            alpha=0.2, zorder=1)
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)

    # SFL: flat reference (+/- SEM band)
    ax.axhspan(sfl - ssd, sfl + ssd, color=SFL_C, alpha=.45, lw=0, zorder=2)
    ax.axhline(sfl, color=SFL_LINE, ls="--", lw=2.0, zorder=4)

    # EMB: faded everywhere, solid over the significant runs
    ax.fill_between(x, y - e, y + e, color=EMB_C, alpha=.45, lw=0, zorder=3)
    ax.plot(x, y, color=EMB_LINE, lw=2.0, alpha=FADE, zorder=5)
    for a, b in runs(sig, x):
        sl = slice(a, b + 1)
        ax.plot(x[sl], y[sl], color=EMB_LINE, lw=2.4, alpha=1.0, zorder=6)

    ax.set_xlim(0, x.max())
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("layer", fontsize=YLABEL_SIZE)
    ax.set_ylabel("r", fontsize=YLABEL_SIZE)
    ax.tick_params(axis="both", labelsize=YTICK_SIZE)
    ax.set_title(title, fontsize=16)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def legend_figure():
    """Standalone legend, matching figures/legend.png in style."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(1266 / 157.48, 760 / 157.48), dpi=157.48)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.05, 0.06), 0.90, 0.88,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=4, edgecolor="lightgray", facecolor="white",
                                transform=ax.transAxes))
    # bigger strokes + text, spread to fill the frame (was small, mostly white space)
    ax.add_line(Line2D([0.10, 0.34], [0.70, 0.70], color=EMB_LINE, lw=8,
                       transform=ax.transAxes))
    ax.text(0.40, 0.70, "EMB (layer $L$)", fontsize=40, va="center", transform=ax.transAxes)
    ax.add_line(Line2D([0.10, 0.34], [0.30, 0.30], color=SFL_LINE, lw=8, ls="--",
                       transform=ax.transAxes))
    ax.text(0.40, 0.30, "SFL", fontsize=40, va="center", transform=ax.transAxes)
    fig.savefig(FIGDIR / "legend_layerwise.png", dpi=157.48)
    plt.close(fig)


def main():
    if not (CSV.exists() and SIG.exists()):
        sys.exit("run appendix/layerwise_logitlens.py --merge and "
                 "appendix/layerwise_significance.py first")
    d = pd.read_csv(CSV)
    s = pd.read_csv(SIG)
    for stem, title in DATASETS:
        g = d[d.dataset == stem]
        sg = s[s.dataset == stem]
        assert len(g) == len(sg) == 25, f"{stem}: {len(g)} cells / {len(sg)} tests"
        grp = "brain" if g.group.iloc[0] == "brain" else "behavior"
        figure(stem, title, g, sg, YLIM[grp])
    legend_figure()
    print(f"wrote {len(DATASETS)} per-dataset figures + legend -> {FIGDIR}")


if __name__ == "__main__":
    main()
