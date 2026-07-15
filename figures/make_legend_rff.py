#!/usr/bin/env python3
"""Standalone legend for the SFL_RFF supplementary figure (Overleaf asset).

Mimics the original `legend.png` (rounded light-gray frame, three colour patches,
DejaVu Sans labels) but with the RFF3 palette and labels: EMB / SFL_RFF /
SFL_RFF^EMB. Colours must stay in sync with ``RFF3`` in ``plot_results.py``.

    python figures/make_legend_rff.py        # -> figures/legend_rff.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

FIGDIR = Path(__file__).resolve().parent

# (label, colour) -- same order/colours as RFF3 in plot_results.py.
# NB: the frown needs an empty base ({}^\frown) so it sits *between* SFL_RFF and
# EMB rather than stacking on top of the RFF subscript.
ENTRIES = [("EMB", "#8ec3e0"),
           ("SFL$_{RFF}$", "#84cf84"),
           ("SFL$_{RFF}{}^\\frown$EMB", "indianred")]


def main():
    fig = plt.figure(figsize=(1266 / 157.48, 996 / 157.48), dpi=157.48)
    ax = fig.add_axes([0, 0, 1, 1])          # fill the canvas, as in legend.png
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # rounded outer frame, matching legend.png
    ax.add_patch(FancyBboxPatch((0.07, 0.05), 0.86, 0.90,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                linewidth=4, edgecolor="lightgray",
                                facecolor="white", transform=ax.transAxes))

    ys = [0.74, 0.45, 0.16]          # patch bottoms, evenly spaced like the original
    for y, (label, colour) in zip(ys, ENTRIES):
        ax.add_patch(Rectangle((0.11, y), 0.21, 0.11, facecolor=colour,
                               edgecolor="none", transform=ax.transAxes))
        ax.text(0.39, y + 0.055, label, fontsize=34, va="center", ha="left",
                transform=ax.transAxes)

    out = FIGDIR / "legend_rff.png"
    fig.savefig(out, dpi=157.48, bbox_inches=None, transparent=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
