#!/usr/bin/env python3
"""Figures for the layer-matched EMB vs SFL_lens analysis.

Two figures, one per response type (behavior / brain). Each has one panel per
dataset showing predictivity as a function of layer for both readouts taken at the
SAME layer:

    EMB_L      layer-L hidden state
    SFL_L      [logit-lens surprisal at layer L, log-frequency, length]

The last layer is the configuration used in the main analyses (marked).

    python appendix/layerwise_plot.py        # -> figures/layerwise/{behavior,brain}.png
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
FIGDIR = ROOT / "figures/layerwise"

EMB_C, SFL_C = "#8ec3e0", "#84cf84"
ORDER = {
    "behavior": ["Provo", "Meco-en", "ZuCo-2", "UCL_ET",
                 "UCL_SPR", "Brown", "NatStor_SPR", "NatStor_Maze"],
    "brain": ["UCL_N400", "Pereira2018", "Tuckute2024", "NatStor_fMRI", "Wehbe2014"],
}
NICE = {"UCL_ET": r"UCL$_{ET}$", "UCL_SPR": r"UCL$_{SPR}$", "UCL_N400": r"UCL$_{N400}$",
        "NatStor_SPR": r"NatStor$_{SPR}$", "NatStor_Maze": r"NatStor$_{Maze}$",
        "NatStor_fMRI": r"NatStor$_{fMRI}$"}


def panel(ax, d, title):
    d = d.sort_values("layer")
    x = d.layer.to_numpy()
    for col, sd, c, lab in [("emb_r", "emb_sd", EMB_C, "EMB"),
                            ("sfl_r", "sfl_sd", SFL_C, r"SFL$_{lens}$")]:
        y, e = d[col].to_numpy(), d[sd].to_numpy() / np.sqrt(10)
        ax.plot(x, y, color=c, lw=2, label=lab, zorder=3)
        ax.fill_between(x, y - e, y + e, color=c, alpha=.30, lw=0, zorder=2)
    ax.axvline(x.max(), color="0.55", ls=":", lw=1.2, zorder=1)   # main-analysis layer
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=.25, lw=.4)
    ax.set_xlim(0, x.max())
    for s in ax.spines.values():
        s.set_linewidth(1.1)


def plot_all(df):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for grp, names in ORDER.items():
        names = [n for n in names if n in set(df.dataset)]
        if not names:
            continue
        ncol = 4 if grp == "behavior" else 3
        nrow = int(np.ceil(len(names) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow),
                                 dpi=200, squeeze=False)
        axes = axes.ravel()
        for ax, nm in zip(axes, names):
            panel(ax, df[df.dataset == nm], NICE.get(nm, nm))
        for ax in axes[len(names):]:
            ax.axis("off")
        for i, ax in enumerate(axes[:len(names)]):
            if i % ncol == 0:
                ax.set_ylabel("$r$", fontsize=12)
            if i >= len(names) - ncol:
                ax.set_xlabel("layer", fontsize=11)
        h, l = axes[0].get_legend_handles_labels()
        fig.suptitle(f"Layer-matched predictivity — {grp} (GPT-2$_{{355M}}$)",
                     fontsize=13, y=1.00)
        if len(names) < len(axes):          # spare panel -> park the legend in it
            axes[len(names)].legend(h, l, fontsize=12, loc="center", frameon=False)
            fig.tight_layout()
        else:                               # full grid -> reserve a strip underneath
            fig.tight_layout(rect=[0, 0.06, 1, 1])
            fig.legend(h, l, fontsize=12, loc="lower center", frameon=False,
                       ncol=2, bbox_to_anchor=(0.5, 0.0))
        for ext in ("png", "svg"):
            fig.savefig(FIGDIR / f"{grp}.{ext}", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {FIGDIR}/{grp}.png  ({len(names)} datasets)")


if __name__ == "__main__":
    if not CSV.exists():
        sys.exit(f"missing {CSV} -- run appendix/layerwise_logitlens.py first")
    plot_all(pd.read_csv(CSV))
