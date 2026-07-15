#!/usr/bin/env python3
"""Figures for the layer-wise EMB analysis.

Two figures, one per response type (behavior / brain). Each has one panel per
dataset showing EMB predictivity as a function of layer, against the **standard
SFL** predictor as a horizontal reference line.

SFL is layer-independent (surprisal from the model's final layer + log-frequency +
length -- exactly the main analysis), so it is a constant, not a curve. Its value is
read from results/<stem>_results.csv (gpt2_medium) and cross-checked against the
layer-24 cell of the sweep, which is the same quantity computed by an independent
code path.

Shaded bands are +/- 1 SEM over the 10 outer folds.

    python appendix/layerwise_plot.py     # -> figures/layerwise/{behavior,brain}.png
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
NFOLD = 10

EMB_C, SFL_C = "#4b93c3", "#3f9e3f"
ORDER = {
    "behavior": ["Provo", "Meco-en", "ZuCo-2", "UCL_ET",
                 "UCL_SPR", "Brown", "NatStor_SPR", "NatStor_Maze"],
    "brain": ["UCL_N400", "Pereira2018", "Tuckute2024", "NatStor_fMRI", "Wehbe2014"],
}
NICE = {"UCL_ET": r"UCL$_{ET}$", "UCL_SPR": r"UCL$_{SPR}$", "UCL_N400": r"UCL$_{N400}$",
        "NatStor_SPR": r"NatStor$_{SPR}$", "NatStor_Maze": r"NatStor$_{Maze}$",
        "NatStor_fMRI": r"NatStor$_{fMRI}$"}
STEM = {"Provo": "provo_fpass", "Meco-en": "meco_fpass", "ZuCo-2": "zuco_fpass",
        "UCL_ET": "ucl_et_fpass", "UCL_SPR": "ucl_spr", "Brown": "brown",
        "NatStor_SPR": "natstor_spr", "NatStor_Maze": "maze_maze",
        "UCL_N400": "ucl_n400_n400", "Pereira2018": "pereira", "Tuckute2024": "control",
        "NatStor_fMRI": "natstorfmri", "Wehbe2014": "HP"}


def sfl_standard(name, df):
    """(r, sd) of the standard SFL predictor, from the main results table.

    Asserted equal to the sweep's layer-24 SFL cell (independent code path).
    """
    m = pd.read_csv(ROOT / f"results/{STEM[name]}_results.csv").set_index("model")
    r, sd = float(m.loc["gpt2_medium", "sfl"]), float(m.loc["gpt2_medium", "sfl_sd"])
    cell = df[(df.dataset == name) & (df.layer == df.layer.max())]
    if len(cell):
        assert abs(float(cell.iloc[0].sfl_r) - r) < 1e-4, \
            f"{name}: main-pipeline SFL {r} != sweep layer-24 SFL {cell.iloc[0].sfl_r}"
    return r, sd


def panel(ax, d, name, df):
    d = d.sort_values("layer")
    x = d.layer.to_numpy()
    y, e = d.emb_r.to_numpy(), d.emb_sd.to_numpy() / np.sqrt(NFOLD)
    ax.plot(x, y, color=EMB_C, lw=2.2, label="EMB (layer $L$)", zorder=3)
    ax.fill_between(x, y - e, y + e, color=EMB_C, alpha=.25, lw=0, zorder=2)

    r, sd = sfl_standard(name, df)
    s = sd / np.sqrt(NFOLD)
    ax.axhspan(r - s, r + s, color=SFL_C, alpha=.18, lw=0, zorder=1)
    ax.axhline(r, color=SFL_C, ls="--", lw=2.2, label="SFL (standard)", zorder=3)

    ax.set_title(NICE.get(name, name), fontsize=11)
    ax.grid(alpha=.25, lw=.4)
    ax.set_xlim(0, x.max())
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)


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
            panel(ax, df[df.dataset == nm], nm, df)
        for ax in axes[len(names):]:
            ax.axis("off")
        for i, ax in enumerate(axes[:len(names)]):
            if i % ncol == 0:
                ax.set_ylabel("$r$", fontsize=12)
            if i >= len(names) - ncol:
                ax.set_xlabel("layer", fontsize=11)
        h, l = axes[0].get_legend_handles_labels()
        fig.suptitle(f"EMB by layer vs standard SFL — {grp} (GPT-2$_{{355M}}$)",
                     fontsize=13, y=1.00)
        if len(names) < len(axes):
            axes[len(names)].legend(h, l, fontsize=12, loc="center", frameon=False)
            fig.tight_layout()
        else:
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
