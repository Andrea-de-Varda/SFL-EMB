#!/usr/bin/env python3
"""Figures for the SFL-EMB paper, from the results/ tables.

  main_{behavioral,brain}   the ORIGINAL paper style: per dataset, grouped bars
                            over the 5 models x 3 conditions (EMB / SFL / SFL+EMB),
                            lightblue / lightgreen / indianred, SEM error bars.
  rff_{behavioral,brain}    the same layout PLUS the two RFF conditions
                            (SFL_RFF, SFL_RFF+EMB) -> 5 bars per model.
  sfl_decomposition         appendix A2: S/F/L independent contributions (gpt2_medium).

Reads results/<dataset>_<target>_results.csv (index=model; cols <cond>, <cond>_sd).
Figures are written to figures/ as .svg + .png.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT
from src.models import MODEL_ORDER, MODEL_LABELS

RESULTS = PROJECT_ROOT / "results"
FIGDIR = PROJECT_ROOT / "figures"

# ---- conditions: (results-column, legend label, colour) ----
# original three (identical to the paper) + the two RFF additions
MAIN = [("emb", "EMB", "lightblue"), ("sfl", "SFL", "lightgreen"),
        ("all", r"EMB$^\frown$SFL", "indianred")]
RFF = MAIN + [("sfl_rff", r"SFL$_{RFF}$", "#f2c14e"),
              ("sfl_rff_emb", r"SFL$_{RFF}^\frown$EMB", "#b48ce0")]

# dataset -> (results-file stem, pretty title)
BEHAVIORAL = [("provo_fpass", "Provo"), ("meco_fpass", "Meco-en"), ("zuco_fpass", "ZuCo-2"),
              ("ucl_et_fpass", "UCL-ET"), ("ucl_spr", "UCL-SPR"), ("brown", "Brown"),
              ("natstor_spr", "NatStor-SPR"), ("maze_maze", "NatStor-Maze")]
BRAIN = [("ucl_n400_n400", "UCL-N400"), ("HP", "Wehbe2014"), ("natstorfmri", "NatStor-fMRI"),
         ("pereira", "Pereira2018"), ("control", "Tuckute2024")]


def _load(stem):
    f = RESULTS / f"{stem}_results.csv"
    return pd.read_csv(f, index_col=0) if f.exists() else None


def _panel(ax, df, title, conds, ylim=None):
    """One dataset: 5 models x len(conds) grouped bars, original base_plot look."""
    models = [m for m in MODEL_ORDER if m in df.index]
    nc = len(conds)
    # x positions: groups of nc bars with a gap between models (like the paper)
    step = nc + 1
    for j, (key, _, col) in enumerate(conds):
        x = [1 + i * step + j for i in range(len(models))]
        vals = df.loc[models, key].to_numpy(float)
        sem = (df.loc[models, f"{key}_sd"].to_numpy(float) / np.sqrt(10)
               if f"{key}_sd" in df else None)
        ax.bar(x, vals, width=0.9, color=col, yerr=sem, capsize=2,
               error_kw={"lw": 0.8}, zorder=5)
    centers = [1 + i * step + (nc - 1) / 2 for i in range(len(models))]
    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=8)
    ax.set_ylabel("r", fontsize=9)
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", which="major", color="#666666", ls="-", lw=0.5, zorder=0)
    ax.grid(axis="y", which="minor", color="#999999", ls="-", lw=0.4, alpha=0.3, zorder=0)
    ax.minorticks_on()
    ax.tick_params(axis="x", which="both", bottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure(group, name, conds, fname):
    panels = [(s, t, _load(s)) for s, t in group]
    panels = [(s, t, d) for s, t, d in panels if d is not None]
    if not panels:
        print(f"[skip] no results for {name}")
        return
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax, (s, t, d) in zip(axes, panels):
        _panel(ax, d, t, conds)
    for ax in axes[len(panels):]:
        ax.axis("off")
    handles = [mpatches.Patch(color=c, label=l) for _, l, c in conds]
    axes[len(panels) - 1].legend(handles=handles, fontsize=7.5, loc="best", framealpha=0.9)
    fig.suptitle(name, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("svg", "png"):
        fig.savefig(FIGDIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{fname}.svg/.png")


def decomposition_figure(model="gpt2_medium"):
    stems = dict(BEHAVIORAL + BRAIN)
    order = ["S", "F", "L", "SF", "FL", "SL", "SFL"]
    found = [(s, pd.read_csv(RESULTS / f"{s}_{model}_decomposed.csv", index_col=0))
             for s in stems if (RESULTS / f"{s}_{model}_decomposed.csv").exists()]
    if not found:
        print(f"[skip] no {model} decomposition files")
        return
    ncol = 4
    nrow = int(np.ceil(len(found) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax, (s, df) in zip(axes, found):
        cols = [c for c in order if c in df.columns]
        ax.bar(cols, df.loc["m", cols].to_numpy(float),
               yerr=df.loc["sd", cols].to_numpy(float) / np.sqrt(10),
               color="#9b90c2", capsize=2, zorder=5)
        ax.set_title(stems[s], fontsize=10)
        ax.set_ylabel("r", fontsize=9)
        ax.grid(axis="y", color="#666666", ls="-", lw=0.5, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[len(found):]:
        ax.axis("off")
    fig.suptitle(f"SFL decomposition ({MODEL_LABELS.get(model, model)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("svg", "png"):
        fig.savefig(FIGDIR / f"sfl_decomposition.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/sfl_decomposition.svg/.png")


if __name__ == "__main__":
    # main figures — identical to the paper (EMB / SFL / SFL+EMB)
    figure(BEHAVIORAL, "SFL vs EMB — behavioral", MAIN, "main_behavioral")
    figure(BRAIN, "SFL vs EMB — brain", MAIN, "main_brain")
    # extra figures — with the RFF conditions
    figure(BEHAVIORAL, "with RFF — behavioral", RFF, "rff_behavioral")
    figure(BRAIN, "with RFF — brain", RFF, "rff_brain")
    # appendix A2
    decomposition_figure()
