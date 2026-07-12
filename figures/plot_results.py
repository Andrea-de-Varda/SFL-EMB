#!/usr/bin/env python3
"""Figures for the SFL-EMB paper, from the results/ tables.

  * main comparison (A1): per dataset, grouped bars over the 5 models x 5 conditions,
    split into a behavioral panel and a brain panel;
  * SFL decomposition (A2): per dataset, S/F/L independent contributions (gpt2_medium).

Reads the new results schema written by the probes:
  results/<dataset>_<target>_results.csv  (index=model; cols <cond>, <cond>_sd, p_*)
  results/<dataset>_<target>_<model>_decomposed.csv  (rows m/sd; cols S F L SF FL SL SFL)

Run after the pipeline (any subset of models present is plotted). Figures are
written to figures/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT
from src.models import MODEL_ORDER, MODEL_LABELS

RESULTS = PROJECT_ROOT / "results"
FIGDIR = PROJECT_ROOT / "figures"

# condition -> (label, colour). SFL green, EMB blue, combined red, RFF variants.
CONDS = [("sfl", "SFL", "#8fd18f"), ("emb", "EMB", "#87b9f0"),
         ("all", "SFL+EMB", "#e07b7b"), ("sfl_rff", "SFL$_{RFF}$", "#f2c14e"),
         ("sfl_rff_emb", "SFL$_{RFF}$+EMB", "#b48ce0")]

# dataset -> (results csv, panel, pretty title)
BEHAVIORAL = [
    ("provo_fpass", "Provo"), ("meco_fpass", "Meco-en"), ("zuco_fpass", "ZuCo-2"),
    ("ucl_et_fpass", "UCL-ET"), ("ucl_spr", "UCL-SPR"), ("brown", "Brown"),
    ("natstor_spr", "NatStor-SPR"), ("maze_maze", "NatStor-Maze"),
]
BRAIN = [
    ("ucl_n400_n400", "UCL-N400"), ("HP", "Wehbe2014"),
    ("natstorfmri", "NatStor-fMRI"), ("pereira", "Pereira2018"), ("control", "Tuckute2024"),
]


def _load(stem):
    f = RESULTS / f"{stem}_results.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, index_col=0)


def _bar_panel(ax, df, title):
    models = [m for m in MODEL_ORDER if m in df.index]
    n_c = len(CONDS)
    width = 0.8 / n_c
    x = np.arange(len(models))
    for j, (key, lab, col) in enumerate(CONDS):
        if key not in df.columns:
            continue
        vals = df.loc[models, key].to_numpy(dtype=float)
        sd = df.loc[models, f"{key}_sd"].to_numpy(dtype=float) if f"{key}_sd" in df else None
        err = sd / np.sqrt(10) if sd is not None else None   # SEM over 10 folds
        ax.bar(x + (j - (n_c - 1) / 2) * width, vals, width, label=lab, color=col,
               yerr=err, capsize=1.5, ecolor="0.3", error_kw={"lw": 0.7}, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=35, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("r", fontsize=8)
    ax.grid(axis="y", color="0.85", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main_figure():
    for group, name in [(BEHAVIORAL, "behavioral"), (BRAIN, "brain")]:
        panels = [(s, t, _load(s)) for s, t in group]
        panels = [(s, t, d) for s, t, d in panels if d is not None]
        if not panels:
            print(f"[skip] no results for {name} panel")
            continue
        ncol = 4
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow), dpi=200)
        axes = np.atleast_1d(axes).ravel()
        for ax, (s, t, d) in zip(axes, panels):
            _bar_panel(ax, d, t)
        for ax in axes[len(panels):]:
            ax.axis("off")
        axes[0].legend(fontsize=6.5, ncol=2, loc="upper left", framealpha=0.9)
        fig.suptitle(f"SFL vs EMB — {name}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = FIGDIR / f"main_{name}.svg"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def decomposition_figure(model="gpt2_medium"):
    stems = [s for s, _ in BEHAVIORAL] + [s for s, _ in BRAIN]
    titles = dict(BEHAVIORAL + BRAIN)
    found = []
    for s in stems:
        f = RESULTS / f"{s}_{model}_decomposed.csv"
        if f.exists():
            found.append((s, pd.read_csv(f, index_col=0)))
    if not found:
        print(f"[skip] no {model} decomposition files yet")
        return
    order = ["S", "F", "L", "SF", "FL", "SL", "SFL"]
    ncol = 4
    nrow = int(np.ceil(len(found) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax, (s, df) in zip(axes, found):
        cols = [c for c in order if c in df.columns]
        ax.bar(cols, df.loc["m", cols].to_numpy(float),
               yerr=df.loc["sd", cols].to_numpy(float) / np.sqrt(10),
               color="#9b90c2", capsize=1.5, zorder=3)
        ax.set_title(titles.get(s, s), fontsize=9)
        ax.set_ylabel("r", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", color="0.85", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[len(found):]:
        ax.axis("off")
    fig.suptitle(f"SFL decomposition ({MODEL_LABELS.get(model, model)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "sfl_decomposition.svg"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main_figure()
    decomposition_figure()
