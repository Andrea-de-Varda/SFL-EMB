#!/usr/bin/env python3
"""Figures for the SFL-EMB paper, from the results/ tables.

Per-dataset paper figures (paper/ subdir) — one per dataset, big title, NO legend,
in the original style: 5 models x 3 bars (EMB / SFL / SFL+EMB, lightblue / lightgreen
/ indianred), SEM error bars, and an "n.s." bracket over any pair whose difference is
NOT significant (two-sided permutation test). EMB here is the mean-pooled variant.

Aggregate overview figures (figures/ root):
  main_{behavioral,brain}  EMB / SFL / SFL+EMB grid
  rff_{behavioral,brain}   adds SFL_RFF and SFL_RFF+EMB
  sfl_decomposition        appendix A2

Reads results/<dataset>_<target>_results.csv (index=model; cols <cond>, <cond>_sd,
p_*). Run after the pipeline.
"""
import os
import sys
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT
from src.models import MODEL_ORDER, MODEL_LABELS

RESULTS = PROJECT_ROOT / "results"
FIGDIR = PROJECT_ROOT / "figures"
PAPERDIR = FIGDIR / "paper"

# condition -> (results-column, legend label, colour)
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
ALL_DATASETS = BEHAVIORAL + BRAIN
# every per-target stem to render as its own paper figure
PAPER_STEMS = [(s, t) for s, t in ALL_DATASETS] + [
    ("provo_ffix", "Provo (first-fix)"), ("provo_totfix", "Provo (total-fix)"),
    ("meco_ffix", "Meco-en (first-fix)"), ("meco_totfix", "Meco-en (total-fix)"),
    ("zuco_ffix", "ZuCo-2 (first-fix)"), ("zuco_totfix", "ZuCo-2 (total-fix)"),
    ("ucl_et_ffix", "UCL-ET (first-fix)"), ("ucl_et_totfix", "UCL-ET (total-fix)")]


def _load(stem):
    f = RESULTS / f"{stem}_results.csv"
    return pd.read_csv(f, index_col=0) if f.exists() else None


def _two_sided_ns(p):
    """A directed permutation p = P(A>=B) is 'n.s.' (two-sided) if min(p,1-p) > .025."""
    return min(p, 1 - p) > 0.025


# ---------------------------------------------------------------- per-dataset (paper)
def _bracket(ax, i, j, ns, centers, heights, yerr, dh, barh):
    """Draw an 'n.s.' bracket over bars i,j only when the pair is non-significant."""
    if not ns:
        return
    lx, rx = centers[i], centers[j]
    y = max(heights[i] + yerr[i], heights[j] + yerr[j]) + dh
    ax.plot([lx, lx, rx, rx], [y, y + barh, y + barh, y], c="black", lw=0.8)
    ax.text((lx + rx) / 2, y + barh, "n.s.", ha="center", va="bottom", fontsize=7)


def paper_figure(stem, title):
    df = _load(stem)
    if df is None:
        return
    models = [m for m in MODEL_ORDER if m in df.index]
    keys = [k for k, _, _ in MAIN]                      # emb, sfl, all
    pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19][:3 * len(models)]
    heights, yerr, colors = [], [], []
    for m in models:
        for k, _, c in MAIN:
            heights.append(float(df.loc[m, k]))
            yerr.append(float(df.loc[m, f"{k}_sd"]) / sqrt(10))
            colors.append(c)
    fig, ax = plt.subplots(figsize=(5 / 1.4, 3.5 / 1.4), dpi=300)
    ax.grid(axis="y", which="major", color="#666666", lw=0.5, zorder=0)
    ax.bar(pos, heights, width=0.9, color=colors, yerr=yerr, capsize=2,
           error_kw={"lw": 0.8}, zorder=5)
    ymax = max(h + e for h, e in zip(heights, yerr))
    ax.set_ylim(0, ymax * 1.28)
    # significance brackets: EMB-SFL, SFL-all, EMB-all per model (n.s. only)
    P = {c: (list(df.loc[models, c]) if c in df else [np.nan] * len(models))
         for c in ("p_sfl_emb", "p_all_sfl", "p_all_emb")}
    for g in range(len(models)):
        b0, b1, b2 = 3 * g, 3 * g + 1, 3 * g + 2       # EMB, SFL, all
        _bracket(ax, b0, b1, _two_sided_ns(P["p_sfl_emb"][g]), pos, heights, yerr, ymax * 0.03, ymax * 0.03)
        _bracket(ax, b1, b2, _two_sided_ns(P["p_all_sfl"][g]), pos, heights, yerr, ymax * 0.09, ymax * 0.03)
        _bracket(ax, b0, b2, _two_sided_ns(P["p_all_emb"][g]), pos, heights, yerr, ymax * 0.15, ymax * 0.03)
    centers = [pos[3 * g + 1] for g in range(len(models))]
    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=11)
    ax.set_ylabel("r", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="x", which="both", bottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(PAPERDIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def paper_figures():
    for stem, title in PAPER_STEMS:
        paper_figure(stem, title)
    print(f"wrote per-dataset paper figures -> figures/paper/  "
          f"({sum((RESULTS / f'{s}_results.csv').exists() for s, _ in PAPER_STEMS)} datasets)")


# ---------------------------------------------------------------- aggregate grids
def _panel(ax, df, title, conds):
    models = [m for m in MODEL_ORDER if m in df.index]
    nc = len(conds)
    step = nc + 1
    for j, (key, _, col) in enumerate(conds):
        if key not in df.columns:
            continue
        x = [1 + i * step + j for i in range(len(models))]
        vals = df.loc[models, key].to_numpy(float)
        sem = df.loc[models, f"{key}_sd"].to_numpy(float) / np.sqrt(10) if f"{key}_sd" in df else None
        ax.bar(x, vals, width=0.9, color=col, yerr=sem, capsize=2, error_kw={"lw": 0.7}, zorder=5)
    centers = [1 + i * step + (nc - 1) / 2 for i in range(len(models))]
    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=8)
    ax.set_ylabel("r", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#666666", lw=0.5, zorder=0)
    ax.tick_params(axis="x", which="both", bottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def grid(group, name, conds, fname):
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
    axes[len(panels) - 1].legend(
        handles=[mpatches.Patch(color=c, label=l) for _, l, c in conds],
        fontsize=7.5, loc="best", framealpha=0.9)
    fig.suptitle(name, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("svg", "png"):
        fig.savefig(FIGDIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{fname}.svg/.png")


def decomposition_figure(model="gpt2_medium"):
    stems = dict(ALL_DATASETS)
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
        ax.grid(axis="y", color="#666666", lw=0.5, zorder=0)
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
    paper_figures()                                     # per-dataset, paper style
    grid(BEHAVIORAL, "SFL vs EMB — behavioral", MAIN, "main_behavioral")
    grid(BRAIN, "SFL vs EMB — brain", MAIN, "main_brain")
    grid(BEHAVIORAL, "with RFF — behavioral", RFF, "rff_behavioral")
    grid(BRAIN, "with RFF — brain", RFF, "rff_brain")
    decomposition_figure()
