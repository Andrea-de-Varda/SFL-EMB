#!/usr/bin/env python3
"""Figures for the SFL-EMB paper, from the results/ tables.

Per-dataset paper figures — one per dataset, big title, NO legend, original style:
5 models x 3 bars, SEM error bars, fixed y-axis (0..1) with identical grid, and an
"n.s." bracket (aligned across model-groups) over any pair whose difference is NOT
significant (two-sided permutation test). Two variants:
  paper/       EMB / SFL / SFL+EMB          (lightblue / lightgreen / indianred)
  paper_rff/   EMB / SFL_RFF / SFL_RFF+EMB  (same code, slightly shifted blue+green)

Aggregate overview figures (figures/ root):
  main_{behavioral,brain}  EMB / SFL / SFL+EMB grid
  rff_{behavioral,brain}   EMB / SFL_RFF / SFL_RFF+EMB grid
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
PAPERDIR_RFF = FIGDIR / "paper_rff"

# condition -> (results-column, legend label, colour)
# MAIN: EMB / SFL / EMB+SFL (the paper's main comparison).
MAIN = [("emb", "EMB", "lightblue"), ("sfl", "SFL", "lightgreen"),
        ("all", r"EMB$^\frown$SFL", "indianred")]
# RFF3: same layout with SFL expanded to its RFF version -- EMB / SFL_RFF / SFL_RFF+EMB.
# Same colour code as MAIN (blue / green / red) but slightly shifted blue+green so the
# RFF figures are visually distinguishable from the main ones.
RFF3 = [("emb", "EMB", "#8ec3e0"), ("sfl_rff", r"SFL$_{RFF}$", "#84cf84"),
        ("sfl_rff_emb", r"SFL$_{RFF}^\frown$EMB", "indianred")]

# per-variant bracket p-key triples, in bar order: (bar0-bar1, bar1-bar2, bar0-bar2)
MAIN_PK = ("p_sfl_emb", "p_all_sfl", "p_all_emb")
RFF3_PK = ("p_sflrff_emb", "p_sflrffemb_sflrff", "p_sflrffemb_emb")

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


def _ns(p):
    """A directed permutation p = P(A>=B) is 'n.s.' (two-sided) if min(p,1-p) > .025."""
    return min(p, 1 - p) > 0.025


# ---------------------------------------------------------------- per-dataset (paper)
# Bracket geometry. Levels are placed ONCE per panel and shared by every bar triplet,
# so a given level sits at the same height across models:
#   level 1 = VSHIFT above the tallest MARKED bar (+ its error bar) in the panel --
#             a taller bar that carries no bracket does not count;
#   level 2 = VSHIFT above the top of level 1;  level 3 = VSHIFT above level 2.
# Within a triplet the drawn brackets fill the levels from the bottom up, so a triplet
# with a single n.s. pair keeps it at level 1 rather than floating.
VSHIFT = .03      # gap: bars -> level 1, and between levels (axes fraction)
BARH = .05        # bracket arm height (axes fraction), as in the original

SPINE_LW = 1.6    # plot border (all four spines)
GRID_LW = 0.4     # y-grid lines (minor grid is drawn at half this)


def _bracket_at(ax, num1, num2, center, y, barh, label=True):
    """Draw one bracket with its baseline at absolute height ``y``; return (x_mid, y_top).
    ``label=False`` omits the 'n.s.' text so a triplet can share a single label."""
    lx, rx = center[num1], center[num2]
    ax.plot([lx, lx, rx, rx], [y, y + barh, y + barh, y], c="black")
    mid = ((lx + rx) / 2, y + barh)
    if label:
        ax.text(mid[0], mid[1] + 0.002, "n.s.", ha="center", va="bottom")
    return mid


def _levels(marked_top, barh, vspace):
    """Absolute y for bracket levels 1..3. ``vspace`` = {level: extra gap below it},
    so bumping a level pushes it AND everything above it up."""
    out, y = [], marked_top
    for lvl in (1, 2, 3):
        y += vspace.get(lvl, 0.0) + VSHIFT
        out.append(y)
        y += barh                       # next level starts above this bracket's top
    return out


# Manual per-dataset, per-level extra vertical space (axes fraction). Level is 1/2/3.
#   "HP": {2: .04}   -> adds .04 of space below level 2 (levels 2 and 3 move up)
PAPER_VSPACE = {}
RFF_VSPACE = {}


def paper_figure(stem, title, conds, pk, outdir, vspace=None):
    """One per-dataset paper figure: 3 bars/model, fixed y-axis (0..1), n.s. brackets.
    ``conds`` is a 3-entry [(col,label,colour)]; ``pk`` the bracket p-keys (bar0-bar1,
    bar1-bar2, bar0-bar2); ``vspace`` per-level extra gaps (see PAPER_VSPACE);
    ``outdir`` where to write <stem>.{svg,png}."""
    df = _load(stem)
    if df is None:
        return
    models = [m for m in MODEL_ORDER if m in df.index]
    n = len(models)
    pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19][:3 * n]
    heights, yerr, colors = [], [], []
    for m in models:
        for k, _, c in conds:
            heights.append(float(df.loc[m, k]))
            yerr.append(float(df.loc[m, f"{k}_sd"]) / sqrt(10))
            colors.append(c)
    # geometry/styling identical to the original plotting.py base_plot, so the files
    # drop straight into the Overleaf source (same size, same scale).
    fig, ax = plt.subplots(figsize=(5 / 1.7, 3.5 / 1.7), dpi=300)
    ax.grid(axis="y", which="major", color="#666666", linestyle="-", lw=GRID_LW, zorder=0)
    ax.minorticks_on()
    ax.grid(axis="y", which="minor", color="#999999", linestyle="-", lw=GRID_LW / 2,
            alpha=0.2, zorder=3)
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)
    ax.bar(pos, heights, width=0.9, color=colors, yerr=yerr, capsize=2, zorder=5)
    ax.set_ylim(0, 1)                                   # fixed across every dataset
    ax.set_yticks(np.arange(0, 1.01, 0.2))              # identical grid everywhere
    # which n.s. brackets each triplet draws (fixed pair order: left, right, spanning)
    p01, p12, p02 = (list(df.loc[models, pk[0]]),
                     list(df.loc[models, pk[1]]),
                     list(df.loc[models, pk[2]]))
    drawn, marked = [], set()
    for g in range(n):
        b = 3 * g
        d = [(i, j) for i, j, is_ns in ((b, b + 1, _ns(p01[g])),
                                        (b + 1, b + 2, _ns(p12[g])),
                                        (b, b + 2, _ns(p02[g]))) if is_ns]
        drawn.append(d)
        marked.update(k for pair in d for k in pair)
    if marked:
        # levels shared by every triplet, measured off the tallest MARKED bar only
        top = max(heights[k] + yerr[k] for k in marked)
        lv = _levels(top, BARH, vspace or {})
        for d in drawn:
            if not d:
                continue
            single = len(d) == 1
            mids = [_bracket_at(ax, i, j, pos, lv[k], BARH, label=single)
                    for k, (i, j) in enumerate(d)]
            if not single:                                # one shared label, on top
                x, y = max(mids, key=lambda m: m[1])
                ax.text(x, y + 0.002, "n.s.", ha="center", va="bottom")
    centers = [pos[3 * g + 1] for g in range(n)]
    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=11)
    ax.set_ylabel("r")
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=True)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def paper_figures():
    for stem, title in PAPER_STEMS:
        # EMB / SFL / EMB+SFL
        paper_figure(stem, title, MAIN, MAIN_PK, PAPERDIR, vspace=PAPER_VSPACE.get(stem))
        # EMB / SFL_RFF / SFL_RFF+EMB -- new figure
        paper_figure(stem, title, RFF3, RFF3_PK, PAPERDIR_RFF, vspace=RFF_VSPACE.get(stem))
    nd = sum((RESULTS / f'{s}_results.csv').exists() for s, _ in PAPER_STEMS)
    print(f"wrote per-dataset paper figures -> figures/paper/ + figures/paper_rff/  ({nd} datasets)")


# ---------------------------------------------------------------- aggregate grids
def _panel(ax, df, title, conds, ylim=1.0):
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
    ax.set_ylim(0, ylim)                                # identical y-axis on every panel
    ax.set_yticks(np.arange(0, ylim + 1e-9, 0.2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#666666", lw=0.5)
    ax.tick_params(axis="x", which="both", bottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def grid(group, name, conds, fname, ylim=1.0):
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
        _panel(ax, d, t, conds, ylim=ylim)
    empties = axes[len(panels):]
    for ax in empties:
        ax.axis("off")
    handles = [mpatches.Patch(color=c, label=l) for _, l, c in conds]
    if len(empties):                                    # legend outside the data panels
        empties[0].legend(handles=handles, fontsize=10, loc="center", framealpha=0)
    else:                                               # full grid -> legend below
        fig.legend(handles=handles, fontsize=9, loc="lower center",
                   ncol=len(conds), framealpha=0.9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(name, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("svg", "png"):
        fig.savefig(FIGDIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{fname}.svg/.png")


# SFL decomposition (appendix): the 7 feature subsets, GROUPED by cardinality
# (singles S/F/L | pairs SF/FL/SL | triple SFL) with a light->dark blue gradient,
# one big-title figure per dataset (the paper style).
DECOMP_ORDER = ["S", "F", "L", "SF", "FL", "SL", "SFL"]
DECOMP_POS = [1, 2, 3, 4.5, 5.5, 6.5, 8]
DECOMP_BLUE = {1: "#87CEFA", 2: "#4169E1", 3: "#0000CD"}   # by number of features
DECOMPDIR = FIGDIR / "decomp"


def decomposition_figures(model="gpt2_medium"):
    n = 0
    for stem, title in ALL_DATASETS:
        f = RESULTS / f"{stem}_{model}_decomposed.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col=0)
        cols = [c for c in DECOMP_ORDER if c in df.columns]
        pos = [DECOMP_POS[DECOMP_ORDER.index(c)] for c in cols]
        colors = [DECOMP_BLUE[len(c)] for c in cols]
        # geometry/styling identical to the original plot_sfl_decomposed.py
        fig, ax = plt.subplots(figsize=(5 / 1.7, 3.5 / 1.7), dpi=300)
        ax.grid(axis="y", which="major", color="#666666", linestyle="-", lw=GRID_LW, zorder=0)
        ax.minorticks_on()
        ax.grid(axis="y", which="minor", color="#999999", linestyle="-", lw=GRID_LW / 2,
                alpha=0.2, zorder=3)
        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)
        ax.bar(pos, df.loc["m", cols].to_numpy(float), width=0.9, color=colors,
               yerr=df.loc["sd", cols].to_numpy(float) / sqrt(10), capsize=2, zorder=5)
        ax.set_xticks(pos)
        ax.set_xticklabels(cols, fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_ylabel("r")
        ax.set_ylim(-0.1, 0.9)
        ax.set_title(title, fontsize=20)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=True)
        DECOMPDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("svg", "png"):
            fig.savefig(DECOMPDIR / f"{stem}.{ext}", bbox_inches="tight")
        plt.close(fig)
        n += 1
    print(f"wrote SFL decomposition figures -> figures/decomp/  "
          f"({n} datasets, {MODEL_LABELS.get(model, model)})")


# ---------------------------------------------------------------- model-size boxplots
# Effect of model size: within each dataset z-score r across the 5 models, then box the
# distribution (across datasets) of that z per model. One small figure per condition x
# {behavioral, brain}, identical to the original plotting.py `bxplt`.
MSIZE_CONDS = [("sfl", "SFL"), ("emb", "EMB"), ("all", r"EMB$^\frown$SFL")]
MSIZE_LABELS = ["GPT", "GPT2$_{124M}$", "GPT2$_{355M}$", "GPT2$_{774M}$", "GPT2$_{1.5B}$"]
MSIZE_COLORS = ["pink", "lightblue", "lightgreen", "indianred"]   # 5th box left default (as in paper)
MSIZEDIR = FIGDIR / "model_size"


def _zbox_data(group, cond):
    """Per-model list of z(r) values, one entry per dataset in ``group``."""
    per_model = [[] for _ in MODEL_ORDER]
    for stem, _ in group:
        df = _load(stem)
        if df is None or cond not in df.columns:
            continue
        ms = [m for m in MODEL_ORDER if m in df.index]
        if len(ms) != len(MODEL_ORDER):
            continue
        r = df.loc[MODEL_ORDER, cond].to_numpy(float)
        if r.std() == 0:
            continue
        for i, v in enumerate((r - r.mean()) / r.std()):
            per_model[i].append(v)
    return per_model


def _bxplt(data, title, fname):
    meanprops = {"marker": "s", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4}
    fig = plt.figure(dpi=300, figsize=(3.75 / 2.3, 2.5 / 2.3))
    bp = plt.boxplot(data, vert=True, patch_artist=True, tick_labels=MSIZE_LABELS,
                     showmeans=True, meanprops=meanprops, whis=5)
    plt.ylabel(r"$z(r)$", labelpad=-5)
    plt.xticks(fontsize=10, rotation=35, ha="right", rotation_mode="anchor")
    for patch, color in zip(bp["boxes"], MSIZE_COLORS):
        patch.set_facecolor(color)
    plt.setp(bp["medians"], color="black")
    plt.ylim(-2.1, 2.1)
    plt.title(title)
    MSIZEDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(MSIZEDIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)


def modelsize_boxplots():
    for group, gname in [(BEHAVIORAL, "behavioral"), (BRAIN, "brain")]:
        for cond, clab in MSIZE_CONDS:
            _bxplt(_zbox_data(group, cond), clab, f"{gname}_{cond}")
    print("wrote model-size boxplots -> figures/model_size/  (6 figures)")


if __name__ == "__main__":
    paper_figures()                                     # per-dataset, paper style
    grid(BEHAVIORAL, "SFL vs EMB — behavioral", MAIN, "main_behavioral")
    grid(BRAIN, "SFL vs EMB — brain", MAIN, "main_brain")
    grid(BEHAVIORAL, r"SFL$_{RFF}$ expanded — behavioral", RFF3, "rff_behavioral")
    grid(BRAIN, r"SFL$_{RFF}$ expanded — brain", RFF3, "rff_brain", ylim=0.6)
    decomposition_figures()
    modelsize_boxplots()
