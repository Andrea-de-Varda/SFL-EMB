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
# Bracket placement is the original plotting.py `barplot_annotate_brackets` verbatim:
# the bracket sits `dh` above the taller of the two bars (dh/barh are axes fractions),
# and the label uses the default font size.
def _annotate_bracket(ax, num1, num2, center, height, yerr=None, dh=.05, barh=.05,
                      fs=None, label=True):
    """Draw one bracket; return its (x_mid, y_top). ``label=False`` omits the 'n.s.'
    text, so a group of brackets can share a single label."""
    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]
    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]
    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)
    y = max(ly, ry) + dh
    ax.plot([lx, lx, rx, rx], [y, y + barh, y + barh, y], c="black")
    mid = ((lx + rx) / 2, y + barh)
    if label:
        kwargs = dict(ha="center", va="bottom")
        if fs is not None:
            kwargs["fontsize"] = fs
        ax.text(mid[0], mid[1] + 0.002, "n.s.", **kwargs)
    return mid


# Per-dataset bracket tuning, verbatim from the original base_plot call sites.
# Defaults (as in the original signature): dh1=dh2=dh3=.03, barh1=barh2=barh3=.05.
# dhlist2/dhlist3 give a per-model dh for levels 2/3 (the original `annotate=False` path).
PAPER_TUNING = {
    "provo_fpass":  dict(barh2=.018),
    "provo_ffix":   dict(barh2=.018),
    "provo_totfix": dict(barh2=.018),
    "ucl_spr":      dict(dh2=.22),
    "HP":           dict(dh1=.07, dhlist2=[.4, .3, .3, .3, .3],
                         dhlist3=[.52, .42, .44, .43, .43]),
    # Brain sets where EMB ~= EMB+SFL: the EMB-SFL (L1) and EMB-all (L3) brackets
    # otherwise land on top of each other, so L3 is lifted per-model clear of L1's
    # label -- and left low for models where L1 is not drawn.
    "ucl_n400_n400":  dict(dhlist3=[.03, .14, .14, .03, .03]),
    # natstorfmri: gpt2_medium lifted to sit level with GPT's tallest bracket.
    "natstorfmri":    dict(dhlist3=[.16, .18, .117, .03, .03]),
}

# Same idea for the RFF variant (a new figure, so nothing to copy from the paper):
# only the brain sets where EMB ~= SFL_RFF+EMB stack more than one bracket.
RFF_TUNING = {
    "HP":          dict(dhlist2=[.21, .23, .215, .205, .03],
                        dhlist3=[.35, .35, .35, .35, .19]),
    "natstorfmri": dict(dhlist3=[.17, .17, .03, .141, .03]),
    "pereira":     dict(dhlist2=[.16, .158, .03, .161, .165]),
}


def paper_figure(stem, title, conds, pk, outdir, tuning=None):
    """One per-dataset paper figure: 3 bars/model, fixed y-axis (0..1), n.s. brackets.
    ``conds`` is a 3-entry [(col,label,colour)]; ``pk`` the bracket p-keys (bar0-bar1,
    bar1-bar2, bar0-bar2); ``tuning`` the per-plot dh/barh overrides (see PAPER_TUNING);
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
    ax.grid(axis="y", which="major", color="#666666", linestyle="-", zorder=0)
    ax.minorticks_on()
    ax.grid(axis="y", which="minor", color="#999999", linestyle="-", alpha=0.2, zorder=3)
    ax.bar(pos, heights, width=0.9, color=colors, yerr=yerr, capsize=2, zorder=5)
    ax.set_ylim(0, 1)                                   # fixed across every dataset
    ax.set_yticks(np.arange(0, 1.01, 0.2))              # identical grid everywhere
    # n.s. brackets, placed exactly as in the original (dh above the taller bar)
    t = dict(dh1=.03, dh2=.03, dh3=.03, barh1=.05, barh2=.05, barh3=.05,
             dhlist2=None, dhlist3=None)
    t.update(tuning or {})
    p01, p12, p02 = (list(df.loc[models, pk[0]]),
                     list(df.loc[models, pk[1]]),
                     list(df.loc[models, pk[2]]))
    for g in range(n):
        b = 3 * g
        dh2 = t["dhlist2"][g] if t["dhlist2"] else t["dh2"]
        dh3 = t["dhlist3"][g] if t["dhlist3"] else t["dh3"]
        drawn = [(i, j, dh, bh) for i, j, is_ns, dh, bh in (
            (b,     b + 1, _ns(p01[g]), t["dh1"], t["barh1"]),
            (b + 1, b + 2, _ns(p12[g]), dh2,      t["barh2"]),
            (b,     b + 2, _ns(p02[g]), dh3,      t["barh3"])) if is_ns]
        # with 2-3 n.s. pairs in a triplet, draw the brackets but label them once
        single = len(drawn) == 1
        mids = [_annotate_bracket(ax, i, j, pos, heights, yerr, dh=dh, barh=bh, label=single)
                for i, j, dh, bh in drawn]
        if not single and mids:
            x, y = max(mids, key=lambda m: m[1])          # above the topmost bracket
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
        # EMB / SFL / EMB+SFL -- with the paper's hand-tuned bracket placement
        paper_figure(stem, title, MAIN, MAIN_PK, PAPERDIR, tuning=PAPER_TUNING.get(stem))
        # EMB / SFL_RFF / SFL_RFF+EMB -- new figure
        paper_figure(stem, title, RFF3, RFF3_PK, PAPERDIR_RFF, tuning=RFF_TUNING.get(stem))
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
        ax.grid(axis="y", which="major", color="#666666", linestyle="-", zorder=0)
        ax.minorticks_on()
        ax.grid(axis="y", which="minor", color="#999999", linestyle="-", alpha=0.2, zorder=3)
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
