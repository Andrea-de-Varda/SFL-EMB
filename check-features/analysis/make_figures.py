#!/usr/bin/env python3
"""
Figures for: does SFL->RFF encode MEANING, or only FORM? And is EMB's content more
semantic than SFL_RFF even at matched brain-predictivity?

Fig 1  feature_predictivity  -- grouped 3-dot plot (SFL / SFL+RFF / EMB) x 14 features,
       split semantic vs form. The headline.
Fig 2  rff_gain_dumbbell     -- base -> sfl_rff dumbbell per feature (the RFF gain),
       colored by feature group, gray when the gain is not robust across LMs.
Fig 3  brain_agreement       -- from the brain over/under-prediction runs: EMB and
       SFL_RFF reach comparable predictivity yet rank items differently (low
       Spearman agreement), while the two EMB poolings agree almost perfectly.

Run (analysis env):  python check-features/analysis/make_figures.py
Writes check-features/analysis/plots/*.{svg,png}.
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/andrea/.claude/skills/scientific-figure-style/scripts")
from figure_style import apply_style, style_spines, save_fig  # noqa: E402
apply_style()

OUT = str(Path(__file__).resolve().parent / "plots")
# predictor palette (color encodes predictor, independent of feature-group colors)
C_SFL, C_RFF, C_EMB = "#999999", "#DD8452", "#4C72B0"
# feature-group palette (skill: meaning=red, form=blue)
C_MEAN, C_FORM = "#C44E52", "#4C72B0"
NS = "#bfbfbf"

LABELS = {
    "rating_imageability_mean": "Imageability", "rating_physical_mean": "Physical",
    "rating_places_mean": "Places", "rating_others_thoughts_mean": "Others' thoughts",
    "rating_valence_mean": "Valence", "rating_arousal_mean": "Arousal",
    "rating_sense_mean": "Makes sense",
    "avg_word_length": "Word length", "avg_word_freq_zipf": "Word freq (zipf)",
    "rating_frequency_mean": "Frequency (rated)", "rating_conversational_mean": "Conversational",
    "rating_gram_mean": "Grammaticality", "log-prob-pcfg_mean": "PCFG log-prob",
    "dep_length_mean": "Dep. length",
}


def load_features():
    d = pd.read_csv(ROOT / "check-features" / "results_features.csv")
    g = d.groupby(["group", "feature"]).agg(
        r_base=("r_base", "mean"), r_rff=("r_sflrff", "mean"), r_emb=("r_act", "mean"),
        sd_emb=("r_act", "std"),
        sig_rff=("p_sflrff_vs_base", lambda s: int((s < .05).sum()))).reset_index()
    g["gain"] = g.r_rff - g.r_base
    return g


# ── Fig 1: grouped 3-dot predictivity, semantic (top) vs form (bottom) ──────────
def fig_predictivity(g):
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 5.2), dpi=300,
                             gridspec_kw={"height_ratios": [1, 1]}, sharex=True)
    for ax, grp, title in [(axes[0], "semantic", "Semantic features"),
                           (axes[1], "form", "Form / syntax features")]:
        sub = g[g.group == grp].sort_values("r_emb").reset_index(drop=True)
        y = np.arange(len(sub))
        for col, c, lab in [("r_base", C_SFL, "SFL (3)"),
                            ("r_rff", C_RFF, "SFL+RFF"), ("r_emb", C_EMB, "EMB")]:
            ax.scatter(sub[col], y, s=46, color=c, edgecolors="black",
                       linewidths=0.5, zorder=3, label=lab)
        # faint connector base..emb to guide the eye
        for i in range(len(sub)):
            ax.hlines(y[i], sub.r_base[i], sub.r_emb[i], color="#dddddd", lw=1.2, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels([LABELS[f] for f in sub.feature], fontsize=9)
        ax.set_title(title, fontsize=10, weight="bold", loc="left")
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(labelsize=9)
    axes[1].set_xlabel("Prediction accuracy (Pearson $r$)", fontsize=11)
    handles = [Line2D([0], [0], marker="o", lw=0, markerfacecolor=c, markeredgecolor="black",
                      markersize=7, label=l) for c, l in
               [(C_SFL, "SFL (3-dim)"), (C_RFF, "SFL + RFF"), (C_EMB, "EMB")]]
    axes[0].legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    save_fig(fig, "feature_predictivity", OUT)


# ── Fig 2: RFF gain dumbbell (base -> sfl_rff), colored by group, ns grayed ─────
def fig_rff_gain(g):
    d = g.sort_values("gain").reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(4.5, 4.6), dpi=300)
    for i, r in d.iterrows():
        robust = r.sig_rff >= 3                       # >=3/5 LMs p<.05
        gc = (C_MEAN if r.group == "semantic" else C_FORM) if robust else NS
        ax.hlines(y[i], r.r_base, r.r_rff, color=gc, alpha=0.5, lw=3, zorder=2)
        ax.scatter(r.r_base, y[i], s=34, facecolors="white", edgecolors=gc,
                   linewidths=1.2, zorder=3)
        ax.scatter(r.r_rff, y[i], s=42, color=gc, edgecolors="black",
                   linewidths=0.5, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[f] for f in d.feature], fontsize=9)
    ax.set_xlabel("Prediction accuracy (Pearson $r$)", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(labelsize=9)
    handles = [
        Line2D([0], [0], marker="o", lw=0, markerfacecolor="white", markeredgecolor="black",
               markersize=7, label="SFL (3-dim)"),
        Line2D([0], [0], marker="o", lw=0, markerfacecolor="black", markeredgecolor="black",
               markersize=7, label="SFL + RFF"),
        Line2D([0], [0], color=C_MEAN, lw=3, alpha=0.6, label="Δ semantic (robust)"),
        Line2D([0], [0], color=C_FORM, lw=3, alpha=0.6, label="Δ form (robust)"),
        Line2D([0], [0], color=NS, lw=3, alpha=0.6, label="Δ not robust"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    ax.set_title("RFF gain over raw SFL", fontsize=10, weight="bold", loc="left")
    fig.tight_layout()
    save_fig(fig, "rff_gain_dumbbell", OUT)


# ── Fig 3: brain-side — comparable predictivity, different content ──────────────
def fig_brain_agreement():
    dsets = [("pereira", "Pereira"), ("control", "Tuckute"), ("n400", "UCL-N400")]
    pred = {}
    for key, _ in dsets:
        c = pd.read_csv(ROOT / "results_extremes" / f"{key}_xl_predictions.csv")
        pred[key] = dict(
            r_emb=pearsonr(c.pred_act, c.y)[0], r_rff=pearsonr(c.pred_sfl_rff, c.y)[0],
            agr_embemb=spearmanr(c.pred_act, c.pred_emb_sum)[0],
            agr_embrff=spearmanr(c.pred_act, c.pred_sfl_rff)[0])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), dpi=300)
    x = np.arange(len(dsets)); w = 0.38
    # (A) predictivity: EMB vs SFL+RFF
    ax = axes[0]
    ax.bar(x - w/2, [pred[k]["r_emb"] for k, _ in dsets], w, color=C_EMB,
           edgecolor="black", linewidth=0.5, label="EMB")
    ax.bar(x + w/2, [pred[k]["r_rff"] for k, _ in dsets], w, color=C_RFF,
           edgecolor="black", linewidth=0.5, label="SFL+RFF")
    ax.set_ylabel("Brain predictivity ($r$)", fontsize=11)
    ax.set_title("Comparable predictivity", fontsize=10, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    # (B) agreement: EMB-internal vs EMB-vs-RFF
    ax = axes[1]
    ax.bar(x - w/2, [pred[k]["agr_embemb"] for k, _ in dsets], w, color="#6b6b6b",
           edgecolor="black", linewidth=0.5, label="EMB vs EMB$_{sum}$")
    ax.bar(x + w/2, [pred[k]["agr_embrff"] for k, _ in dsets], w, color=C_RFF,
           edgecolor="black", linewidth=0.5, label="EMB vs SFL+RFF")
    ax.set_ylabel("Predicted-response agreement\n(Spearman $\\rho$)", fontsize=10)
    ax.set_title("...but different content", fontsize=10, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in dsets], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    save_fig(fig, "brain_agreement", OUT)
    return pred


def main():
    g = load_features()
    fig_predictivity(g)
    fig_rff_gain(g)
    pred = fig_brain_agreement()
    print("wrote figures to", OUT)
    print("\nbrain predictivity / agreement:")
    for k, v in pred.items():
        print(f"  {k:8s} r_emb={v['r_emb']:.3f} r_rff={v['r_rff']:.3f} "
              f"agr(emb,emb_sum)={v['agr_embemb']:.3f} agr(emb,rff)={v['agr_embrff']:.3f}")


if __name__ == "__main__":
    main()
