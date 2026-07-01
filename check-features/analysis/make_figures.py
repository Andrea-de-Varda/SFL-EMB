#!/usr/bin/env python3
"""
Feature-predictivity figure: how well SFL / SFL+RFF / EMB predict 14 sentence
features, semantic panel and form panel side by side, three bars per feature plus a
separated group-mean at the right of each panel.

Run (analysis env):  python check-features/analysis/make_figures.py
Writes check-features/analysis/plots/feature_predictivity.{svg,png}.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/andrea/.claude/skills/scientific-figure-style/scripts")
from figure_style import apply_style, style_spines, save_fig  # noqa: E402
apply_style()

OUT = str(Path(__file__).resolve().parent / "plots")
C_SFL, C_RFF, C_EMB = "#999999", "#DD8452", "#4C72B0"

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


def panel(ax, sub, title):
    sub = sub.sort_values("r_emb", ascending=False).reset_index(drop=True)
    n = len(sub)
    # per-feature x positions 0..n-1, then a gap, then the Mean group at n+0.5
    x = np.concatenate([np.arange(n), [n + 0.5]])
    w = 0.27
    vals = {c: np.append(sub[c].to_numpy(), sub[c].mean()) for c in ["r_base", "r_rff", "r_emb"]}
    ax.bar(x - w, vals["r_base"], w, color=C_SFL, edgecolor="black", linewidth=0.4)
    ax.bar(x,      vals["r_rff"],  w, color=C_RFF, edgecolor="black", linewidth=0.4)
    ax.bar(x + w,  vals["r_emb"],  w, color=C_EMB, edgecolor="black", linewidth=0.4)
    ax.axvline(n - 0.25, color="#cccccc", lw=0.8, zorder=0)          # separate the Mean group
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in sub.feature] + ["Mean"], rotation=40,
                       ha="right", fontsize=9)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_title(title, fontsize=10, weight="bold", loc="left")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(labelsize=9)


def main():
    d = pd.read_csv(ROOT / "check-features" / "results_features.csv")
    g = (d.groupby(["group", "feature"])
         .agg(r_base=("r_base", "mean"), r_rff=("r_sflrff", "mean"), r_emb=("r_act", "mean"))
         .reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=300, sharey=True)
    panel(axes[0], g[g.group == "semantic"], "Semantic features")
    panel(axes[1], g[g.group == "form"], "Form / syntax features")
    axes[0].set_ylabel("Prediction accuracy (Pearson $r$)", fontsize=11)
    handles = [Line2D([0], [0], color=c, lw=6, label=l) for c, l in
               [(C_SFL, "SFL (3-dim)"), (C_RFF, "SFL + RFF"), (C_EMB, "EMB")]]
    axes[0].legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    save_fig(fig, "feature_predictivity", OUT)
    print("wrote", OUT + "/feature_predictivity.{svg,png}")


if __name__ == "__main__":
    main()
