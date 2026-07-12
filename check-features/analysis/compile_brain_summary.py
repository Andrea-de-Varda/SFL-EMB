#!/usr/bin/env python3
"""
Compile the dimensionality-matched brain-encoding results into ONE tidy dataset:
per brain dataset x per LM, the prediction accuracy of
  SFL      (base, 3-dim)      SFL+RFF  (sfl_rff, D)      EMB  (emb_sum, D)  [EMB uses SUM pooling]

Reads the per-model matched CSVs (rows 0..4 = GPT, GPT-2 124M/355M/774M/1.5B). Any
dataset whose per-model CSV is not in the repo is reported as MISSING (its numbers were
produced on the cluster but not committed) and back-filled at the DATASET-MEAN level
from RESULTS_dimensionality_matched.md so the overview is still complete.

Run:  python check-features/analysis/compile_brain_summary.py
Writes  brain_summary_permodel.csv, brain_summary_bydataset.csv, plots/brain_summary.{svg,png}.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
try:
    sys.path.insert(0, "/home/andrea/.claude/skills/scientific-figure-style/scripts")
    from figure_style import apply_style, style_spines, save_fig  # noqa: E402
    apply_style()
except Exception:                                # style helper not present -> plain mpl
    def style_spines(ax, *a, **k): [ax.spines[s].set_visible(False) for s in ("top", "right")]
    def save_fig(fig, name, path=".", *a, **k): fig.savefig(f"{path}/{name}.png", bbox_inches="tight")

MODELS = ["GPT", "GPT-2 124M", "GPT-2 355M", "GPT-2 774M", "GPT-2 1.5B"]
# dataset -> per-model matched CSV (rows 0..4 aligned to MODELS). These are the ACTUAL
# output paths of the matched scripts (each writes <dataset_dir>/results/*.csv).
MATCHED = {
    "Pereira":      ROOT / "results/pereira_results.csv",
    "Tuckute":      ROOT / "results/control_results.csv",
    "NatStor-fMRI": ROOT / "results/natstorfmri_results.csv",
    "Wehbe-HP":     ROOT / "results/HP_results.csv",
    "UCL-N400":     ROOT / "results/ucl_n400_n400_results.csv",
}
DOC_MEANS = {}                                  # all datasets have per-model CSVs; no fallback needed
TR_CONFOUNDED = {"NatStor-fMRI", "Wehbe-HP"}   # sum-per-TR pooling confound (see RESULTS doc)


def main():
    long_rows, bydataset, missing = [], [], []
    for ds, path in MATCHED.items():
        if path.is_file():
            d = pd.read_csv(path)
            for i, m in enumerate(MODELS):
                long_rows.append(dict(dataset=ds, model=m,
                                      SFL=d["sfl"].iloc[i], SFL_RFF=d["sfl_rff"].iloc[i],
                                      EMB=d["emb"].iloc[i]))
            bydataset.append(dict(dataset=ds, source="per-model",
                                  SFL=d["sfl"].mean(), SFL_RFF=d["sfl_rff"].mean(),
                                  EMB=d["emb"].mean()))
        else:
            missing.append(ds)
            if ds in DOC_MEANS:
                bydataset.append(dict(dataset=ds, source="dataset-mean (per-model uncommitted)",
                                      **DOC_MEANS[ds]))

    lp = pd.DataFrame(long_rows)
    bd = pd.DataFrame(bydataset)
    for df in (lp, bd):
        for c in ["SFL", "SFL_RFF", "EMB"]:
            df[c] = df[c].round(3)
    lp.to_csv(HERE / "brain_summary_permodel.csv", index=False)
    bd.to_csv(HERE / "brain_summary_bydataset.csv", index=False)

    print("=== per-model rows available for:", sorted(lp.dataset.unique().tolist()))
    if missing:
        print("=== MISSING per-model CSVs (committed only as dataset means):", missing)
        print("    to complete: on Engaging, git add results/results_{control,natstorfmri,HP}_matched.csv "
              "&& commit && push, then rerun this script.")
    print("\n--- brain_summary_bydataset.csv ---")
    print(bd.to_string(index=False))
    print("\n--- brain_summary_permodel.csv (head) ---")
    print(lp.to_string(index=False))

    order = ["Pereira", "Tuckute", "NatStor-fMRI", "Wehbe-HP", "UCL-N400"]
    C = {"SFL": "#999999", "SFL_RFF": "#DD8452", "EMB": "#4C72B0"}
    LEG = [("SFL (3-dim)", C["SFL"]), ("SFL + RFF", C["SFL_RFF"]), ("EMB (sum)", C["EMB"])]
    PLOTS = str(HERE / "plots")

    # ---- Fig A: mean +- SEM across the 5 LMs, with individual model dots overlaid ----
    m = lp.groupby("dataset")[["SFL", "SFL_RFF", "EMB"]].mean().reindex(order)
    sem = (lp.groupby("dataset")[["SFL", "SFL_RFF", "EMB"]]
           .agg(lambda v: np.std(v, ddof=1) / np.sqrt(len(v))).reindex(order))
    x = np.arange(len(order)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=300)
    for k, off in [("SFL", -w), ("SFL_RFF", 0), ("EMB", w)]:
        ax.bar(x + off, m[k], w, yerr=sem[k], color=C[k], edgecolor="black", linewidth=0.4,
               error_kw=dict(lw=1.0, ecolor="black", capsize=2.5), zorder=2)
        for i, ds in enumerate(order):                       # per-model dots
            pts = lp[lp.dataset == ds][k].to_numpy()
            ax.scatter(np.full(len(pts), x[i] + off), pts, s=9, color="black",
                       alpha=0.55, zorder=3, linewidths=0)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Brain predictivity (Pearson $r$)", fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(labelsize=9)
    ax.legend(handles=[Line2D([0], [0], color=c, lw=6, label=l) for l, c in LEG],
              loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Mean $\\pm$ SEM over 5 LMs (dots = individual models)", fontsize=9.5, loc="left")
    fig.tight_layout()
    save_fig(fig, "brain_summary", PLOTS)

    # ---- Fig B: per-model detail — one panel per dataset, 3 predictors over model size ----
    mods = MODELS
    fig, axes = plt.subplots(1, len(order), figsize=(11.5, 2.6), dpi=300, sharey=True)
    for ax, ds in zip(axes, order):
        sub = lp[lp.dataset == ds].set_index("model").reindex(mods)
        for k in ["SFL", "SFL_RFF", "EMB"]:
            ax.plot(range(len(mods)), sub[k], marker="o", ms=4, lw=1.4, color=C[k])
        ax.set_title(ds, fontsize=9.5, weight="bold")
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels(["GPT", "124M", "355M", "774M", "1.5B"], rotation=45, ha="right", fontsize=7.5)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Brain predictivity ($r$)", fontsize=10)
    axes[0].set_ylim(0, 0.55)
    axes[-1].legend(handles=[Line2D([0], [0], color=c, marker="o", lw=1.4, label=l) for l, c in LEG],
                    loc="upper right", frameon=False, fontsize=7.5)
    fig.suptitle("Per-model results (across GPT / GPT-2 124M–1.5B)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, "brain_summary_permodel", PLOTS)
    print("\nwrote plots/brain_summary.{svg,png} (mean±SEM) and brain_summary_permodel.{svg,png}")


if __name__ == "__main__":
    main()
