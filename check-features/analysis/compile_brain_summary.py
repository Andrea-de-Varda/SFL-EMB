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
sys.path.insert(0, "/home/andrea/.claude/skills/scientific-figure-style/scripts")
from figure_style import apply_style, style_spines, save_fig  # noqa: E402
apply_style()

MODELS = ["GPT", "GPT-2 124M", "GPT-2 355M", "GPT-2 774M", "GPT-2 1.5B"]
# dataset -> per-model matched CSV (rows 0..4 aligned to MODELS)
MATCHED = {
    "Pereira":      ROOT / "pereira_1/results/results.csv",
    "Tuckute":      ROOT / "results/results_control_matched.csv",
    "NatStor-fMRI": ROOT / "results/results_natstorfmri_matched.csv",
    "Wehbe-HP":     ROOT / "results/results_HP_matched.csv",
    "UCL-N400":     ROOT / "results/results_UCL_N400_matched.csv",
}
# dataset-mean fallback (mean over 5 LMs) from RESULTS_dimensionality_matched.md, EMB = emb_sum
DOC_MEANS = {
    "Tuckute":      dict(SFL=0.30, SFL_RFF=0.37, EMB_sum=0.41),
    "NatStor-fMRI": dict(SFL=0.29, SFL_RFF=0.26, EMB_sum=0.31),
    "Wehbe-HP":     dict(SFL=0.13, SFL_RFF=0.11, EMB_sum=0.20),
}
TR_CONFOUNDED = {"NatStor-fMRI", "Wehbe-HP"}   # sum-per-TR pooling confound (see RESULTS doc)


def main():
    long_rows, bydataset, missing = [], [], []
    for ds, path in MATCHED.items():
        if path.is_file():
            d = pd.read_csv(path)
            for i, m in enumerate(MODELS):
                long_rows.append(dict(dataset=ds, model=m,
                                      SFL=d["base"].iloc[i], SFL_RFF=d["sfl_rff"].iloc[i],
                                      EMB_sum=d["emb_sum"].iloc[i]))
            bydataset.append(dict(dataset=ds, source="per-model",
                                  SFL=d["base"].mean(), SFL_RFF=d["sfl_rff"].mean(),
                                  EMB_sum=d["emb_sum"].mean()))
        else:
            missing.append(ds)
            if ds in DOC_MEANS:
                bydataset.append(dict(dataset=ds, source="dataset-mean (per-model uncommitted)",
                                      **DOC_MEANS[ds]))

    lp = pd.DataFrame(long_rows)
    bd = pd.DataFrame(bydataset)
    for df in (lp, bd):
        for c in ["SFL", "SFL_RFF", "EMB_sum"]:
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

    # overview barplot: datasets x {SFL, SFL+RFF, EMB(sum)}
    order = ["Pereira", "Tuckute", "NatStor-fMRI", "Wehbe-HP", "UCL-N400"]
    bd = bd.set_index("dataset").reindex(order).reset_index()
    x = np.arange(len(order)); w = 0.27
    C = {"SFL": "#999999", "SFL_RFF": "#DD8452", "EMB_sum": "#4C72B0"}
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=300)
    mean_only = bd["source"].str.startswith("dataset-mean").fillna(False).to_numpy()
    for k, off in [("SFL", -w), ("SFL_RFF", 0), ("EMB_sum", w)]:
        # solid where per-model, hatched where dataset-mean-only
        ax.bar(x[~mean_only] + off, bd[k][~mean_only], w, color=C[k], edgecolor="black", linewidth=0.4)
        ax.bar(x[mean_only] + off, bd[k][mean_only], w, color=C[k], edgecolor="black",
               linewidth=0.4, hatch="////", alpha=0.85)
    labels = [f"{o} *" if mean_only[i] else o for i, o in enumerate(order)]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Brain predictivity (Pearson $r$)", fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(labelsize=9)
    handles = [Line2D([0], [0], color=c, lw=6, label=l) for l, c in
               [("SFL (3-dim)", C["SFL"]), ("SFL + RFF", C["SFL_RFF"]), ("EMB (sum)", C["EMB_sum"])]]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    save_fig(fig, "brain_summary", str(HERE / "plots"))
    print("\nwrote plots/brain_summary.{svg,png} (hatched bars = dataset-mean only)")


if __name__ == "__main__":
    main()
