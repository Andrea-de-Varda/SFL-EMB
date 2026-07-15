#!/usr/bin/env python3
"""Significance for the layer-wise EMB vs standard-SFL comparison.

For every (dataset, layer) cell, tests EMB at that layer against the **standard SFL**
predictor using the paper's own test: a paired permutation test over the per-fold r
values, enumerating all 2^n sign flips (src/significance.permutation_statistic).

SFL is layer-independent, so its fold vector is taken once per dataset from the
layer-24 cell -- the same folds as every EMB cell, which is what makes the test
paired. That cell is asserted equal to the main pipeline's SFL for gpt2_medium.

Two-sided at .025, matching the criterion used for the paper figures.

    python appendix/layerwise_significance.py
        -> appendix/layerwise_significance.csv  + a per-measurement-type summary
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.significance import permutation_statistic

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "appendix/layerwise_logitlens.csv"
OUT = ROOT / "appendix/layerwise_significance.csv"

STEM = {"Provo": "provo_fpass", "Meco-en": "meco_fpass", "ZuCo-2": "zuco_fpass",
        "UCL_ET": "ucl_et_fpass", "UCL_SPR": "ucl_spr", "Brown": "brown",
        "NatStor_SPR": "natstor_spr", "NatStor_Maze": "maze_maze",
        "UCL_N400": "ucl_n400_n400", "Pereira2018": "pereira", "Tuckute2024": "control",
        "NatStor_fMRI": "natstorfmri", "Wehbe2014": "HP"}
TYPE = {**{k: "ET" for k in ["Provo", "Meco-en", "ZuCo-2", "UCL_ET"]},
        **{k: "SPR" for k in ["UCL_SPR", "Brown", "NatStor_SPR"]},
        "NatStor_Maze": "Maze",
        **{k: "brain" for k in ["UCL_N400", "Pereira2018", "Tuckute2024",
                                "NatStor_fMRI", "Wehbe2014"]}}
ORDER = ["Provo", "Meco-en", "ZuCo-2", "UCL_ET", "UCL_SPR", "Brown", "NatStor_SPR",
         "NatStor_Maze", "UCL_N400", "Pereira2018", "Tuckute2024", "NatStor_fMRI",
         "Wehbe2014"]


def sig(p):
    """Two-sided at .025, as used for the paper figures."""
    return min(p, 1 - p) <= 0.025


def rng(ls):
    if not len(ls):
        return "-"
    ls = sorted(ls)
    out, s, prev = [], ls[0], ls[0]
    for v in ls[1:]:
        if v == prev + 1:
            prev = v
        else:
            out.append((s, prev)); s = prev = v
    out.append((s, prev))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV} -- run appendix/layerwise_logitlens.py --merge first")
    d = pd.read_csv(CSV)
    if "emb_folds" not in d.columns:
        sys.exit("this csv has no per-fold r; re-run the sweep with the current script")

    rows, skipped = [], []
    for nm in ORDER:
        g = d[d.dataset == nm].sort_values("layer")
        g = g.dropna(subset=["emb_folds", "sfl_folds"])
        if not len(g):
            skipped.append(nm)          # not yet re-run with per-fold r
            continue
        last = g[g.layer == g.layer.max()].iloc[0]
        sfl_folds = json.loads(last.sfl_folds)          # standard SFL, layer-independent
        m = pd.read_csv(ROOT / f"results/{STEM[nm]}_results.csv").set_index("model")
        assert abs(float(np.mean(sfl_folds)) - float(m.loc["gpt2_medium", "sfl"])) < 1e-4, \
            f"{nm}: layer-24 SFL folds do not match the main pipeline"
        for _, r in g.iterrows():
            emb_folds = json.loads(r.emb_folds)
            assert len(emb_folds) == len(sfl_folds), f"{nm} L{r.layer}: fold mismatch"
            p = permutation_statistic(emb_folds, sfl_folds)   # one-sided: EMB > SFL
            rows.append(dict(dataset=nm, type=TYPE[nm], layer=int(r.layer),
                             emb_r=r.emb_r, sfl_r=float(np.mean(sfl_folds)),
                             delta=r.emb_r - float(np.mean(sfl_folds)),
                             p=p, sig=sig(p), emb_wins=r.emb_r > float(np.mean(sfl_folds))))
    if not rows:
        sys.exit("no dataset has per-fold r yet -- re-run the sweep "
                 "(sbatch cluster/slurm_layerwise.sh)")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    if skipped:
        print(f"SKIPPED (no per-fold r yet, re-run needed): {', '.join(skipped)}\n")
    print("EMB(layer L) vs standard SFL -- paired permutation test over folds, two-sided .025\n")
    print(f"{'dataset':<14}{'type':<6}{'layers EMB>SFL':>15}{'SIG EMB>SFL':>28}"
          f"{'SIG SFL>EMB':>16}{'best delta':>11}")
    for nm in ORDER:
        g = out[out.dataset == nm]
        if not len(g):
            continue
        win = g[g.emb_wins & g.sig].layer.tolist()
        los = g[(~g.emb_wins) & g.sig].layer.tolist()
        b = g.loc[g.delta.idxmax()]
        print(f"{nm:<14}{TYPE[nm]:<6}{int(g.emb_wins.sum()):>12}/25{rng(win):>28}"
              f"{rng(los):>16}{b.delta:>+11.4f}")

    print("\n--- by measurement type ---")
    print(f"{'type':<7}{'cells':>7}{'sig EMB>SFL':>13}{'sig SFL>EMB':>13}{'mean best delta':>17}")
    for t in ["ET", "SPR", "Maze", "brain"]:
        g = out[out.type == t]
        nb = g.groupby("dataset").delta.max().mean()
        print(f"{t:<7}{len(g):>7}{int((g.emb_wins & g.sig).sum()):>13}"
              f"{int(((~g.emb_wins) & g.sig).sum()):>13}{nb:>+17.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
