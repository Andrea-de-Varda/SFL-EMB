#!/usr/bin/env python3
"""Recompute the significance contrasts from the saved per-fold prediction pickles,
WITHOUT refitting any model.

Use this after adding a new contrast to ``conditions.CONTRASTS`` (e.g.
``p_sflrffemb_sflrff``): every condition's per-fold r vector is already on disk at
``results/<prefix>_<cond>_predictions`` from the last pipeline run, so we just re-run
the permutation tests and rewrite each ``results/<stem>_results.csv`` in place. No GPU,
no re-extraction; runs in seconds.

If the prediction pickles are NOT present (e.g. a fresh checkout), there is nothing to
recompute from -- re-run the probe scripts instead.

    python recompute_significance.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT
from src import conditions

RESULTS = PROJECT_ROOT / "results"


def main():
    col_order = ([c for cond in conditions.CONDITIONS for c in (cond, f"{cond}_sd")]
                 + list(conditions.CONTRASTS))
    updated, skipped = [], []
    for csv in sorted(RESULTS.glob("*_results.csv")):
        stem = csv.name[: -len("_results.csv")]
        df = pd.read_csv(csv, index_col=0)
        changed = False
        for model in df.index:
            prefix = f"{stem}_{model}"
            # need every condition's per-fold pickle to recompute all contrasts
            if not all((RESULTS / f"{prefix}_{cond}_predictions").exists()
                       for cond in conditions.CONDITIONS):
                continue
            for name, value in conditions.significance_row(prefix, results_dir=str(RESULTS)).items():
                df.loc[model, name] = value
            changed = True
        if changed:
            df = df.reindex(columns=[c for c in col_order if c in df.columns
                                     or c in conditions.CONTRASTS])
            df.to_csv(csv)
            updated.append(csv.name)
        else:
            skipped.append(csv.name)
    print(f"updated {len(updated)} results CSV(s): {', '.join(updated) or '(none)'}")
    if skipped:
        print(f"skipped (no prediction pickles on disk): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
