"""SFL decomposition (paper appendix A2): the independent contribution of
surprisal / frequency / length.

For a dataset's SFL design matrix, score all 7 non-empty subsets of {S, F, L}.
A feature *type* is selected jointly across any spillover lags / TR-lag blocks
(you enumerate over the three types, not over lags) — so for a 9-column SPR
spillover matrix, "S" means all three surprisal columns together.

Per the paper this is run for GPT-2 355M (``gpt2_medium``) only; the drivers call
:func:`maybe_decompose`, which is a no-op for the other models. Output mirrors the
5-condition tables: ``results/<prefix>_decomposed.csv`` (rows m, sd; columns
S F L SF FL SL SFL).
"""
from pathlib import Path

import numpy as np

from . import fit

DECOMPOSE_MODEL = "gpt2_medium"
SUBSETS = {"S": "s", "F": "f", "L": "l", "SF": "sf", "FL": "fl", "SL": "sl", "SFL": "sfl"}


def sfl_layout(n_blocks):
    """Column indices of each feature type for an SFL matrix laid out as
    ``n_blocks`` consecutive [S, F, L] triples (1 block word/sentence, 3 SPR
    spillover, 4 TR lags)."""
    return {"s": [3 * k for k in range(n_blocks)],
            "f": [3 * k + 1 for k in range(n_blocks)],
            "l": [3 * k + 2 for k in range(n_blocks)]}


def decompose(X_sfl, y, folds, prefix, n_blocks, results_dir="results",
              backend="himalaya"):
    """Score the 7 feature-type subsets; write ``<prefix>_decomposed.csv``."""
    import pandas as pd

    X_sfl = np.asarray(X_sfl, dtype=np.float64)
    idx = sfl_layout(n_blocks)
    out = {}
    for name, types in SUBSETS.items():
        cols = sorted(c for t in types for c in idx[t])
        m, sd, _, _ = fit.score(X_sfl[:, cols], y, folds, saveto=None,
                                backend=backend, warn_boundary=False)
        out[name] = [m, sd]
    df = pd.DataFrame(out, index=["m", "sd"])
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(results_dir) / f"{prefix}_decomposed.csv")
    return df


def maybe_decompose(model, X_sfl, y, folds, prefix, n_blocks, results_dir="results",
                    backend="himalaya"):
    """Run the decomposition only for the designated model (A2 = gpt2_medium)."""
    if model == DECOMPOSE_MODEL:
        decompose(X_sfl, y, folds, prefix, n_blocks, results_dir, backend)
