"""Paired permutation test on fold-level Pearson r values.

This is the exact test used in the paper (verbatim logic from the original
``utils.py``): for two conditions scored on the SAME outer folds, enumerate all
2**n sign-flips of the paired per-fold r's and report the proportion of permuted
mean-differences >= the observed one.

``score_condition`` (in ``fit.py``) saves each condition's per-fold r vector to
``<results_dir>/<name>_predictions``; ``significance_testing`` reads two of those
and compares them. Naming and file format are kept compatible with the old
pipeline so existing downstream code keeps working.
"""
import copy
import itertools
import pickle
from pathlib import Path

import numpy as np


def _swap_elements(a, b, indices):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for i in indices:
        a[i], b[i] = b[i], a[i]
    return [a, b]


def _generate_swaps(a1, a2):
    swap_indices = []
    for r in range(1, len(a1) + 1):  # exclude the no-swap case (added back below)
        swap_indices.extend(itertools.combinations(range(len(a1)), r))
    return [_swap_elements(a1, a2, idx) for idx in swap_indices]


def permutation_statistic(a1, a2):
    """One-sided p that mean(a1) - mean(a2) is >= chance, over all 2**n sign flips."""
    a1, a2 = list(a1), list(a2)
    mean_diff = np.mean(a1) - np.mean(a2)
    diffs = [np.mean(xa1) - np.mean(xa2) for xa1, xa2 in _generate_swaps(a1, a2)]
    # +1 in the numerator accounts for the held-out identity (no-swap) permutation.
    return (sum(d >= mean_diff for d in diffs) + 1) / 2 ** len(a1)


def two_sided_p(p_one, n):
    """Exact two-sided p from the one-sided P(D >= d) returned by permutation_statistic.

    The 2**n sign-flip distribution is symmetric, so the opposite tail is
    P(D <= d) = 1 - P(D >= d) + 1/2**n (the identity permutation sits in both tails).
    """
    return min(1.0, 2 * min(p_one, 1 - p_one + 1 / 2 ** n))


def benjamini_hochberg(p):
    """Benjamini-Hochberg adjusted p-values (q-values) for a 1-D array of p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def significance_testing(name1, name2, results_dir="results"):
    """Compare two saved conditions by their per-fold r vectors."""
    d = Path(results_dir)
    with open(d / f"{name1}_predictions", "rb") as h:
        a1 = pickle.load(h)
    with open(d / f"{name2}_predictions", "rb") as h:
        a2 = pickle.load(h)
    return permutation_statistic(a1, a2)
