"""Ridge encoding with interchangeable sklearn / himalaya backends.

Fitting convention (confirmed 2026-07, see PLAN.md):
  * Outer CV: 10-fold **sequential** (or grouped, e.g. Pereira passages) - caller
    supplies the fold test-index lists via ``make_folds``.
  * Inner alpha selection: ``KFold(5, shuffle=False)`` over a widened log grid.
  * Per-fold ``StandardScaler`` on X and y, fit on the TRAIN fold only.
  * ``fit_intercept=False`` (data is centered by the scaler) -> the two backends
    agree up to float precision.
  * Metric: Pearson r between held-out y and prediction, per fold.

The himalaya backend is the default (GPU when available). Per fold it uses
``RidgeCV`` when ``n_samples > n_features`` and ``KernelRidgeCV`` (linear kernel)
otherwise, which is much faster for the high-dimensional EMB / SFL_RFF+EMB
conditions. sklearn always uses ``RidgeCV`` (the reference backend for the
published numbers, up to the per-fold-scaling change).

Everything runs through :func:`run_folds`, which takes a per-fold ``build_fn`` so
that conditions needing train-only feature fitting (SFL_RFF: gamma/scaler fit on
the train fold) share the exact same leakage-safe loop as the plain conditions.
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

# Widened alpha grid (was 1e-5..1e4 x10 in the paper). 13 log-spaced points.
ALPHAS = tuple(np.logspace(-6, 6, 13))
INNER_CV = 5

_HIMALAYA_READY = False
_DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# backend setup
# --------------------------------------------------------------------------- #
def setup_himalaya():
    """Set the himalaya backend once (torch_cuda if a GPU is present)."""
    global _HIMALAYA_READY, _DEVICE
    if _HIMALAYA_READY:
        return _DEVICE
    import torch
    from himalaya.backend import set_backend

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    set_backend("torch_cuda" if _DEVICE == "cuda" else "numpy")
    _HIMALAYA_READY = True
    return _DEVICE


def _to_backend(a):
    if _DEVICE != "cuda":
        return np.asarray(a, dtype=np.float32)
    import torch

    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to("cuda")


def _from_backend(a):
    try:
        import torch

        if torch.is_tensor(a):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def make_folds(n, groups=None, n_splits=10):
    """Return a list of test-index arrays.

    ``groups=None`` -> sequential ``KFold`` (paper default). Passing group labels
    -> ``GroupKFold`` (used only for Pereira, grouped by passage).
    """
    idx = np.arange(n)
    if groups is None:
        return [test for _, test in KFold(n_splits=n_splits, shuffle=False).split(idx)]
    return [test for _, test in GroupKFold(n_splits=n_splits).split(idx, groups=groups)]


# --------------------------------------------------------------------------- #
# per-fold regressor (shared by every condition)
# --------------------------------------------------------------------------- #
def _score_fold(X_tr, X_te, y_tr, y_te, backend):
    """Standardize on train, fit ridge, return (Pearson r, selected alpha).

    ``y_te`` is left in its raw scale: Pearson r is invariant to the affine
    scaling of the target, so only ``y_tr`` needs centering for the fit.
    """
    xs, ys = StandardScaler(), StandardScaler()
    X_tr = xs.fit_transform(X_tr)
    X_te = xs.transform(X_te)
    y_trs = ys.fit_transform(np.asarray(y_tr, dtype=np.float64)[:, None]).ravel()

    if backend == "sklearn":
        from sklearn.linear_model import RidgeCV

        reg = RidgeCV(alphas=ALPHAS, cv=KFold(INNER_CV, shuffle=False),
                      fit_intercept=False)
        reg.fit(X_tr, y_trs)
        y_pred = reg.predict(X_te)
        alpha = float(reg.alpha_)
    elif backend == "himalaya":
        # himalaya RidgeCV matches sklearn RidgeCV to ~1e-4 in BOTH the n>p and
        # n<p regimes (verified). KernelRidgeCV is faster for n<<p but its inner
        # alpha selection can diverge in small folds, so we use RidgeCV throughout
        # for exact parity; it stays GPU-fast at these sizes (p up to ~3200).
        from himalaya.ridge import RidgeCV as HRidgeCV

        reg = HRidgeCV(alphas=ALPHAS, cv=INNER_CV, fit_intercept=False)
        reg.fit(_to_backend(X_tr), _to_backend(y_trs[:, None]))
        y_pred = _from_backend(reg.predict(_to_backend(X_te))).squeeze()
        alpha = float(_from_backend(reg.best_alphas_).ravel()[0])
    else:
        raise ValueError(f"unknown backend {backend!r}")

    y_te = np.asarray(y_te, dtype=np.float64).ravel()
    if np.std(y_pred) == 0 or np.std(y_te) == 0:
        return 0.0, alpha
    return float(pearsonr(y_te, y_pred)[0]), alpha


# --------------------------------------------------------------------------- #
# the encoder
# --------------------------------------------------------------------------- #
def run_folds(build_fn, y, folds, backend="himalaya"):
    """Score a condition over ``folds``.

    ``build_fn(train_idx, test_idx) -> (X_tr, X_te)`` returns the (raw, un-scaled)
    design matrices for a fold. For plain conditions it slices a fixed matrix;
    for SFL_RFF it fits gamma/scaler/RBFSampler on the train rows and projects.
    Returns ``(fold_rs, fold_alphas)``.
    """
    if backend == "himalaya":
        setup_himalaya()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    fold_rs, fold_alphas = [], []
    for test in folds:
        test = np.asarray(test)
        train = np.setdiff1d(np.arange(n), test, assume_unique=False)
        X_tr, X_te = build_fn(train, test)
        r, alpha = _score_fold(X_tr, X_te, y[train], y[test], backend)
        fold_rs.append(r)
        fold_alphas.append(alpha)
    return fold_rs, fold_alphas


def _slicer(X):
    X = np.asarray(X, dtype=np.float64)
    return lambda train, test: (X[train], X[test])


def score(X=None, y=None, folds=None, saveto=None, results_dir="results",
          backend="himalaya", build_fn=None, warn_boundary=True):
    """Run a condition and (optionally) persist the per-fold r vector.

    Provide either a fixed ``X`` (plain condition) or a ``build_fn`` (RFF). Saves
    to ``<results_dir>/<saveto>_predictions`` (pickle of the fold-r list), the
    format ``significance.significance_testing`` reads. Returns
    ``(mean_r, sd_r, fold_rs, fold_alphas)``.
    """
    if build_fn is None:
        if X is None:
            raise ValueError("provide X or build_fn")
        build_fn = _slicer(X)
    fold_rs, fold_alphas = run_folds(build_fn, y, folds, backend=backend)
    if warn_boundary and any(a in (ALPHAS[0], ALPHAS[-1]) for a in fold_alphas):
        warnings.warn(f"[{saveto}] alpha hit grid boundary in some folds "
                      f"({[f'{a:.0e}' for a in fold_alphas]}); consider widening ALPHAS.")
    if saveto is not None:
        d = Path(results_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{saveto}_predictions", "wb") as h:
            pickle.dump(fold_rs, h, protocol=pickle.HIGHEST_PROTOCOL)
    return float(np.mean(fold_rs)), float(np.std(fold_rs)), fold_rs, fold_alphas
