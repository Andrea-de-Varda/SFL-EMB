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
def _fit_predict(X_tr, X_te, y_tr, backend):
    """Standardize X and y on train, fit ridge, return (test prediction, alpha).

    The prediction is in the standardized-y space; that is fine for Pearson r
    (affine-invariant) and for the relative ordering used by OOF interpretation.
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
        return reg.predict(X_te), float(reg.alpha_)
    elif backend == "himalaya":
        # himalaya RidgeCV matches sklearn RidgeCV to ~1e-4 in BOTH the n>p and
        # n<p regimes (verified). KernelRidgeCV is faster for n<<p but its inner
        # alpha selection can diverge in small folds, so we use RidgeCV throughout
        # for exact parity; it stays GPU-fast at these sizes (p up to ~3200).
        from himalaya.ridge import RidgeCV as HRidgeCV

        reg = HRidgeCV(alphas=ALPHAS, cv=INNER_CV, fit_intercept=False)
        reg.fit(_to_backend(X_tr), _to_backend(y_trs[:, None]))
        y_pred = _from_backend(reg.predict(_to_backend(X_te))).squeeze()
        return y_pred, float(_from_backend(reg.best_alphas_).ravel()[0])
    raise ValueError(f"unknown backend {backend!r}")


def _score_fold(X_tr, X_te, y_tr, y_te, backend):
    """Fit on train, return (Pearson r vs held-out y, selected alpha)."""
    y_pred, alpha = _fit_predict(X_tr, X_te, y_tr, backend)
    y_te = np.asarray(y_te, dtype=np.float64).ravel()
    if np.std(y_pred) == 0 or np.std(y_te) == 0:
        return 0.0, alpha
    return float(pearsonr(y_te, y_pred)[0]), alpha


# --------------------------------------------------------------------------- #
# banded ridge — a SEPARATE ridge penalty per feature block (SFL vs EMB, or
# RFF vs EMB). A single global alpha over-regularizes a small strong block (the 3
# SFL scalars) to control a large weak block (768-1600 EMB dims), so the
# concatenation can score BELOW the strong block alone. Banded ridge fixes this and
# guarantees all >= max(components). For our exactly-two-block concatenations we
# select the block2/block1 penalty RATIO on a deterministic grid by inner CV, then
# the overall alpha as usual -> reproducible, fast, no random search. See README.
# --------------------------------------------------------------------------- #
BANDED_RATIOS = tuple(np.logspace(-3, 3, 7))


def _ridgecv_fit(X_tr, y_tr, backend):
    """Fit RidgeCV; return (inner-CV score, alpha, predict-fn on new X)."""
    if backend == "sklearn":
        from sklearn.linear_model import RidgeCV

        reg = RidgeCV(alphas=ALPHAS, cv=KFold(INNER_CV, shuffle=False),
                      fit_intercept=False).fit(X_tr, y_tr)
        return float(getattr(reg, "best_score_", np.nan)), float(reg.alpha_), reg.predict
    from himalaya.ridge import RidgeCV as HRidgeCV

    reg = HRidgeCV(alphas=ALPHAS, cv=INNER_CV, fit_intercept=False)
    reg.fit(_to_backend(X_tr), _to_backend(y_tr[:, None]))
    score = float(np.asarray(_from_backend(reg.cv_scores_)).mean())
    alpha = float(_from_backend(reg.best_alphas_).ravel()[0])
    return score, alpha, lambda X: _from_backend(reg.predict(_to_backend(X))).squeeze()


def _banded_fit_predict(X_tr, X_te, y_tr, split, backend):
    """Two-block banded ridge (block boundary at column ``split``). Standardize,
    pick the between-block penalty ratio by inner CV, refit, predict."""
    xs, ys = StandardScaler(), StandardScaler()
    X_tr = xs.fit_transform(X_tr)
    X_te = xs.transform(X_te)
    y_trs = ys.fit_transform(np.asarray(y_tr, dtype=np.float64)[:, None]).ravel()
    best = (-np.inf, 1.0)
    for g in BANDED_RATIOS:
        s = X_tr.copy()
        s[:, split:] *= g
        score, _, _ = _ridgecv_fit(s, y_trs, backend)
        if score > best[0]:
            best = (score, g)
    g = best[1]
    s_tr = X_tr.copy(); s_tr[:, split:] *= g
    s_te = X_te.copy(); s_te[:, split:] *= g
    _, alpha, predict = _ridgecv_fit(s_tr, y_trs, backend)
    return predict(s_te), (g, alpha)


def _score_fold_banded(X_tr, X_te, y_tr, y_te, split, backend):
    y_pred, info = _banded_fit_predict(X_tr, X_te, y_tr, split, backend)
    y_te = np.asarray(y_te, dtype=np.float64).ravel()
    if np.std(y_pred) == 0 or np.std(y_te) == 0:
        return 0.0, info
    return float(pearsonr(y_te, y_pred)[0]), info


# --------------------------------------------------------------------------- #
# the encoder
# --------------------------------------------------------------------------- #
def run_folds(build_fn, y, folds, backend="himalaya", banded_split=None):
    """Score a condition over ``folds``.

    ``build_fn(train_idx, test_idx) -> (X_tr, X_te)`` returns the (raw, un-scaled)
    design matrices for a fold. For plain conditions it slices a fixed matrix;
    for SFL_RFF it fits gamma/scaler/RBFSampler on the train rows and projects.
    ``banded_split`` (an int) turns on two-block banded ridge with the block
    boundary at that column (used for the SFL+EMB / RFF+EMB concatenations).
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
        if banded_split:
            r, alpha = _score_fold_banded(X_tr, X_te, y[train], y[test], banded_split, backend)
        else:
            r, alpha = _score_fold(X_tr, X_te, y[train], y[test], backend)
        fold_rs.append(r)
        fold_alphas.append(alpha)
    return fold_rs, fold_alphas


def oof_predict(y, folds, X=None, build_fn=None, backend="himalaya"):
    """Leave-fold-out predictions for every item (for interpretation, not scoring).

    Same per-fold standardized ridge as :func:`run_folds`; returns a length-n array
    of held-out predictions (standardized-y space). Provide a fixed ``X`` or a
    per-fold ``build_fn`` (e.g. pooled RFF).
    """
    if backend == "himalaya":
        setup_himalaya()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if build_fn is None:
        build_fn = _slicer(X)
    pred = np.full(n, np.nan)
    for test in folds:
        test = np.asarray(test)
        train = np.setdiff1d(np.arange(n), test, assume_unique=False)
        X_tr, X_te = build_fn(train, test)
        pred[test], _ = _fit_predict(X_tr, X_te, y[train], backend)
    return pred


def _slicer(X):
    X = np.asarray(X, dtype=np.float64)
    return lambda train, test: (X[train], X[test])


def score(X=None, y=None, folds=None, saveto=None, results_dir="results",
          backend="himalaya", build_fn=None, warn_boundary=True, banded_split=None):
    """Run a condition and (optionally) persist the per-fold r vector.

    Provide either a fixed ``X`` (plain condition) or a ``build_fn`` (RFF). Saves
    to ``<results_dir>/<saveto>_predictions`` (pickle of the fold-r list), the
    format ``significance.significance_testing`` reads. ``banded_split`` enables
    two-block banded ridge (SFL+EMB / RFF+EMB). Returns
    ``(mean_r, sd_r, fold_rs, fold_alphas)``.
    """
    if build_fn is None:
        if X is None:
            raise ValueError("provide X or build_fn")
        build_fn = _slicer(X)
    fold_rs, fold_alphas = run_folds(build_fn, y, folds, backend=backend,
                                     banded_split=banded_split)
    scalar_alphas = [a for a in fold_alphas if np.isscalar(a)]
    if warn_boundary and any(a in (ALPHAS[0], ALPHAS[-1]) for a in scalar_alphas):
        warnings.warn(f"[{saveto}] alpha hit grid boundary in some folds "
                      f"({[f'{a:.0e}' for a in fold_alphas]}); consider widening ALPHAS.")
    if saveto is not None:
        d = Path(results_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{saveto}_predictions", "wb") as h:
            pickle.dump(fold_rs, h, protocol=pickle.HIGHEST_PROTOCOL)
    return float(np.mean(fold_rs)), float(np.std(fold_rs)), fold_rs, fold_alphas
