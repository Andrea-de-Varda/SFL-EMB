"""Random Fourier Feature expansion of the SFL predictors (leakage-free).

The SFL_RFF condition projects the low-dimensional SFL predictors into a
``n_components``-dimensional random Fourier feature space (an approximation of an
RBF kernel), so that SFL is given the same dimensional budget as EMB and can
express nonlinear structure.

Convention (confirmed 2026-07, see PLAN.md):
  * ``n_components = D`` (model hidden size) -> SFL_RFF matches EMB; SFL_RFF+EMB = 2D.
  * RBF bandwidth by the **median heuristic** on the standardized TRAIN features,
    uniformly for every dataset (including Pereira).
  * Leakage-free: the imputer, scaler, median-gamma and ``RBFSampler`` are all fit
    on the TRAIN rows/words of the current outer fold only. (``RBFSampler.fit``
    itself only reads ``n_features`` + gamma to sample its random weights, so the
    single data-dependent choice is gamma + the train-fit scaler/imputer.)

Two granularities:
  * per-unit (word / sentence / TR is the modelling unit) -> :func:`fit_rff` +
    :func:`rff_build_fn` / :func:`rff_concat_build_fn`. Used by the behavioral and
    EEG datasets where the unit IS the row of the SFL design matrix.
  * per-word-then-pool (fMRI) -> :func:`fit_rff` on train words, then
    :func:`pool_words_to_units` (sentence sum-pool) or :func:`aggregate_to_tr`
    (TR sum-pool + BOLD lag). Used by the four fMRI datasets.
"""
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

RFF_SEED = 0


def median_heuristic_gamma(X_std, sample=2000, seed=0):
    """gamma = 1/(2 * median pairwise squared distance) on standardized features."""
    if len(X_std) > sample:
        sub = np.random.RandomState(seed).choice(len(X_std), sample, replace=False)
        X_std = X_std[sub]
    med = np.median(pdist(X_std, "sqeuclidean"))
    return float(1.0 / (2.0 * med)) if med > 0 else 1.0


def fit_rff(X_train, n_components, seed=RFF_SEED):
    """Fit imputer+scaler+median-gamma+RBFSampler on TRAIN rows; return transform fn.

    The returned callable maps a raw SFL matrix (any rows, same columns) to its
    ``n_components``-dim RFF representation. ``.gamma`` is attached for logging.
    """
    imp = SimpleImputer(strategy="mean").fit(X_train)
    sca = StandardScaler().fit(imp.transform(X_train))
    Xs = sca.transform(imp.transform(X_train))
    gamma = median_heuristic_gamma(Xs, seed=seed)
    rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=seed).fit(Xs)

    def transform(X):
        return rff.transform(sca.transform(imp.transform(np.asarray(X, dtype=np.float64))))

    transform.gamma = gamma
    return transform


# --------------------------------------------------------------------------- #
# per-unit build_fns for fit.run_folds (unit == row of X_sfl)
# --------------------------------------------------------------------------- #
def rff_build_fn(X_sfl, n_components, seed=RFF_SEED):
    """SFL_RFF: fit RFF on train rows, project train & test rows."""
    X_sfl = np.asarray(X_sfl, dtype=np.float64)

    def build(train, test):
        t = fit_rff(X_sfl[train], n_components, seed)
        return t(X_sfl[train]), t(X_sfl[test])

    return build


def rff_concat_build_fn(X_sfl, X_extra, n_components, seed=RFF_SEED):
    """SFL_RFF+EMB: hstack(RFF(X_sfl), X_extra), RFF fit on train rows only."""
    X_sfl = np.asarray(X_sfl, dtype=np.float64)
    X_extra = np.asarray(X_extra, dtype=np.float64)

    def build(train, test):
        t = fit_rff(X_sfl[train], n_components, seed)
        return (np.hstack([t(X_sfl[train]), X_extra[train]]),
                np.hstack([t(X_sfl[test]), X_extra[test]]))

    return build


# --------------------------------------------------------------------------- #
# per-word-then-pool helpers (fMRI). ``word_unit`` maps each word to its unit id
# (sentence index or TR index); units with no words get a zero row.
# --------------------------------------------------------------------------- #
def pool_words_to_units(word_feats, word_unit, unit_ids, pooling="sum"):
    """Pool per-word features to units. Returns (len(unit_ids), n_features)."""
    word_feats = np.asarray(word_feats, dtype=np.float64)
    word_unit = np.asarray(word_unit)
    k = word_feats.shape[1]
    out = np.zeros((len(unit_ids), k), dtype=np.float64)
    counts = np.zeros(len(unit_ids), dtype=np.float64)
    row_of = {int(u): i for i, u in enumerate(unit_ids)}
    for vec, uid in zip(word_feats, word_unit):
        if int(uid) in row_of:
            r = row_of[int(uid)]
            out[r] += vec
            counts[r] += 1
    if pooling == "mean":
        nz = counts > 0
        out[nz] /= counts[nz, None]
    return out


def _delay_one(mat, d):
    new = np.zeros_like(mat)
    if d > 0:
        new[d:] = mat[:-d]
    elif d < 0:
        new[:d] = mat[-d:]
    else:
        new = mat
    return new


def _impute_na(a):
    return np.where(np.isnan(a), np.nanmean(np.where(np.isnan(a), np.nan, a), axis=0), a)


def aggregate_to_tr(word_feats, word_tr, n_tr, pooling="sum", delays=(1, 2, 3, 4)):
    """Pool per-word features to TRs then concatenate ``delays`` lagged TR blocks.

    Returns (n_tr, k * len(delays)). Wordless TRs -> NaN -> column-mean imputed.
    Mirrors the BOLD-delay handling used across the fMRI scripts.
    """
    word_feats = np.asarray(word_feats, dtype=np.float64)
    ids = np.asarray(word_tr).astype(int)
    k = word_feats.shape[1]
    tr = np.full((n_tr, k), np.nan, dtype=np.float64)
    for i in range(n_tr):
        sel = word_feats[ids == i]
        if len(sel):
            tr[i] = sel.sum(axis=0) if pooling == "sum" else sel.mean(axis=0)
    out = np.concatenate([_delay_one(tr, d) for d in delays], axis=-1)
    # column-mean impute any all-zero/NaN lags
    col_mean = np.nanmean(out, axis=0)
    inds = np.where(np.isnan(out))
    out[inds] = np.take(col_mean, inds[1])
    return out
