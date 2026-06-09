#!/usr/bin/env python3
"""
Shared helpers for the dimensionality-matched SFL-vs-EMB brain analyses
(Pereira-style sentence-level, plus the TR-level and word-level variants).

Core idea (see probe_Pereira_passages.py for the original): make the SFL predictor
fair to EMB on dimensionality by projecting per-word SFL through random Fourier
features (RBFSampler) to the embedding dim D, fit ONLY on training-fold words
(imputer/scaler/RBF all train-fit) so there is no leakage from held-out data.

Scoring mirrors utils.test_model_Ridge (same alpha grid, Pearson r per fold) and
saves the per-fold r vector as results/<name>_predictions so the inlined
permutation significance test (identical to utils.significance_testing) works.

This module is import-only (no side effects); each dataset script supplies its own
fold structure (sequential KFold / GroupKFold passages / leave-one-story-out / TR
KFold) and its own pooling (sum-per-sentence, sum-per-TR + lag, or per-word).
"""
import copy
import itertools
import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler

RIDGE_ALPHAS = (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)  # as utils
SFL_RFF_GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
RFF_SEED = 0

# --- permutation significance test (verbatim from utils.py; avoids importing
#     utils, whose pygam/fasttext chain isn't in every env) ---
def swap_elements(a, b, indices):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for i in indices:
        a[i], b[i] = b[i], a[i]
    return [a, b]

def generate_swaps(a1, a2):
    swap_indices = []
    for r in range(1, len(a1) + 1):
        swap_indices.extend(itertools.combinations(range(len(a1)), r))
    return [swap_elements(a1, a2, indices) for indices in swap_indices]

def permutation_statistic(a1, a2):
    mean_diff = np.mean(a1) - np.mean(a2)
    diffs = [(np.mean(xa1) - np.mean(xa2)) for xa1, xa2 in generate_swaps(a1, a2)]
    return sum([(d >= mean_diff) for d in diffs]) / 2 ** len(a1)

def significance_testing(name1, name2, thedir=""):
    with open(thedir + 'results/' + name1 + '_predictions', 'rb') as h:
        a1 = pickle.load(h)
    with open(thedir + 'results/' + name2 + '_predictions', 'rb') as h:
        a2 = pickle.load(h)
    return permutation_statistic(a1, a2)

# --- ridge + embedding pooling ---
def fit_ridge(X, y):
    reg = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    reg.fit(X, y)
    return reg

def get_layer(vec, layer, dim):
    return list(vec[layer * dim:(layer + 1) * dim])

def get_act(act_list, layer, dim, pool):
    """Sentence-level embedding by pooling token activations (dropping the first
    token, as in the original scripts). pool in {'mean','sum'}."""
    out = []
    for act_sent in act_list:
        v = act_sent[1:]
        v = v.mean(axis=0) if pool == "mean" else v.sum(axis=0)
        out.append(get_layer(v, layer, dim))
    return np.array(out)

# --- folds: sequential KFold (groups=None) or GroupKFold (groups given) ---
def make_folds(n, groups=None, n_splits=10):
    idx = np.arange(n)
    if groups is None:
        kf = KFold(n_splits=n_splits)                 # sequential, like utils
        for _, test in kf.split(idx):
            yield test
    else:
        gkf = GroupKFold(n_splits=n_splits)
        for _, test in gkf.split(idx, groups=groups):
            yield test

# --- leakage-free SFL random-Fourier-features (fit on TRAIN words only) ---
def fit_sfl_rff(word_sfl, wsi, train_units, gamma, n_components):
    mask = np.isin(wsi, train_units)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    train_words = scaler.fit_transform(imputer.fit_transform(word_sfl[mask]))
    rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=RFF_SEED)
    rff.fit(train_words)
    return imputer, scaler, rff

def words_to_units(word_sfl, wsi, unit_ids, imputer, scaler, rff, pooling):
    """Transform per-word SFL through the (train-fit) RFF and pool to units.
    Returns (len(unit_ids), n_components). pooling in {'sum','mean'}. Units with
    no words get a zero row (TR datasets can have wordless TRs)."""
    mask = np.isin(wsi, unit_ids)
    word_rff = rff.transform(scaler.transform(imputer.transform(word_sfl[mask])))
    n_comp = int(rff.n_components)
    out = np.zeros((len(unit_ids), n_comp), dtype=np.float64)
    counts = np.zeros(len(unit_ids), dtype=np.float64)
    row_of = {int(u): i for i, u in enumerate(unit_ids)}
    for vector, uid in zip(word_rff, wsi[mask].astype(np.int64)):
        r = row_of[int(uid)]
        out[r] += vector
        counts[r] += 1
    if pooling == "mean":
        nz = counts > 0
        out[nz] /= counts[nz, None]
    return out

def save_predictions(rs, saveto):
    if saveto is not None:
        with open(f"results/{saveto}_predictions", "wb") as h:
            pickle.dump(rs, h, protocol=pickle.HIGHEST_PROTOCOL)
    return float(np.mean(rs)), float(np.std(rs))

# ============================================================================
# SENTENCE-LEVEL scoring (Pereira / Tuckute): one response per sentence; the RFF
# per-word features are SUM-pooled per sentence. groups=None -> sequential KFold;
# groups=<passage labels> -> GroupKFold. Unit ids == sentence positions (sentnum).
# ============================================================================
def score_matrix(X, y, saveto=None, groups=None, n_splits=10):
    rs = []
    for test in make_folds(len(y), groups, n_splits):
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        reg = fit_ridge(X[train], y[train])
        rs.append(pearsonr(y[test], reg.predict(X[test]))[0])
    return save_predictions(rs, saveto)

def _select_gamma_sentence(word_sfl, wsi, y, train_units, groups, n_components, inner_splits=5):
    losses = {g: [] for g in SFL_RFF_GAMMAS}
    inner_groups = None if groups is None else groups[train_units]
    for vpos in make_folds(len(train_units), inner_groups, inner_splits):
        inner_val = train_units[vpos]
        inner_train = np.setdiff1d(train_units, inner_val, assume_unique=True)
        ys = StandardScaler()
        y_tr = ys.fit_transform(y[inner_train, None]).ravel()
        y_va = ys.transform(y[inner_val, None]).ravel()
        for g in SFL_RFF_GAMMAS:
            imp, sca, rff = fit_sfl_rff(word_sfl, wsi, inner_train, g, n_components)
            x_tr = words_to_units(word_sfl, wsi, inner_train, imp, sca, rff, "sum")
            x_va = words_to_units(word_sfl, wsi, inner_val, imp, sca, rff, "sum")
            losses[g].append(float(np.mean((fit_ridge(x_tr, y_tr).predict(x_va) - y_va) ** 2)))
    return float(min(losses, key=lambda g: (np.mean(losses[g]), g)))

def score_sfl_rff_sentence(word_sfl, wsi, y, emb_sum, n_components, groups=None,
                           name_rff=None, name_all=None, n_splits=10):
    """sfl_rff (SUM-pooled per sentence) and all_matched = concat(emb_sum, sfl_rff).
    Same outer folds as score_matrix -> paired per-fold r."""
    rs_rff, rs_all = [], []
    for test in make_folds(len(y), groups, n_splits):
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        gamma = _select_gamma_sentence(word_sfl, wsi, y, train, groups, n_components)
        imp, sca, rff = fit_sfl_rff(word_sfl, wsi, train, gamma, n_components)
        x_tr = words_to_units(word_sfl, wsi, train, imp, sca, rff, "sum")
        x_te = words_to_units(word_sfl, wsi, test, imp, sca, rff, "sum")
        rs_rff.append(pearsonr(y[test], fit_ridge(x_tr, y[train]).predict(x_te))[0])
        xa_tr = np.hstack([emb_sum[train], x_tr])
        xa_te = np.hstack([emb_sum[test], x_te])
        rs_all.append(pearsonr(y[test], fit_ridge(xa_tr, y[train]).predict(xa_te))[0])
    return save_predictions(rs_rff, name_rff), save_predictions(rs_all, name_all)
