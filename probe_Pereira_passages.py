#!/usr/bin/env python3
"""
Pereira2018 SFL-vs-EMB analysis with (1) PASSAGE-GROUPED cross-validation
(whole passages held out for testing, no passage leakage) and (2) a
dimensionality-matched SFL predictor so the EMB-vs-SFL comparison is fair on
dimensionality.

Replaces the index/sequential CV of probe_Pereira_1.py. Reuses the already-computed
activations (pereira_1/activations_*.json) and per-word SFL (pereira_1/SFL.csv).

Predictors (per model; D = embedding dim):
  base        SFL, mean-pooled                         (3)
  act         EMB, mean-pooled                         (D)
  all         SFL concat EMB                           (3+D)
  emb_sum     EMB, SUM-pooled                          (D)   matched-pooling comparator
  sfl_rff     per-word SFL -> StandardScaler(train) -> RBFSampler(D, grouped-CV gamma)
              -> SUMMED per sentence                   (D)   dimensionality-matched SFL
  all_matched emb_sum concat sfl_rff                    (2D)  matched analogue of `all`

Scoring mirrors utils.test_model_Ridge (RidgeCV over the same alpha grid, Pearson r
per fold) and saves the per-fold r vector as results/<name>_predictions so the
existing utils.significance_testing (2^n paired permutation test) works unchanged.
Outer folds are identical across predictors (GroupKFold is deterministic) -> the
per-fold r vectors are paired.

Output: overwrites pereira_1/results/results.csv keeping the columns plotting.py
needs (base/act/all + _sd + p_base_emb/p_all_base/p_all_emb) and adding the
matched-predictor columns + all-pairs permutation p-values.

NOTE on D: RFF n_components is set to the model embedding dim D. If a future variant
adds spill-over (concatenating previous words' features), the EMB dimensionality grows
to k*D; set n_components to that matched value too.
"""
import sys
import copy
import itertools
import pickle
from os import chdir
from pathlib import Path as _Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())

# ----------------------------- configuration --------------------------------
N_SPLITS       = 10
INNER_SPLITS   = 5                               # grouped inner CV for RFF gamma
RIDGE_ALPHAS   = (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)  # as utils
SFL_RFF_GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
RFF_SEED       = 0
# (model key, surprisal column in SFL.csv, last-layer index, embedding dim D)
MODELS = [("gpt", "s_GPT", 12, 768), ("small", "s_small", 12, 768),
          ("med", "s_med", 24, 1024), ("large", "s_large", 36, 1280),
          ("xl", "s_xl", 48, 1600)]
ACT_FILE = {"gpt": "activations_gpt.json", "small": "activations_gpt2_small.json",
            "med": "activations_gpt2_medium.json", "large": "activations_gpt2_large.json",
            "xl": "activations_gpt2_xl.json"}
ROW_INDEX = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
PREDS = ["base", "act", "all", "emb_sum", "sfl_rff", "all_matched"]
PRED_SUFFIX = {"base": "_base", "act": "", "all": "_all",
               "emb_sum": "_embsum", "sfl_rff": "_sflrff", "all_matched": "_allmatched"}

# --- permutation significance test (copied verbatim from utils.py to keep the
#     method identical while avoiding utils' heavy import chain: pygam/fasttext) ---
def swap_elements(a, b, indices):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for i in indices:
        a[i], b[i] = b[i], a[i]
    return [a, b]

def generate_swaps(a1, a2):
    swap_indices = []
    for r in range(1, len(a1) + 1):
        swap_indices.extend(itertools.combinations(range(len(a1)), r))
    result = []
    for indices in swap_indices:
        result.append(swap_elements(a1, a2, indices))
    return result

def permutation_statistic(a1, a2):
    mean_diff = np.mean(a1) - np.mean(a2)
    swaps = generate_swaps(a1, a2)
    diffs = [(np.mean(xa1) - np.mean(xa2)) for xa1, xa2 in swaps]
    return sum([(diff >= mean_diff) for diff in diffs]) / 2 ** len(a1)

def significance_testing(name1, name2, thedir=""):
    with open(thedir + 'results/' + name1 + '_predictions', 'rb') as handle:
        a1 = pickle.load(handle)
    with open(thedir + 'results/' + name2 + '_predictions', 'rb') as handle:
        a2 = pickle.load(handle)
    return permutation_statistic(a1, a2)

# ----------------------------- feature builders -----------------------------
def get_layer(vec, layer, dim):
    return list(vec[layer * dim:(layer + 1) * dim])

def get_act(act_list, layer, dim, pool):
    """Sentence-level embedding by pooling token activations (dropping the first
    token, as in the original script). pool in {'mean','sum'}."""
    out = []
    for act_sent in act_list:
        v = act_sent[1:]
        v = v.mean(axis=0) if pool == "mean" else v.sum(axis=0)
        out.append(get_layer(v, layer, dim))
    return np.array(out)

def pool_sfl(df, model_col, n):
    """Per-sentence mean-pooled [surprisal, len, freq] -> (n, 3)."""
    return np.array([list(df[df.sentnum == s][[model_col, "len", "freq"]].mean())
                     for s in range(n)])

# ----------------------------- CV machinery ---------------------------------
def grouped_folds(group_labels, n_splits):
    """Yield test-position arrays from GroupKFold (deterministic; whole groups held out)."""
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(group_labels))
    for _, test in gkf.split(idx, groups=group_labels):
        yield test

def fit_ridge(X, y):
    reg = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    reg.fit(X, y)
    return reg

def score_matrix(X, y, group_labels, saveto=None):
    """Fixed sentence-level matrix X: outer GroupKFold, RidgeCV per fold, Pearson r.
    Saves the per-fold r vector as results/<saveto>_predictions when saveto is given."""
    rs = []
    for test in grouped_folds(group_labels, N_SPLITS):
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        reg = fit_ridge(X[train], y[train])
        rs.append(pearsonr(y[test], reg.predict(X[test]))[0])
    if saveto is not None:
        with open(f"results/{saveto}_predictions", "wb") as h:
            pickle.dump(rs, h, protocol=pickle.HIGHEST_PROTOCOL)
    return float(np.mean(rs)), float(np.std(rs))

# ----------------------------- SFL-RFF helpers ------------------------------
def fit_sfl_rff(word_sfl, wsi, train_sent, gamma, n_components):
    mask = np.isin(wsi, train_sent)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    train_words = scaler.fit_transform(imputer.fit_transform(word_sfl[mask]))
    rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=RFF_SEED)
    rff.fit(train_words)
    return imputer, scaler, rff

def aggregate_sfl_rff(word_sfl, wsi, sentence_indices, imputer, scaler, rff, pooling):
    mask = np.isin(wsi, sentence_indices)
    word_rff = rff.transform(scaler.transform(imputer.transform(word_sfl[mask])))
    n_comp = int(rff.n_components)
    out = np.zeros((len(sentence_indices), n_comp), dtype=np.float64)
    counts = np.zeros(len(sentence_indices), dtype=np.float64)
    row_of = {int(s): i for i, s in enumerate(sentence_indices)}
    for vector, sent_id in zip(word_rff, wsi[mask].astype(np.int64)):
        row = row_of[int(sent_id)]
        out[row] += vector
        counts[row] += 1
    if (counts == 0).any():
        raise RuntimeError("At least one sentence has no SFL RFF words.")
    return out / counts[:, None] if pooling == "mean" else out

def select_sfl_rff_gamma(word_sfl, wsi, y, train_sent, group_labels, n_components, pooling):
    losses = {g: [] for g in SFL_RFF_GAMMAS}
    inner_groups = group_labels[train_sent]
    for val_pos in grouped_folds(inner_groups, INNER_SPLITS):
        inner_val = train_sent[val_pos]
        inner_train = np.setdiff1d(train_sent, inner_val, assume_unique=True)
        ys = StandardScaler()
        y_tr = ys.fit_transform(y[inner_train, None]).ravel()
        y_va = ys.transform(y[inner_val, None]).ravel()
        for g in SFL_RFF_GAMMAS:
            imp, sca, rff = fit_sfl_rff(word_sfl, wsi, inner_train, g, n_components)
            x_tr = aggregate_sfl_rff(word_sfl, wsi, inner_train, imp, sca, rff, pooling)
            x_va = aggregate_sfl_rff(word_sfl, wsi, inner_val, imp, sca, rff, pooling)
            reg = fit_ridge(x_tr, y_tr)
            losses[g].append(float(np.mean((reg.predict(x_va) - y_va) ** 2)))
    return float(min(losses, key=lambda g: (np.mean(losses[g]), g)))

def score_sfl_rff_and_matched(word_sfl, wsi, y, group_labels, emb_sum, n_components,
                              name_rff=None, name_all=None):
    """Per outer fold: pick gamma (grouped inner CV), fit RFF on TRAIN words, SUM per
    sentence; score sfl_rff and all_matched = concat(emb_sum, sfl_rff). Same outer
    folds as score_matrix -> paired per-fold r."""
    rs_rff, rs_all = [], []
    for test in grouped_folds(group_labels, N_SPLITS):
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        gamma = select_sfl_rff_gamma(word_sfl, wsi, y, train, group_labels, n_components, "sum")
        imp, sca, rff = fit_sfl_rff(word_sfl, wsi, train, gamma, n_components)
        x_tr = aggregate_sfl_rff(word_sfl, wsi, train, imp, sca, rff, "sum")
        x_te = aggregate_sfl_rff(word_sfl, wsi, test, imp, sca, rff, "sum")
        reg = fit_ridge(x_tr, y[train])
        rs_rff.append(pearsonr(y[test], reg.predict(x_te))[0])
        xa_tr = np.hstack([emb_sum[train], x_tr])
        xa_te = np.hstack([emb_sum[test], x_te])
        reg2 = fit_ridge(xa_tr, y[train])
        rs_all.append(pearsonr(y[test], reg2.predict(xa_te))[0])
    if name_rff is not None:
        with open(f"results/{name_rff}_predictions", "wb") as h:
            pickle.dump(rs_rff, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"results/{name_all}_predictions", "wb") as h:
            pickle.dump(rs_all, h, protocol=pickle.HIGHEST_PROTOCOL)
    return ((float(np.mean(rs_rff)), float(np.std(rs_rff))),
            (float(np.mean(rs_all)), float(np.std(rs_all))))

# ----------------------------- main pipeline --------------------------------
def main():
    chdir(str(PROJECT_ROOT / "pereira_1"))
    import neurox.data.loader as data_loader

    meta = pd.read_csv("pereira_averaged_with_passage.csv")
    response = meta["EffectSize"].to_numpy(dtype=np.float64)
    groups = meta["passage_group"].to_numpy()
    n = len(response)
    assert n == 627, f"expected 627 sentences, got {n}"
    assert meta["passage_group"].notna().all(), "every sentence needs a passage_group"

    df = pd.read_csv("SFL.csv")                  # per-word: s_*, len, freq, sentnum
    wsi = df["sentnum"].to_numpy(dtype=np.int64)

    cols = {c: [] for c in PREDS}
    cols.update({c + "_sd": [] for c in PREDS})
    for key, scol, layer, dim in MODELS:
        print(f"\n=== {key} (D={dim}) ===", flush=True)
        act = data_loader.load_activations(ACT_FILE[key], dim, is_brnn=False)[0]
        emb_mean = get_act(act, layer, dim, "mean")
        emb_sum = get_act(act, layer, dim, "sum")
        del act
        base = pool_sfl(df, scol, n)
        allcat = np.hstack([base, emb_mean])
        word_sfl = df[[scol, "len", "freq"]].to_numpy(dtype=np.float64)

        m = {"base": score_matrix(base, response, groups, f"pereira_1_{key}_base"),
             "act": score_matrix(emb_mean, response, groups, f"pereira_1_{key}"),
             "all": score_matrix(allcat, response, groups, f"pereira_1_{key}_all"),
             "emb_sum": score_matrix(emb_sum, response, groups, f"pereira_1_{key}_embsum")}
        m["sfl_rff"], m["all_matched"] = score_sfl_rff_and_matched(
            word_sfl, wsi, response, groups, emb_sum, dim,
            f"pereira_1_{key}_sflrff", f"pereira_1_{key}_allmatched")
        for c in PREDS:
            cols[c].append(m[c][0]); cols[c + "_sd"].append(m[c][1])
        print({c: round(m[c][0], 4) for c in PREDS}, flush=True)

    df_results = pd.DataFrame({c: cols[c] for c in
                               ["base", "base_sd", "act", "act_sd", "all", "all_sd",
                                "emb_sum", "emb_sum_sd", "sfl_rff", "sfl_rff_sd",
                                "all_matched", "all_matched_sd"]}, index=ROW_INDEX)

    # significance: directional permutation test (a > b), all pairs + legacy aliases
    def pname(key, pred):
        return f"pereira_1_{key}{PRED_SUFFIX[pred]}"
    for a, b in itertools.combinations(PREDS, 2):
        df_results[f"p__{a}__{b}"] = [significance_testing(pname(k, a), pname(k, b))
                                      for k, *_ in MODELS]
    for col, (a, b) in {"p_base_emb": ("act", "base"), "p_all_base": ("all", "base"),
                        "p_all_emb": ("all", "act")}.items():
        df_results[col] = [significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]

    df_results.to_csv("results/results.csv", index=False)
    print("\nWROTE results/results.csv")
    print(df_results[PREDS].round(4))

if __name__ == "__main__":
    main()
