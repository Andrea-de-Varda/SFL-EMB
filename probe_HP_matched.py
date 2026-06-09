#!/usr/bin/env python3
"""
Wehbe2014 (Harry Potter) fMRI SFL-vs-EMB with the dimensionality-matched SFL
predictor. TR-level (2 s), 10-fold SEQUENTIAL KFold over the 1211 TR-rows
(unchanged from fMRI_analysis_HP_averaged.py). Per-word features are pooled to TRs
and concatenated with 4 lagged TRs, so the embedding regression input is 4*D.

Matched predictors (final dim after lagging):
  base        SFL(3) -> mean-per-TR -> 4 lags                  (12)
  act         EMB(D) -> mean-per-TR -> 4 lags                  (4*D)
  all         act ^ base                                       (4*D+12)
  emb_sum     EMB(D) -> SUM-per-TR  -> 4 lags                  (4*D)
  sfl_rff     SFL(3) -> RBFSampler(D) -> SUM-per-TR -> 4 lags  (4*D)
  all_matched emb_sum ^ sfl_rff                                (8*D)

Leakage control: per fold the RBFSampler/scaler are fit ONLY on words whose TR-row
is a TRAINING row (word_row map below); the RFF gamma is chosen by an inner KFold
over training rows. The per-TR pooling + lagging is a fixed transform of the
stimulus (exactly as for the embeddings), so cross-boundary lag mixing introduces
no response leakage.

Reuses the precomputed HP embeddings (HP_emb/activations_*.json via neurox) and the
per-word surprisal pickles (SFL/{gpt,small,med,large,xl}); freq/length rebuilt from
sentences.txt + subtlex. Response is the language-network-average ("all" ROI).
Overwrites fMRI/results/all_results.csv (base/act/all + p_* kept for plotting;
matched columns + all-pairs permutation p's added).
"""
import sys
import re
import pickle
import itertools
from os import chdir
from pathlib import Path as _Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())

SKIP_WORDS, END_WORDS = 20, 5176
MODELS = [("gpt", "gpt", 12, 768), ("small", "small", 12, 768), ("med", "med", 24, 1024),
          ("large", "large", 36, 1280), ("xl", "xl", 48, 1600)]
ACT_FILE = {"gpt": "HP_emb/activations_gpt.json", "small": "HP_emb/activations_gpt2_small.json",
            "med": "HP_emb/activations_gpt2_medium.json", "large": "HP_emb/activations_gpt2_large.json",
            "xl": "HP_emb/activations_gpt2_xl.json"}
SFL_PICKLE = {"gpt": "SFL/gpt", "small": "SFL/small", "med": "SFL/med",
              "large": "SFL/large", "xl": "SFL/xl"}
ROW_INDEX = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
PREDS = ["base", "act", "all", "emb_sum", "sfl_rff", "all_matched"]
SUF = {"base": "_base", "act": "", "all": "_all",
       "emb_sum": "_embsum", "sfl_rff": "_sflrff", "all_matched": "_allmatched"}

def get_layer(vec, layer, dim):
    return list(vec[layer * dim:(layer + 1) * dim])

def get_act(act_list, layer, dim):
    out = []
    for act_sent in act_list:
        for v in act_sent:
            out.append(get_layer(v, layer, dim))
    return np.array(out)

def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    import rff_matched as R
    chdir(str(PROJECT_ROOT / "fMRI"))
    import neurox.data.loader as data_loader

    words = np.load("fMRI/words_fmri.npy")
    time = np.load("fMRI/time_fmri.npy")                       # (1351,)
    runs = np.load("fMRI/runs_fmri.npy")                       # (1351,)
    time_words = np.load("fMRI/time_words_fmri.npy")[SKIP_WORDS:END_WORDS]  # (5156,)
    words_id = np.array([np.where(tw > time)[0][-1] for tw in time_words])  # (5156,) TR per word
    n_tr = time.shape[0]
    n_words = END_WORDS                                        # 5176

    # --- response: language-network ("all") average over voxels then subjects ---
    parts = [np.load(f"fMRI/data_subject_{p}.npy") for p in "FHIJKLMN"]
    roi = np.load("fMRI/HP_subj_roi_inds.npy", allow_pickle=True).tolist()
    y = np.mean([parts[k][:, roi[p]["all"]].mean(axis=1) for k, p in enumerate("FHIJKLMN")], axis=0)
    n_rows = len(y)                                            # 1211

    # --- TR(1351) -> final-row(1211) map (per-run trim [20:-15]); then word -> row ---
    tr2row = np.full(n_tr, -1, dtype=np.int64)
    r = 0
    for run in range(1, 5):
        run_tr = np.where(runs == run)[0]                     # TR indices of this run, in order
        for t in run_tr[20:-15]:
            tr2row[t] = r; r += 1
    assert r == n_rows, f"row count {r} != {n_rows}"
    word_row = np.full(n_words, -1, dtype=np.int64)           # word -> final row (or -1)
    for j in range(len(words_id)):                            # j indexes words[SKIP_WORDS:]
        word_row[SKIP_WORDS + j] = tr2row[int(words_id[j])]

    def embed_words(features, pooling):
        """per-word (n_words,k) -> mean|sum per TR -> 4 lags -> per-run trim -> impute -> (1211, 4k)."""
        f = features[SKIP_WORDS:]
        k = features.shape[1]
        tr = np.full((n_tr, k), np.nan, dtype=np.float64)
        for i in range(n_tr):
            sel = f[words_id.astype(int) == i]
            if len(sel):
                tr[i] = sel.mean(axis=0) if pooling == "mean" else sel.sum(axis=0)
        out = np.concatenate([R._delay_one(tr, dl) for dl in (1, 2, 3, 4)], axis=-1)
        out = np.vstack([out[runs == run][20:-15] for run in range(1, 5)])
        return R._imputate_na(out)

    # --- per-word SFL: freq/length from sentences.txt + subtlex; surprisal from pickles ---
    freq_tab = pd.read_excel(PROJECT_ROOT / "main_data/subtlex.xlsx")
    fdict = {row.Word: np.log(row.FREQcount) for _, row in freq_tab.iterrows()}
    minf = min(fdict.values())
    def get_f(w):
        w = re.sub('[\\.\\,\\:\\-\\?\\!\\)\\(\\"]', "", w)
        return fdict.get(w, minf)
    sent = [re.sub("\n", "", s).split() for s in open("sentences.txt").readlines()]
    freq = np.array([get_f(w) for s in sent for w in s])
    length = np.array([len(w) for s in sent for w in s])

    folds = list(R.make_folds(n_rows, None, 10))

    def kfold_fixed(X, saveto):
        rs = []
        for test in folds:
            train = np.setdiff1d(np.arange(n_rows), test, assume_unique=True)
            rs.append(pearsonr(y[test], R.fit_ridge(X[train], y[train]).predict(X[test]))[0])
        return R.save_predictions(rs, saveto)

    def rff_word_to_rows(sfl, train_words, gamma, D, pooling="sum"):
        imp, sca, rff = R.fit_sfl_rff(sfl, np.arange(n_words), train_words, gamma, D)
        word_rff = rff.transform(sca.transform(imp.transform(sfl)))   # (n_words, D)
        return embed_words(word_rff, pooling)

    def select_gamma(sfl, train_rows, D):
        losses = {g: [] for g in R.SFL_RFF_GAMMAS}
        for vpos in R.make_folds(len(train_rows), None, 5):
            inner_val = train_rows[vpos]
            inner_train = np.setdiff1d(train_rows, inner_val, assume_unique=True)
            tw = np.where(np.isin(word_row, inner_train))[0]
            for g in R.SFL_RFF_GAMMAS:
                X = rff_word_to_rows(sfl, tw, g, D)
                losses[g].append(float(np.mean((R.fit_ridge(X[inner_train], y[inner_train]).predict(X[inner_val]) - y[inner_val]) ** 2)))
        return float(min(losses, key=lambda g: (np.mean(losses[g]), g)))

    cols = {c: [] for c in PREDS}
    cols.update({c + "_sd": [] for c in PREDS})
    for key, sflkey, layer, D in MODELS:
        print(f"\n=== {key} (D={D}, 4*D={4*D}) ===", flush=True)
        act = data_loader.load_activations(ACT_FILE[key], D, is_brnn=False)[0]
        word_emb = get_act(act, layer, D); del act
        assert len(word_emb) == n_words, f"emb words {len(word_emb)} != {n_words}"
        with open(SFL_PICKLE[key], "rb") as h:
            s = pickle.load(h)
        sfl = np.stack([np.asarray(s, dtype=np.float64), freq, length], axis=1)   # (n_words, 3)

        act_X = embed_words(word_emb, "mean")
        emb_sum_X = embed_words(word_emb, "sum")
        base_X = embed_words(sfl, "mean")
        all_X = np.hstack([act_X, base_X])

        m = {"base": kfold_fixed(base_X, f"HP_{key}_base"),
             "act": kfold_fixed(act_X, f"HP_{key}"),
             "all": kfold_fixed(all_X, f"HP_{key}_all"),
             "emb_sum": kfold_fixed(emb_sum_X, f"HP_{key}_embsum")}

        rs_rff, rs_all = [], []
        for test in folds:
            train = np.setdiff1d(np.arange(n_rows), test, assume_unique=True)
            gamma = select_gamma(sfl, train, D)
            tw = np.where(np.isin(word_row, train))[0]
            rffX = rff_word_to_rows(sfl, tw, gamma, D)
            rs_rff.append(pearsonr(y[test], R.fit_ridge(rffX[train], y[train]).predict(rffX[test]))[0])
            amX = np.hstack([emb_sum_X, rffX])
            rs_all.append(pearsonr(y[test], R.fit_ridge(amX[train], y[train]).predict(amX[test]))[0])
        m["sfl_rff"] = R.save_predictions(rs_rff, f"HP_{key}_sflrff")
        m["all_matched"] = R.save_predictions(rs_all, f"HP_{key}_allmatched")

        for c in PREDS:
            cols[c].append(m[c][0]); cols[c + "_sd"].append(m[c][1])
        print({c: round(m[c][0], 4) for c in PREDS}, flush=True)

    df_results = pd.DataFrame({c: cols[c] for c in
                               ["base", "base_sd", "act", "act_sd", "all", "all_sd",
                                "emb_sum", "emb_sum_sd", "sfl_rff", "sfl_rff_sd",
                                "all_matched", "all_matched_sd"]}, index=ROW_INDEX)

    def pname(key, pred):
        return f"HP_{key}{SUF[pred]}"
    for a, b in itertools.combinations(PREDS, 2):
        df_results[f"p__{a}__{b}"] = [R.significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]
    for col, (a, b) in {"p_base_emb": ("act", "base"), "p_all_base": ("all", "base"),
                        "p_all_emb": ("all", "act")}.items():
        df_results[col] = [R.significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]

    df_results.to_csv("results/all_results.csv", index=False)
    print("\nWROTE fMRI/results/all_results.csv")
    print(df_results[PREDS].round(4))

if __name__ == "__main__":
    main()
