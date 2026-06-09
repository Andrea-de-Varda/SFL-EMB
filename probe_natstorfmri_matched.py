#!/usr/bin/env python3
"""
NatStor-fMRI SFL-vs-EMB with the dimensionality-matched SFL predictor.
TR-level (2 s), LEAVE-ONE-STORY-OUT CV (9 stories) -- unchanged from
fit_encoding_natstor.py. Per-word features are pooled to TRs and concatenated with
4 lagged TRs (delay_mat), so the embedding regression input is 4*D.

Matched predictors (D = per-TR embedding dim; final dim 4*D after lagging):
  base        per-word SFL(3) -> mean-per-TR -> 4 lags                 (12)
  act         per-word EMB(D) -> mean-per-TR -> 4 lags                 (4*D)
  all         act ^ base                                              (4*D+12)
  emb_sum     per-word EMB(D) -> SUM-per-TR  -> 4 lags                 (4*D)
  sfl_rff     per-word SFL(3) -> RBFSampler(D) -> SUM-per-TR -> 4 lags (4*D)
  all_matched emb_sum ^ sfl_rff                                        (8*D)

Leakage control: the RBFSampler/scaler are fit ONLY on the words of the training
stories each fold (held-out story's words never touch the transform); the RFF gamma
is chosen by an inner leave-one-training-story-out CV. Lagging is within-story
(preproc_align operates per story), so no cross-story mixing.

Reuses the precomputed per-word embeddings (embeddings/{prefix}_{lang}) and per-word
SFL (embeddings/sfl_{prefix}_{lang}); word->TR timing from transcribed/{lang}.csv.
Overwrites natstor_fmri/results/all_results.csv (base/act/all + p_* kept for
plotting; matched columns + all-pairs permutation p's added).
"""
import sys
import itertools
import pickle
from os import chdir
from pathlib import Path as _Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())

STORIES = ["1", "2", "3", "4", "5", "6", "7", "9", "10"]
STORY_NAME = {"1": "boar", "2": "aqua", "3": "matchstickseller", "4": "kingofbirds",
              "5": "elvis", "6": "mrsticky", "7": "highschool", "10": "tree", "9": "tulips"}
# (model key, file prefix, dim D)
MODELS = [("gpt", "gpt", 768), ("small", "gpt2_small", 768), ("med", "gpt2_med", 1024),
          ("large", "gpt2_large", 1280), ("xl", "gpt2_xl", 1600)]
ROW_INDEX = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
PREDS = ["base", "act", "all", "emb_sum", "sfl_rff", "all_matched"]
SUF = {"base": "_base", "act": "", "all": "_all",
       "emb_sum": "_embsum", "sfl_rff": "_sflrff", "all_matched": "_allmatched"}

def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    import rff_matched as R
    chdir(str(PROJECT_ROOT / "natstor_fmri"))

    def load(name):
        with open("embeddings/" + name, "rb") as h:
            return pickle.load(h)

    with open("response/d_natstor", "rb") as h:
        d = pickle.load(h)
    y_by = [np.asarray(d[STORY_NAME[s]], dtype=np.float64) for s in STORIES]
    n_tr = [len(v) for v in y_by]

    def words_id_for(lang, n):
        """word -> TR index, replicating preproc_align (drop first word, end-times)."""
        df = pd.read_csv("transcribed/" + lang + ".csv").drop(index=0).reset_index(drop=True)
        time = np.arange(0, n * 2, 2)
        end = df["end"].to_numpy()
        wid = np.array([np.where(end[i] > time)[0][-1] for i in range(len(end))])
        return wid
    WID = [words_id_for(s, n_tr[i]) for i, s in enumerate(STORIES)]

    S = len(STORIES)
    folds = list(range(S))  # leave-one-story-out

    def loso_fixed(feat_by, saveto):
        rs = []
        for i in folds:
            tr = [j for j in folds if j != i]
            Xtr = np.concatenate([feat_by[j] for j in tr]); ytr = np.concatenate([y_by[j] for j in tr])
            rs.append(pearsonr(y_by[i], R.fit_ridge(Xtr, ytr).predict(feat_by[i]))[0])
        return R.save_predictions(rs, saveto)

    def build_rff_story(word_sfl_by, imp, sca, rff, j):
        wr = rff.transform(sca.transform(imp.transform(word_sfl_by[j])))
        return R.aggregate_to_tr(wr, WID[j], n_tr[j], "sum")

    def select_gamma(word_sfl_by, train):
        losses = {g: [] for g in R.SFL_RFF_GAMMAS}
        for v in train:                                   # inner leave-one-train-story-out
            inner = [j for j in train if j != v]
            tw = np.concatenate([word_sfl_by[j] for j in inner])
            for g in R.SFL_RFF_GAMMAS:
                imp, sca, rff = R.fit_sfl_rff(tw, np.arange(len(tw)), np.arange(len(tw)), g, D)
                Xtr = np.concatenate([build_rff_story(word_sfl_by, imp, sca, rff, j) for j in inner])
                Xv = build_rff_story(word_sfl_by, imp, sca, rff, v)
                ytr = np.concatenate([y_by[j] for j in inner])
                losses[g].append(float(np.mean((R.fit_ridge(Xtr, ytr).predict(Xv) - y_by[v]) ** 2)))
        return float(min(losses, key=lambda g: (np.mean(losses[g]), g)))

    cols = {c: [] for c in PREDS}
    cols.update({c + "_sd": [] for c in PREDS})
    for key, prefix, D in MODELS:
        print(f"\n=== {key} (D={D}, 4*D={4*D}) ===", flush=True)
        n_layer = max(load(f"{prefix}_{STORIES[0]}").keys())
        word_emb = [load(f"{prefix}_{s}")[n_layer][1:] for s in STORIES]
        word_sfl = [np.asarray(load(f"sfl_{prefix}_{s}"), dtype=np.float64) for s in STORIES]
        emb_mean = [R.aggregate_to_tr(word_emb[j], WID[j], n_tr[j], "mean") for j in folds]
        emb_sum = [R.aggregate_to_tr(word_emb[j], WID[j], n_tr[j], "sum") for j in folds]
        base = [R.aggregate_to_tr(word_sfl[j], WID[j], n_tr[j], "mean") for j in folds]
        allc = [np.hstack([emb_mean[j], base[j]]) for j in folds]

        m = {"base": loso_fixed(base, f"natstorfmri_{key}_base"),
             "act": loso_fixed(emb_mean, f"natstorfmri_{key}"),
             "all": loso_fixed(allc, f"natstorfmri_{key}_all"),
             "emb_sum": loso_fixed(emb_sum, f"natstorfmri_{key}_embsum")}

        rs_rff, rs_all = [], []
        for i in folds:
            train = [j for j in folds if j != i]
            gamma = select_gamma(word_sfl, train)
            tw = np.concatenate([word_sfl[j] for j in train])
            imp, sca, rff = R.fit_sfl_rff(tw, np.arange(len(tw)), np.arange(len(tw)), gamma, D)
            rffX = [build_rff_story(word_sfl, imp, sca, rff, j) for j in folds]
            Xtr = np.concatenate([rffX[j] for j in train]); ytr = np.concatenate([y_by[j] for j in train])
            rs_rff.append(pearsonr(y_by[i], R.fit_ridge(Xtr, ytr).predict(rffX[i]))[0])
            am = [np.hstack([emb_sum[j], rffX[j]]) for j in folds]
            Xtr2 = np.concatenate([am[j] for j in train])
            rs_all.append(pearsonr(y_by[i], R.fit_ridge(Xtr2, ytr).predict(am[i]))[0])
        m["sfl_rff"] = R.save_predictions(rs_rff, f"natstorfmri_{key}_sflrff")
        m["all_matched"] = R.save_predictions(rs_all, f"natstorfmri_{key}_allmatched")

        for c in PREDS:
            cols[c].append(m[c][0]); cols[c + "_sd"].append(m[c][1])
        print({c: round(m[c][0], 4) for c in PREDS}, flush=True)

    df_results = pd.DataFrame({c: cols[c] for c in
                               ["base", "base_sd", "act", "act_sd", "all", "all_sd",
                                "emb_sum", "emb_sum_sd", "sfl_rff", "sfl_rff_sd",
                                "all_matched", "all_matched_sd"]}, index=ROW_INDEX)

    def pname(key, pred):
        return f"natstorfmri_{key}{SUF[pred]}"
    for a, b in itertools.combinations(PREDS, 2):
        df_results[f"p__{a}__{b}"] = [R.significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]
    for col, (a, b) in {"p_base_emb": ("act", "base"), "p_all_base": ("all", "base"),
                        "p_all_emb": ("all", "act")}.items():
        df_results[col] = [R.significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]

    df_results.to_csv("results/all_results.csv", index=False)
    print("\nWROTE natstor_fmri/results/all_results.csv")
    print(df_results[PREDS].round(4))

if __name__ == "__main__":
    main()
