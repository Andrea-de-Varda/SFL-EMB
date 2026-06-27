#!/usr/bin/env python3
"""
SYMMETRIC RFF comparison for the TR-level brain datasets (NatStor-fMRI, Wehbe2014/HP).
Routes BOTH EMB and SFL through the identical pipeline
  per-word -> standardize(train) -> RBFSampler(D, gamma) -> SUM-per-TR -> 4 lags -> ridge
with gamma nested-CV-tuned independently for each side. (Same SUM-per-TR pooling as the
current matched scripts -- mean-pooling deferred per request.) Predictors per model:
  base  SFL(3) mean-per-TR+lags     act  EMB(D) mean-per-TR+lags     (the originals)
  sfl_rff  SFL->RFF(D)->sum-per-TR+lags     emb_rff  EMB->RFF(D)->sum-per-TR+lags
  all_rff  concat(emb_rff, sfl_rff)
RFF fit per fold on TRAIN-fold words only (NatStor: 8 training stories; HP: training-row
words via the word->row map). Results -> results_rffsym/ (TEMP).

Usage (neurox env):  python probe_rffsym_tr.py {natstorfmri|HP}
HP needs openpyxl (pip install openpyxl) for subtlex.xlsx.
"""
import sys, re, pickle, itertools
from os import chdir, makedirs
from pathlib import Path as _Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
OUTDIR = str(PROJECT_ROOT / "results_rffsym")
ROW_INDEX = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
PREDS = ["base", "act", "sfl_rff", "emb_rff", "all_rff"]
SUF = {"base": "_base", "act": "_act", "sfl_rff": "_sflrff", "emb_rff": "_embrff", "all_rff": "_allrff"}
KEYS = ["gpt", "small", "med", "large", "xl"]

def get_layer(vec, layer, dim):
    return list(vec[layer * dim:(layer + 1) * dim])

# ---------------------------------------------------------------- NatStor (LOSO)
def run_natstor(R):
    chdir(str(PROJECT_ROOT / "natstor_fmri"))
    STORIES = ["1", "2", "3", "4", "5", "6", "7", "9", "10"]
    NAME = {"1": "boar", "2": "aqua", "3": "matchstickseller", "4": "kingofbirds", "5": "elvis",
            "6": "mrsticky", "7": "highschool", "10": "tree", "9": "tulips"}
    MODELS = [("gpt", "gpt", 768), ("small", "gpt2_small", 768), ("med", "gpt2_med", 1024),
              ("large", "gpt2_large", 1280), ("xl", "gpt2_xl", 1600)]
    def load(nm):
        with open("embeddings/" + nm, "rb") as h:
            return pickle.load(h)
    with open("response/d_natstor", "rb") as h:
        d = pickle.load(h)
    y_by = [np.asarray(d[NAME[s]], float) for s in STORIES]
    n_tr = [len(v) for v in y_by]
    def wid_for(lang, n):
        df = pd.read_csv("transcribed/" + lang + ".csv").drop(index=0).reset_index(drop=True)
        time = np.arange(0, n * 2, 2); end = df["end"].to_numpy()
        return np.array([np.where(end[i] > time)[0][-1] for i in range(len(end))])
    WID = [wid_for(s, n_tr[i]) for i, s in enumerate(STORIES)]
    folds = list(range(len(STORIES)))

    def story_rff(feats_by, imp, sca, rff, j):
        wr = rff.transform(sca.transform(imp.transform(feats_by[j])))
        return R.aggregate_to_tr(wr, WID[j], n_tr[j], "sum")
    def fit_on(stories_words):
        tw = np.concatenate(stories_words)
        return tw
    def select_gamma(feats_by, train, D):
        loss = {g: [] for g in R.SFL_RFF_GAMMAS}
        for v in train:
            inner = [j for j in train if j != v]
            tw = np.concatenate([feats_by[j] for j in inner])
            for g in R.SFL_RFF_GAMMAS:
                imp, sca, rff = R.fit_sfl_rff(tw, np.arange(len(tw)), np.arange(len(tw)), g, D)
                Xtr = np.concatenate([story_rff(feats_by, imp, sca, rff, j) for j in inner])
                Xv = story_rff(feats_by, imp, sca, rff, v)
                ytr = np.concatenate([y_by[j] for j in inner])
                loss[g].append(float(np.mean((R.fit_ridge(Xtr, ytr).predict(Xv) - y_by[v]) ** 2)))
        return float(min(loss, key=lambda g: (np.mean(loss[g]), g)))
    def loso_fixed(feat_by, saveto):
        rs = []
        for i in folds:
            tr = [j for j in folds if j != i]
            Xtr = np.concatenate([feat_by[j] for j in tr]); ytr = np.concatenate([y_by[j] for j in tr])
            rs.append(pearsonr(y_by[i], R.fit_ridge(Xtr, ytr).predict(feat_by[i]))[0])
        return R.save_predictions(rs, saveto)

    cols = {c: [] for c in PREDS}; cols.update({c + "_sd": [] for c in PREDS})
    for key, prefix, D in MODELS:
        print(f"\n=== natstorfmri {key} (D={D}) ===", flush=True)
        nlayer = max(load(f"{prefix}_{STORIES[0]}").keys())
        word_emb = [load(f"{prefix}_{s}")[nlayer][1:] for s in STORIES]
        word_sfl = [np.asarray(load(f"sfl_{prefix}_{s}"), float) for s in STORIES]
        emb_mean = [R.aggregate_to_tr(word_emb[j], WID[j], n_tr[j], "mean") for j in folds]
        base = [R.aggregate_to_tr(word_sfl[j], WID[j], n_tr[j], "mean") for j in folds]
        m = {"base": loso_fixed(base, f"natstorfmri_{key}_base"),
             "act": loso_fixed(emb_mean, f"natstorfmri_{key}_act")}
        rs_e, rs_s, rs_a = [], [], []
        for i in folds:
            train = [j for j in folds if j != i]
            ge = select_gamma(word_emb, train, D); gs = select_gamma(word_sfl, train, D)
            twe = fit_on([word_emb[j] for j in train]); tws = fit_on([word_sfl[j] for j in train])
            ie, se, re = R.fit_sfl_rff(twe, np.arange(len(twe)), np.arange(len(twe)), ge, D)
            isf, ssf, rsf = R.fit_sfl_rff(tws, np.arange(len(tws)), np.arange(len(tws)), gs, D)
            Xe = [story_rff(word_emb, ie, se, re, j) for j in folds]
            Xs = [story_rff(word_sfl, isf, ssf, rsf, j) for j in folds]
            ytr = np.concatenate([y_by[j] for j in train])
            rs_e.append(pearsonr(y_by[i], R.fit_ridge(np.concatenate([Xe[j] for j in train]), ytr).predict(Xe[i]))[0])
            rs_s.append(pearsonr(y_by[i], R.fit_ridge(np.concatenate([Xs[j] for j in train]), ytr).predict(Xs[i]))[0])
            Xa = [np.hstack([Xe[j], Xs[j]]) for j in folds]
            rs_a.append(pearsonr(y_by[i], R.fit_ridge(np.concatenate([Xa[j] for j in train]), ytr).predict(Xa[i]))[0])
        m["emb_rff"] = R.save_predictions(rs_e, f"natstorfmri_{key}_embrff")
        m["sfl_rff"] = R.save_predictions(rs_s, f"natstorfmri_{key}_sflrff")
        m["all_rff"] = R.save_predictions(rs_a, f"natstorfmri_{key}_allrff")
        for c in PREDS:
            cols[c].append(m[c][0]); cols[c + "_sd"].append(m[c][1])
        print({c: round(m[c][0], 4) for c in PREDS}, flush=True)
    return "natstorfmri", cols

# --------------------------------------------------------------------- HP (KFold)
def run_hp(R):
    chdir(str(PROJECT_ROOT / "fMRI"))
    import neurox.data.loader as dl
    SKIP, END = 20, 5176
    MODELS = [("gpt", "gpt", 12, 768), ("small", "small", 12, 768), ("med", "med", 24, 1024),
              ("large", "large", 36, 1280), ("xl", "xl", 48, 1600)]
    ACT = {"gpt": "HP_emb/activations_gpt.json", "small": "HP_emb/activations_gpt2_small.json",
           "med": "HP_emb/activations_gpt2_medium.json", "large": "HP_emb/activations_gpt2_large.json",
           "xl": "HP_emb/activations_gpt2_xl.json"}
    SFLP = {"gpt": "SFL/gpt", "small": "SFL/small", "med": "SFL/med", "large": "SFL/large", "xl": "SFL/xl"}
    time = np.load("fMRI/time_fmri.npy"); runs = np.load("fMRI/runs_fmri.npy")
    tw_ = np.load("fMRI/time_words_fmri.npy")[SKIP:END]
    words_id = np.array([np.where(t > time)[0][-1] for t in tw_])
    n_tr = time.shape[0]; n_words = END
    parts = [np.load(f"fMRI/data_subject_{p}.npy") for p in "FHIJKLMN"]
    roi = np.load("fMRI/HP_subj_roi_inds.npy", allow_pickle=True).tolist()
    y = np.mean([parts[k][:, roi[p]["all"]].mean(axis=1) for k, p in enumerate("FHIJKLMN")], axis=0)
    n_rows = len(y)
    tr2row = np.full(n_tr, -1, np.int64); r = 0
    for run in range(1, 5):
        for t in np.where(runs == run)[0][20:-15]:
            tr2row[t] = r; r += 1
    word_row = np.full(n_words, -1, np.int64)
    for j in range(len(words_id)):
        word_row[SKIP + j] = tr2row[int(words_id[j])]

    def embed(features, pooling):
        f = features[SKIP:]; k = features.shape[1]
        tr = np.full((n_tr, k), np.nan)
        for i in range(n_tr):
            sel = f[words_id.astype(int) == i]
            if len(sel):
                tr[i] = sel.mean(0) if pooling == "mean" else sel.sum(0)
        out = np.concatenate([R._delay_one(tr, dl_) for dl_ in (1, 2, 3, 4)], axis=-1)
        out = np.vstack([out[runs == run][20:-15] for run in range(1, 5)])
        return R._imputate_na(out)

    freq_tab = pd.read_excel(PROJECT_ROOT / "main_data/subtlex.xlsx")
    fdict = {row.Word: np.log(row.FREQcount) for _, row in freq_tab.iterrows()}; minf = min(fdict.values())
    def getf(w):
        w = re.sub('[\\.\\,\\:\\-\\?\\!\\)\\(\\"]', "", w); return fdict.get(w, minf)
    sent = [re.sub("\n", "", s).split() for s in open("sentences.txt").readlines()]
    freq = np.array([getf(w) for s in sent for w in s]); length = np.array([len(w) for s in sent for w in s])
    folds = list(R.make_folds(n_rows, None, 10))

    def feat_rows(feats, train_words, gamma, D):
        imp, sca, rff = R.fit_sfl_rff(feats, np.arange(n_words), train_words, gamma, D)
        return embed(rff.transform(sca.transform(imp.transform(feats))), "sum")
    def select_gamma(feats, train_rows, D):
        loss = {g: [] for g in R.SFL_RFF_GAMMAS}
        for vpos in R.make_folds(len(train_rows), None, 5):
            iv = train_rows[vpos]; it = np.setdiff1d(train_rows, iv, assume_unique=True)
            tw = np.where(np.isin(word_row, it))[0]
            for g in R.SFL_RFF_GAMMAS:
                X = feat_rows(feats, tw, g, D)
                loss[g].append(float(np.mean((R.fit_ridge(X[it], y[it]).predict(X[iv]) - y[iv]) ** 2)))
        return float(min(loss, key=lambda g: (np.mean(loss[g]), g)))
    def kfold_fixed(X, saveto):
        rs = []
        for test in folds:
            tr = np.setdiff1d(np.arange(n_rows), test, assume_unique=True)
            rs.append(pearsonr(y[test], R.fit_ridge(X[tr], y[tr]).predict(X[test]))[0])
        return R.save_predictions(rs, saveto)

    cols = {c: [] for c in PREDS}; cols.update({c + "_sd": [] for c in PREDS})
    for key, sflkey, layer, D in MODELS:
        print(f"\n=== HP {key} (D={D}) ===", flush=True)
        act = dl.load_activations(ACT[key], D, is_brnn=False)[0]
        word_emb = np.array([get_layer(v, layer, D) for s_ in act for v in s_]); del act
        assert len(word_emb) == n_words
        with open(SFLP[key], "rb") as h:
            s = pickle.load(h)
        sfl = np.stack([np.asarray(s, float), freq, length], axis=1)
        m = {"base": kfold_fixed(embed(sfl, "mean"), f"HP_{key}_base"),
             "act": kfold_fixed(embed(word_emb, "mean"), f"HP_{key}_act")}
        rs_e, rs_s, rs_a = [], [], []
        for test in folds:
            train = np.setdiff1d(np.arange(n_rows), test, assume_unique=True)
            tw = np.where(np.isin(word_row, train))[0]
            ge = select_gamma(word_emb, train, D); gs = select_gamma(sfl, train, D)
            Xe = feat_rows(word_emb, tw, ge, D); Xs = feat_rows(sfl, tw, gs, D)
            rs_e.append(pearsonr(y[test], R.fit_ridge(Xe[train], y[train]).predict(Xe[test]))[0])
            rs_s.append(pearsonr(y[test], R.fit_ridge(Xs[train], y[train]).predict(Xs[test]))[0])
            Xa = np.hstack([Xe, Xs])
            rs_a.append(pearsonr(y[test], R.fit_ridge(Xa[train], y[train]).predict(Xa[test]))[0])
        m["emb_rff"] = R.save_predictions(rs_e, f"HP_{key}_embrff")
        m["sfl_rff"] = R.save_predictions(rs_s, f"HP_{key}_sflrff")
        m["all_rff"] = R.save_predictions(rs_a, f"HP_{key}_allrff")
        for c in PREDS:
            cols[c].append(m[c][0]); cols[c + "_sd"].append(m[c][1])
        print({c: round(m[c][0], 4) for c in PREDS}, flush=True)
    return "HP", cols

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "natstorfmri"
    sys.path.insert(0, str(PROJECT_ROOT))
    import rff_matched as R
    makedirs(OUTDIR, exist_ok=True); R.RESULTS_DIR = OUTDIR
    prefix, cols = run_natstor(R) if name == "natstorfmri" else run_hp(R)
    df = pd.DataFrame({c: cols[c] for c in
                       ["base", "base_sd", "act", "act_sd", "sfl_rff", "sfl_rff_sd",
                        "emb_rff", "emb_rff_sd", "all_rff", "all_rff_sd"]}, index=ROW_INDEX)
    for a, b in itertools.combinations(PREDS, 2):
        df[f"p__{a}__{b}"] = [R.significance_testing(f"{prefix}_{k}{SUF[a]}", f"{prefix}_{k}{SUF[b]}") for k in KEYS]
    df.to_csv(f"{OUTDIR}/{prefix}_rffsym.csv", index=False)
    print(f"\nWROTE {OUTDIR}/{prefix}_rffsym.csv")
    print(df[PREDS].round(4))

if __name__ == "__main__":
    main()
