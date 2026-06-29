#!/usr/bin/env python3
"""
Does the SFL->RFF expansion encode MEANING / high-level form features?

We predict 14 sentence-level features (7 "semantic": imageability, others'-thoughts,
physical, places, valence, arousal, sense; 7 "form/syntax": pcfg log-prob,
grammaticality, frequency, conversational, dependency length, avg word log-freq,
avg word length) from three predictors, using the SAME ridge pipeline as the brain
encoding (10-fold sequential KFold, RidgeCV nested alpha, Pearson r per fold):

  base     SFL (3): surprisal, length, frequency        (per-sentence, mean over words)
  act      EMB (D): LM last layer, mean-over-token pooled
  sfl_rff  SFL(3) -> RBFSampler(D), SUM-pooled per sentence  (the matched SFL)

Logic: if the nonlinear RFF expansion of SFL captures meaning, sfl_rff should predict
the SEMANTIC ratings far better than raw SFL (base) and approach EMB (act). If it only
recovers form/length structure, it should help the FORM features but not the semantic
ones. Per-fold r vectors are saved so paired permutation tests compare predictors.

The 1000 feature sentences are a subset of the 2000 Tuckute/control sentences, so we
reuse control_fmri activations + SFL.csv, aligning by sentence string (sentnum order
== groupby-sorted == activation order; verified: word counts match with a constant
+1 offset from the dropped sentence-initial word).

Usage (neurox env, cluster):  python predict_features.py
Writes check-features/results_features.csv (+ per-fold preds under check-features/preds/).
"""
import sys
from os import chdir, makedirs
from pathlib import Path as _Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
sys.path.insert(0, str(PROJECT_ROOT))
import rff_matched as R                       # noqa: E402

CHECK_DIR = PROJECT_ROOT / "check-features"
OUTDIR = str(CHECK_DIR / "preds")
# (model key, surprisal column in control SFL.csv, last-layer index, dim D)
MODELS = [("gpt", "s_GPT", 12, 768), ("small", "s_small", 12, 768),
          ("med", "s_med", 24, 1024), ("large", "s_large", 36, 1280),
          ("xl", "s_xl", 48, 1600)]
ACT_FILE = {"gpt": "activations_gpt.json", "small": "activations_gpt2_small.json",
            "med": "activations_gpt2_medium.json", "large": "activations_gpt2_large.json",
            "xl": "activations_gpt2_xl.json"}
SEM = ["rating_imageability_mean", "rating_others_thoughts_mean", "rating_physical_mean",
       "rating_places_mean", "rating_valence_mean", "rating_arousal_mean", "rating_sense_mean"]
SYN = ["log-prob-pcfg_mean", "rating_gram_mean", "rating_frequency_mean",
       "rating_conversational_mean", "dep_length_mean", "avg_word_freq_zipf", "avg_word_length"]
TARGETS = [(c, "semantic") for c in SEM] + [(c, "form") for c in SYN]


def score_sflrff_only(word_sfl, wsi, y, D, name, n_splits=10):
    """OOF sfl_rff scoring identical to rff_matched.score_sfl_rff_sentence's sfl_rff
    branch (nested-gamma selection, SUM-pool per sentence), saving per-fold r."""
    rs = []
    for test in R.make_folds(len(y), None, n_splits):
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        g = R._select_gamma_sentence(word_sfl, wsi, y, train, None, D)
        imp, sca, rff = R.fit_sfl_rff(word_sfl, wsi, train, g, D)
        x_tr = R.words_to_units(word_sfl, wsi, train, imp, sca, rff, "sum")
        x_te = R.words_to_units(word_sfl, wsi, test, imp, sca, rff, "sum")
        rs.append(pearsonr(y[test], R.fit_ridge(x_tr, y[train]).predict(x_te))[0])
    return R.save_predictions(rs, name)


def main():
    makedirs(OUTDIR, exist_ok=True)
    R.RESULTS_DIR = OUTDIR
    import neurox.data.loader as dl

    # --- feature targets ---
    sem = pd.read_csv(CHECK_DIR / "features_semantics.csv")
    syn = pd.read_csv(CHECK_DIR / "features_syntax.csv")
    feat = sem.merge(syn, on="sentence")
    assert len(feat) == 1000, len(feat)

    # --- align feature sentences to control sentnum / activation order ---
    chdir(str(PROJECT_ROOT / "control_fmri"))
    ctl = pd.read_csv("data/brain-lang-data_participant_20230728.csv")
    sents = (ctl[ctl["roi"] == "lang_LH_netw"].groupby("sentence")
             .agg({"response_target": "mean"}).index.to_numpy())          # sorted == sentnum order
    s2i = {s: i for i, s in enumerate(sents)}
    sel = np.array([s2i[s] for s in feat["sentence"]])                     # sentnum per feat row
    assert len(set(sel)) == 1000
    old2new = {int(sn): pos for pos, sn in enumerate(sel)}                 # sentnum -> dense 0..999

    sfl = pd.read_csv("SFL.csv")
    mask = sfl["sentnum"].isin(sel.tolist())
    wsi_sel = sfl.loc[mask, "sentnum"].map(old2new).to_numpy(int)         # word -> unit (0..999)

    rows = []
    for key, scol, layer, dim in MODELS:
        print(f"\n=== {key} (D={dim}) ===", flush=True)
        # SFL base (3) and word-level SFL for this model's surprisal column
        cols3 = [scol, "len", "freq"]
        base = (sfl.groupby("sentnum")[cols3].mean().reindex(sel).to_numpy(float))   # (1000,3)
        word_sfl = sfl.loc[mask, cols3].to_numpy(float)
        # EMB act (D), mean-pooled, then subset to the 1000
        act_list = dl.load_activations(ACT_FILE[key], dim, is_brnn=False)[0]
        act_full = R.get_act(act_list, layer, dim, "mean")               # (2000, D)
        act = np.asarray(act_full)[sel]                                   # (1000, D)
        del act_list, act_full

        for col, group in TARGETS:
            y = feat[col].to_numpy(float)
            nb = f"feat_{key}_{_clean(col)}_base"
            na = f"feat_{key}_{_clean(col)}_act"
            nr = f"feat_{key}_{_clean(col)}_sflrff"
            r_base = R.score_matrix(base, y, nb)
            r_act = R.score_matrix(act, y, na)
            r_rff = score_sflrff_only(word_sfl, wsi_sel, y, dim, nr)
            rows.append(dict(
                model=key, feature=col, group=group,
                r_base=r_base[0], sd_base=r_base[1],
                r_act=r_act[0], sd_act=r_act[1],
                r_sflrff=r_rff[0], sd_sflrff=r_rff[1],
                p_sflrff_vs_base=R.significance_testing(nr, nb),
                p_sflrff_vs_act=R.significance_testing(nr, na),
                p_act_vs_base=R.significance_testing(na, nb)))
            print(f"  {group:8s} {col:28s} base={r_base[0]:.3f} "
                  f"act={r_act[0]:.3f} sfl_rff={r_rff[0]:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out_path = CHECK_DIR / "results_features.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWROTE {out_path}")
    # quick averaged summary (over the 5 LMs) per feature
    summ = out.groupby(["group", "feature"])[["r_base", "r_act", "r_sflrff"]].mean().round(3)
    print("\nMean r over 5 LMs:\n", summ)


def _clean(col):
    return col.replace("-", "").replace("_mean", "").replace("_", "")


if __name__ == "__main__":
    main()
