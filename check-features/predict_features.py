#!/usr/bin/env python3
"""Does the SFL->RFF expansion encode MEANING, or only form?

Predict 14 sentence-level ratings from three predictors, using the same ridge
pipeline as the brain encoding (10-fold sequential CV, per-fold standardized,
himalaya/sklearn):

  base     SFL (3): surprisal, length, frequency  (mean over the sentence's words)
  act      EMB (D): LM final layer, mean-pooled
  sfl_rff  SFL(3) -> RBFSampler(D), SUM-pooled per sentence   (dimensionality-matched)

If the nonlinear RFF expansion captures meaning, `sfl_rff` should predict the
SEMANTIC ratings well beyond raw SFL and approach EMB; if it only recovers
form/length structure, it should help the FORM ratings but not the semantic ones.

Self-contained: embeddings + surprisal are recomputed ex novo from the 1000 feature
sentences (a subset of the Tuckute set). Writes check-features/results_features.csv
(+ per-fold predictions under check-features/preds/).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT
from src import extract, fit, rff, behavioral, significance
from src.models import MODEL_ORDER, MODELS
from src.brain import _wsi

MODELS_TO_RUN = os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CHECK = PROJECT_ROOT / "check-features"
PREDS = CHECK / "preds"
CACHE = PROJECT_ROOT / "cache"

SEM = ["rating_imageability_mean", "rating_others_thoughts_mean", "rating_physical_mean",
       "rating_places_mean", "rating_valence_mean", "rating_arousal_mean", "rating_sense_mean"]
SYN = ["log-prob-pcfg_mean", "rating_gram_mean", "rating_frequency_mean",
       "rating_conversational_mean", "dep_length_mean", "avg_word_freq_zipf", "avg_word_length"]
TARGETS = [(c, "semantic") for c in SEM] + [(c, "form") for c in SYN]


def _clean(col):
    return col.replace("-", "").replace("_mean", "").replace("_", "")


def main():
    PREDS.mkdir(parents=True, exist_ok=True)
    feat = (pd.read_csv(CHECK / "features_semantics.csv")
              .merge(pd.read_csv(CHECK / "features_syntax.csv"), on="sentence"))
    sentences = [str(s).split() for s in feat["sentence"]]
    n = len(sentences)
    wsi = _wsi(sentences)
    freqs, lens = behavioral.freq_length(sentences)
    units = np.arange(n)
    folds = fit.make_folds(n, n_splits=10)

    rows = []
    for model in MODELS_TO_RUN:
        print(f"\n=== {model} ===", flush=True)
        D = MODELS[model]["dim"]
        emb_w, surp_w = extract.extract_words(model, sentences, drop_first=True,
                                              cache=CACHE / f"features_{model}.npz")
        word_sfl = np.column_stack([surp_w, freqs, lens])
        base = rff.pool_words_to_units(word_sfl, wsi, units, "mean")
        act = rff.pool_words_to_units(emb_w, wsi, units, "mean")
        rff_build = rff.sentence_rff_build_fn(word_sfl, wsi, units, D)

        for col, group in TARGETS:
            y = feat[col].to_numpy(float)
            nb, na, nr = (f"feat_{model}_{_clean(col)}_{s}" for s in ("base", "act", "sflrff"))
            r_base = fit.score(base, y, folds, saveto=nb, results_dir=str(PREDS),
                               backend=BACKEND, warn_boundary=False)
            r_act = fit.score(act, y, folds, saveto=na, results_dir=str(PREDS),
                              backend=BACKEND, warn_boundary=False)
            r_rff = fit.score(y=y, folds=folds, build_fn=rff_build, saveto=nr,
                              results_dir=str(PREDS), backend=BACKEND, warn_boundary=False)
            rows.append(dict(
                model=model, feature=col, group=group,
                r_base=r_base[0], r_act=r_act[0], r_sflrff=r_rff[0],
                p_sflrff_vs_base=significance.significance_testing(nr, nb, str(PREDS)),
                p_sflrff_vs_act=significance.significance_testing(nr, na, str(PREDS)),
                p_act_vs_base=significance.significance_testing(na, nb, str(PREDS))))
            print(f"  {group:8s} {col:28s} base={r_base[0]:.3f} "
                  f"act={r_act[0]:.3f} sfl_rff={r_rff[0]:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(CHECK / "results_features.csv", index=False)
    print("\nMean r over models:\n",
          out.groupby(["group", "feature"])[["r_base", "r_act", "r_sflrff"]].mean().round(3))


if __name__ == "__main__":
    main()
