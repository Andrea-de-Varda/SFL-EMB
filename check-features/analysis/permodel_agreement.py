#!/usr/bin/env python3
"""Do the EMB predicted-response vectors from DIFFERENT LMs agree with each other
more than EMB agrees with SFL+RFF? If the 5 EMB models converge on a common signal
that SFL+RFF does not share, that signal is the (semantic) content unique to EMB.

For each dataset we compute leakage-free out-of-fold predicted responses (same folds
/ RFF as the main pipeline) for each of the 5 LMs, for two predictors: EMB (last
layer, mean-pooled) and SFL+RFF. Then:
  EMB-EMB        mean pairwise r among the 5 EMB prediction vectors
  SFLRFF-SFLRFF  mean pairwise r among the 5 SFL+RFF vectors
  EMB-SFLRFF     mean r between EMB and SFL+RFF (same LM, and all cross pairs)

Self-contained (embeddings recomputed ex novo). Writes permodel_predictions_<ds>.csv
and permodel_agreement.csv.  Usage: python permodel_agreement.py [pereira control n400]
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.paths import PROJECT_ROOT
from src import extract, fit, rff, behavioral
from src.brain import _wsi
from src.models import MODEL_ORDER, MODELS

BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CACHE = PROJECT_ROOT / "cache"
DATA = PROJECT_ROOT / "data"


def load(dataset):
    if dataset == "pereira":
        m = pd.read_csv(DATA / "pereira/pereira_averaged_with_passage.csv")
        return dict(level="sentence", sentences=[str(s).split() for s in m["Sentence"]],
                    y=m["EffectSize"].to_numpy(float), groups=m["passage_group"].to_numpy(),
                    mask=None)
    if dataset == "control":
        ctl = pd.read_csv(DATA / "tuckute/brain-lang-data_participant_20230728.csv")
        g = ctl[ctl.roi == "lang_LH_netw"].groupby("sentence", sort=True)["response_target"].mean()
        return dict(level="sentence", sentences=[str(s).split() for s in g.index],
                    y=g.to_numpy(float), groups=None, mask=None)
    d = pd.read_csv(DATA / "UCL_reading_all.csv")
    sents, n400, words = [], [], []
    for sn in sorted(d["sent_nr"].unique()):
        s = d[d["sent_nr"] == sn].sort_values("word_pos")
        sents.append(list(s["word"]))
        ni = s.iloc[1:]
        n400.extend(ni["N400"].to_numpy())
        words.extend(ni["word"].astype(str))
    n400 = np.array(n400, float)
    mask = np.array([("." not in w) and ("," not in w) for w in words]) & np.isfinite(n400)
    return dict(level="word", sentences=sents, y=n400[mask], groups=None, mask=mask)


def per_model(dataset):
    L = load(dataset)
    sents, y, groups, mask, level = (L["sentences"], L["y"], L["groups"], L["mask"], L["level"])
    wsi = _wsi(sents)
    freqs, lens = behavioral.freq_length(sents)
    folds = fit.make_folds(len(y), groups=groups, n_splits=10)
    units = np.arange(len(sents))
    emb, sflrff = {}, {}
    keys = [m for m in MODEL_ORDER if m in os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")]
    for key in keys:
        Dm = MODELS[key]["dim"]
        emb_w, surp = extract.extract_words(key, sents, drop_first=True,
                                            cache=CACHE / f"permodel_{dataset}_{key}.npz")
        word_sfl = np.column_stack([surp, freqs, lens])
        if level == "sentence":
            act = rff.pool_words_to_units(emb_w, wsi, units, "mean")
            build = rff.sentence_rff_build_fn(word_sfl, wsi, units, Dm)
        else:
            act = emb_w[mask]
            build = rff.rff_build_fn(word_sfl[mask], Dm)
        emb[key] = fit.oof_predict(y, folds, X=act, backend=BACKEND)
        sflrff[key] = fit.oof_predict(y, folds, build_fn=build, backend=BACKEND)
        print(f"  {dataset} {key} done", flush=True)
    return dict(emb=emb, sflrff=sflrff, y=y)


def summarize(name, d):
    keys = [m for m in MODEL_ORDER if m in os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")]
    E = np.array([d["emb"][k] for k in keys])
    S = np.array([d["sflrff"][k] for k in keys])
    pairs = list(itertools.combinations(range(len(keys)), 2))
    emb_pairs = [pearsonr(E[i], E[j])[0] for i, j in pairs]
    sfl_pairs = [pearsonr(S[i], S[j])[0] for i, j in pairs]
    es_same = [pearsonr(E[i], S[i])[0] for i in range(len(keys))]
    es_cross = [pearsonr(E[i], S[j])[0] for i in range(len(keys)) for j in range(len(keys))]
    cols = {f"emb_{k}": d["emb"][k] for k in keys}
    cols.update({f"sflrff_{k}": d["sflrff"][k] for k in keys})
    cols["y"] = d["y"]
    pd.DataFrame(cols).to_csv(os.path.join(HERE, f"permodel_predictions_{name}.csv"), index=False)
    return dict(dataset=name, EMB_EMB=np.mean(emb_pairs), EMB_EMB_min=np.min(emb_pairs),
                EMB_EMB_max=np.max(emb_pairs), SFLRFF_SFLRFF=np.mean(sfl_pairs),
                EMB_SFLRFF_sameLM=np.mean(es_same), EMB_SFLRFF_cross=np.mean(es_cross))


def main():
    rows = [summarize(ds, per_model(ds)) for ds in (sys.argv[1:] or ["pereira", "control", "n400"])]
    out = pd.DataFrame(rows).round(3)
    out.to_csv(os.path.join(HERE, "permodel_agreement.csv"), index=False)
    print("\nPredicted-response agreement (Pearson r)\n" + out.to_string(index=False))
    print("\nEMB_EMB = mean pairwise r among the 5 EMB vectors; EMB_SFLRFF_sameLM = mean r "
          "between EMB and SFL+RFF within an LM. EMB_EMB >> EMB_SFLRFF => EMB models share a "
          "common signal SFL+RFF does not capture.")


if __name__ == "__main__":
    main()
