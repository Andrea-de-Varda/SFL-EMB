#!/usr/bin/env python3
"""Appendix A4: layer sweep (behavioral eye-tracking, GPT-2 355M).

For each layer, fit EMB (that layer's per-word vectors) to first-pass gaze and
compare to the layer-independent SFL baseline. The best layer is chosen on a
held-in dataset (UCL-ET) and evaluated on held-out datasets (here Provo; Meco /
ZuCo follow the same pattern). The paper reports the best held-in ET layer = 3.

Writes appendix/layer_sweep.csv (rows = dataset x layer; cols = emb_r, sfl_r).
gpt2_medium only, per the paper. Sentences are short, so no chunking is needed.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT
from src import extract, fit, behavioral

MODEL = "gpt2_medium"
DATA = PROJECT_ROOT / "data"


def load_ucl_et():
    d = pd.read_csv(DATA / "UCL_reading_all.csv")
    sents, y = [], []
    for sn in sorted(d["sent_nr"].unique()):
        s = d[d["sent_nr"] == sn].sort_values("word_pos")
        sents.append(list(s["word"]))
        y.extend(s.iloc[1:]["RTfirstpass"].to_numpy())
    return sents, np.array(y, float)


def load_provo():
    provo = pd.read_csv(DATA / "Provo/Provo_Corpus-Eyetracking_Data.csv")
    provo["IA_FIRST_RUN_DWELL_TIME"] = provo["IA_FIRST_RUN_DWELL_TIME"].fillna(0)
    provo = provo.groupby(["Sentence_Number", "Word_In_Sentence_Number", "Word", "Text_ID"],
                          as_index=False, sort=False).agg(
        {"IA_FIRST_RUN_DWELL_TIME": "mean", "Text_ID": "mean"})
    first = [s.split("\t")[2][1:-2].split()[0]
             for s in open(DATA / "Provo/sentences.txt").readlines()]
    d_first = {n + 1: first[n] for n in range(55)}
    sents, y = [], []
    for t in sorted(set(provo["Text_ID"])):
        tmp = provo[provo["Text_ID"] == t]
        sents.append([d_first[t]] + list(tmp["Word"]))
        y.extend(tmp["IA_FIRST_RUN_DWELL_TIME"])
    return sents, np.array(y, float)


def sfl_matrix(sentences):
    freqs, lens = behavioral.freq_length(sentences)
    # surprisal from the same model (last layer's LM head); reuse the cached extractor
    _, surp = extract.extract_words(MODEL, sentences, drop_first=True, cache=None)
    return np.column_stack([surp, freqs, lens])


def sweep(sentences, y, name, rows):
    folds = fit.make_folds(len(y), n_splits=10)
    layers = extract.extract_all_layers(MODEL, sentences, drop_first=True)
    X_sfl = sfl_matrix(sentences)
    sfl_r = fit.score(X_sfl, y, folds, warn_boundary=False)[0]
    for li, emb in sorted(layers.items()):
        emb_r = fit.score(emb, y, folds, warn_boundary=False)[0]
        rows.append({"dataset": name, "layer": li, "emb_r": emb_r, "sfl_r": sfl_r})
        print(f"  {name} layer {li:2d}: EMB {emb_r:.4f}  (SFL {sfl_r:.4f})")


if __name__ == "__main__":
    rows = []
    for name, loader in [("UCL_ET", load_ucl_et), ("Provo", load_provo)]:
        print(f"=== {name} ===")
        sweep(*loader(), name, rows)
    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "appendix/layer_sweep.csv"
    df.to_csv(out, index=False)
    best = (df[df.dataset == "UCL_ET"].set_index("layer")["emb_r"].idxmax())
    print(f"\nbest held-in (UCL_ET) EMB layer = {best}")
    print(f"wrote {out}")
