#!/usr/bin/env python3
"""Held-out self-paced-reading evaluation at the best held-in SPR layer.

The SI picks the best EMB layer on a held-IN dataset and evaluates it on the held-OUT
ones. For eye-tracking this is UCL_ET -> {Provo, Meco-en, ZuCo-2}, done in
`layer_sweep.py`. For self-paced reading the held-in dataset is UCL_SPR; the
original submission found the LAST layer to be best there and therefore skipped the
held-out step (the last layer was already the main analysis). Under the new pipeline
the best held-in SPR layer is 9, not the last, so the held-out step is warranted:
this script evaluates {Brown, NatStor_SPR} at that layer.

SFL and EMB both carry i, i-1, i-2 spillover, exactly as in the main SPR analysis.

    python appendix/layer_sweep_spr_heldout.py           # layer read from layer_sweep.csv
    SPR_LAYER=9 python appendix/layer_sweep_spr_heldout.py
"""
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT
from src import extract, fit, behavioral
from src.behavioral import _spill

MODEL = "gpt2_medium"
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "appendix/layer_sweep_spr_heldout.csv"

warnings.filterwarnings("ignore", message="Solving ridge is slower")


def best_spr_layer():
    if os.environ.get("SPR_LAYER"):
        return int(os.environ["SPR_LAYER"])
    df = pd.read_csv(PROJECT_ROOT / "appendix/layer_sweep.csv")
    s = df[df.dataset == "UCL_SPR"].set_index("layer")["emb_r"]
    return int(s.idxmax())


def load_brown():
    sents = [ln.split() for ln in
             (DATA / "corpora/bnc_brown.txt").read_text().splitlines()]
    y = pd.read_csv(DATA / "brown_surprisal.csv")["psychometric"].to_numpy(float)
    sids = np.array([s for s, sent in enumerate(sents) for _ in sent[1:]])
    return sents, y, sids


def load_natstor_spr():
    END = re.compile(r'[\.!?](?:["\')\]]*)$')
    df = pd.read_csv(DATA / "natstor.tsv", sep="\t").sort_values(["item", "zone"])
    sents, rts = [], []
    for item in sorted(df["item"].unique()):
        t = df[df["item"] == item]
        cw, cr = [], []
        for w, r in zip(t["word"].astype(str), t["gmeanItemRT"]):
            cw.append(w); cr.append(r)
            if END.search(w):
                sents.append(cw); rts.append(cr); cw, cr = [], []
        if cw:
            sents.append(cw); rts.append(cr)
    y = np.array([r for rr in rts for r in rr[1:]], float)
    sids = np.array([s for s, sent in enumerate(sents) for _ in sent[1:]])
    return sents, y, sids


def evaluate(name, loader, layer, rows):
    sents, y, sids = loader()
    freqs, lens = behavioral.freq_length(sents)
    # NB: no chunked=True -- these are many short sentences, exactly as
    # behavioral.run_spill_dataset extracts them (chunked mode is for single stories).
    _, surp = extract.extract_words(MODEL, sents, drop_first=True,
                                    cache=PROJECT_ROOT / f"cache/lsweep_{name}_{MODEL}.npz")
    keep = np.array([i for i in range(len(y))
                     if i >= 2 and sids[i] == sids[i - 1] == sids[i - 2]])
    sfl_sp = _spill(np.column_stack([surp, freqs, lens]), keep)
    y_k = y[keep]
    finite = np.isfinite(y_k) & np.isfinite(sfl_sp).all(1)
    y_m = y_k[finite]
    folds = fit.make_folds(len(y_m), n_splits=10)
    s_r, s_sd, _, _ = fit.score(X=sfl_sp[finite], y=y_m, folds=folds, warn_boundary=False)
    layers = extract.extract_all_layers(MODEL, sents, drop_first=True)
    e_r, e_sd, _, _ = fit.score(X=_spill(layers[layer], keep)[finite], y=y_m, folds=folds,
                                warn_boundary=False)
    rows.append(dict(dataset=name, layer=layer, emb_r=e_r, emb_sd=e_sd,
                     sfl_r=s_r, sfl_sd=s_sd))
    print(f"  {name:<12} SFL r = {s_r:.4f} (SD {s_sd:.4f}); EMB r = {e_r:.4f} (SD {e_sd:.4f})"
          f"  -> {'SFL' if s_r > e_r else 'EMB'} wins", flush=True)


if __name__ == "__main__":
    L = best_spr_layer()
    print(f"held-out SPR evaluation at layer {L} ({MODEL})\n")
    rows = []
    for name, loader in [("Brown", load_brown), ("NatStor_SPR", load_natstor_spr)]:
        evaluate(name, loader, L, rows)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
