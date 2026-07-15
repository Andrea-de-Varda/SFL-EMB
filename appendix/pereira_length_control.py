#!/usr/bin/env python3
"""SI dimensionality control: is SFL_RFF's Pereira advantage just sentence length?

Pereira2018 is the one neural dataset where SFL_RFF outperforms EMB. The word-level
RFF features are SUM-pooled per sentence (`src/brain.py`, the original convention),
so a sentence's SFL_RFF vector scales with its word count and encodes sentence
length by construction -- and length is itself a good predictor of the Pereira
response. This script asks what each predictor adds ON TOP of length:

    length          sentence length alone (1 feature)
    SFL_RFF         sum-pooled RFF (reference)
    EMB             mean-pooled last layer (reference)
    length+SFL_RFF  banded ridge, block boundary at column 1
    length+EMB      banded ridge, block boundary at column 1

Banded ridge is required for the concatenations: a single global alpha over 1 strong
scalar and D RFF/EMB dims over-shrinks the scalar, which would depress the
concatenation for reasons unrelated to content (see README).

Same folds (GroupKFold by passage), estimator, and paired permutation test as the
main pipeline. Runs locally on one GPU in a few minutes.

    python appendix/pereira_length_control.py                  # all 5 models
    SFL_MODELS=gpt2_xl python appendix/pereira_length_control.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT
from src import extract, fit, rff, behavioral
from src.brain import _wsi
from src.models import MODELS, MODEL_ORDER
from src.significance import permutation_statistic

MODELS_TO_RUN = os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CACHE = PROJECT_ROOT / "cache"
SEED = 0

warnings.filterwarnings("ignore", message="Solving ridge is slower")


def ns(p):
    """Two-sided n.s. at .025, the criterion used for the paper figures."""
    return min(p, 1 - p) > 0.025


def main():
    m = pd.read_csv(PROJECT_ROOT / "data/pereira/pereira_averaged_with_passage.csv")
    sentences = [str(s).split() for s in m["Sentence"]]
    y = m["EffectSize"].to_numpy(float)
    groups = m["passage_group"].to_numpy()
    nwords = np.array([len(s) for s in sentences], float)[:, None]   # (n, 1)

    folds = fit.make_folds(len(y), groups=groups, n_splits=10)
    wsi = _wsi(sentences)
    freqs, lens = behavioral.freq_length(sentences)
    units = np.arange(len(sentences))

    print(f"Pereira2018: {len(sentences)} sentences, {m['passage_group'].nunique()} passages")
    print(f"corr(sentence length, response) = {pearsonr(nwords.ravel(), y)[0]:+.4f}\n")

    rows = []
    for model in MODELS_TO_RUN:
        D = MODELS[model]["dim"]
        emb_w, surp = extract.extract_words(model, sentences, drop_first=True,
                                            cache=CACHE / f"pereira_{model}.npz")
        word_sfl = np.column_stack([surp, freqs, lens])
        emb = rff.pool_words_to_units(emb_w, wsi, units, "mean")          # (n, D)

        def rff_pooled(train, _ws=word_sfl, _D=D):
            """Sum-pooled RFF, fit on the train fold's words only (leakage-free)."""
            t = rff.fit_rff(_ws[np.isin(wsi, units[train])], _D, SEED)
            return rff.pool_words_to_units(t(_ws), wsi, units, "sum")     # (n, D)

        def b_rff(train, test):
            p = rff_pooled(train)
            return p[train], p[test]

        def b_len_rff(train, test):
            p = rff_pooled(train)
            return (np.hstack([nwords[train], p[train]]),
                    np.hstack([nwords[test], p[test]]))

        specs = {
            "length":         dict(X=nwords),
            "sfl_rff":        dict(build_fn=b_rff),
            "emb":            dict(X=emb),
            "length+sfl_rff": dict(build_fn=b_len_rff, banded_split=1),
            "length+emb":     dict(X=np.hstack([nwords, emb]), banded_split=1),
        }

        r = {}
        for name, spec in specs.items():
            mean_r, sd, fold_rs, _ = fit.score(y=y, folds=folds, backend=BACKEND,
                                               warn_boundary=False, **spec)
            r[name] = fold_rs
            print(f"  {model:<12} {name:<15} r = {mean_r:.4f}  (sd {sd:.4f})")

        rows.append(dict(
            model=model,
            length=np.mean(r["length"]),
            sfl_rff=np.mean(r["sfl_rff"]),
            emb=np.mean(r["emb"]),
            length_sfl_rff=np.mean(r["length+sfl_rff"]),
            length_emb=np.mean(r["length+emb"]),
            d_rff_over_length=np.mean(r["length+sfl_rff"]) - np.mean(r["length"]),
            d_emb_over_length=np.mean(r["length+emb"]) - np.mean(r["length"]),
            p_rff_over_length=permutation_statistic(r["length+sfl_rff"], r["length"]),
            p_emb_over_length=permutation_statistic(r["length+emb"], r["length"]),
            p_emb_vs_rff=permutation_statistic(r["length+emb"], r["length+sfl_rff"]),
        ))
        print()

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results/pereira_length_control.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 220, "display.precision", 4)
    print("=" * 100)
    print(df.to_string(index=False))
    print("\nmean over models:")
    print(f"  length              {df.length.mean():.4f}")
    print(f"  length+SFL_RFF      {df.length_sfl_rff.mean():.4f}   "
          f"(delta over length {df.d_rff_over_length.mean():+.4f}, "
          f"sig {sum(not ns(p) for p in df.p_rff_over_length)}/{len(df)} models)")
    print(f"  length+EMB          {df.length_emb.mean():.4f}   "
          f"(delta over length {df.d_emb_over_length.mean():+.4f}, "
          f"sig {sum(not ns(p) for p in df.p_emb_over_length)}/{len(df)} models)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
