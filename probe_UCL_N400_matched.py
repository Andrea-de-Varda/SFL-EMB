#!/usr/bin/env python3
"""
UCL N400 (ERP) SFL-vs-EMB with the dimensionality-matched SFL predictor.
WORD-LEVEL: one N400 amplitude per word (no spill-over for N400). 10-fold
SEQUENTIAL KFold (unchanged from probe_ET.py).

Because the prediction unit is the word, there is NO sum-over-words pooling: the
matched SFL is simply per-word SFL(3) -> StandardScaler(train) -> RBFSampler(D),
giving a D-dim per-word predictor that is already dimensionality-matched to the
D-dim per-word embedding `act`. Predictors:
  base        SFL (3)
  act         EMB (D)              mean-over-subword pooling (neurox 'average')
  all         SFL ^ EMB            (3+D)
  sfl_rff     SFL(3) -> RBFSampler(D)   dimensionality-matched SFL (per word)
  all_matched act ^ sfl_rff        (2D)

emb_sum (sum-over-subwords EMB) is NOT produced here: neurox's extractor only
supports {first,last,average}, so a sum variant needs a custom re-extraction
(monkeypatch aggregate_repr to np.sum). At the word level this sum-over-subwords
is only weakly analogous to the sentence/TR sum-over-words anyway, so act-vs-sfl_rff
is the dimensionality-matched contrast. To add emb_sum, set MAKE_EMB_SUM=True
(re-extracts activations with sum pooling -- a GPU job).

Writes results/results_UCL_N400_matched.csv (separate from the multi-analysis
results_UCL.csv) with base/act/all + matched columns + all-pairs permutation p's.
"""
import sys
import itertools
from os import chdir, path
from pathlib import Path as _Path
import numpy as np
import pandas as pd

_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())

# (model key, surprisal column in UCL_reading_all.csv, last-layer index, dim D)
MODELS = [("gpt", "s_GPT", 12, 768), ("small", "s_GPT2_small", 12, 768),
          ("med", "s_GPT2_med", 24, 1024), ("large", "s_GPT2_large", 36, 1280),
          ("xl", "s_GPT2_xl", 48, 1600)]
ACT_FILE = {"gpt": "activations_gpt.json", "small": "activations_gpt2_small.json",
            "med": "activations_gpt2_medium.json", "large": "activations_gpt2_large.json",
            "xl": "activations_gpt2_xl.json"}
ROW_INDEX = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
PREDS = ["base", "act", "all", "sfl_rff", "all_matched"]   # no emb_sum (see header)
SUF = {"base": "_base", "act": "", "all": "_all", "sfl_rff": "_sflrff", "all_matched": "_allmatched"}

def get_layer(vec, layer, dim):
    return list(vec[layer * dim:(layer + 1) * dim])

def get_act_eye(act_list, layer, dim, available):
    """Per-word embeddings: for each sentence (ascending), words 1..n (drop the
    sentence-initial word -> aligns with df_noinit, which drops it via NaN surprisal)."""
    out = []
    for i in sorted(available):
        for v in act_list[i][1:]:
            out.append(get_layer(v, layer, dim))
    return np.array(out)

def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    import rff_matched as R
    chdir(str(PROJECT_ROOT))   # UCL activations + UCL_reading_all.csv live at repo root
    import neurox.data.loader as data_loader

    df_all = pd.read_csv("UCL_reading_all.csv")
    df_noinit = df_all.dropna(subset=["s_GPT2_small"])          # drop sentence-initial words
    filtr = np.array([("." not in w) and ("," not in w) for w in df_noinit["word"]])  # drop punctuated
    y = np.asarray(df_noinit["N400"], dtype=np.float64)[filtr]
    available = set(df_all.sent_nr - 1)
    valid = ~np.isnan(y)                                        # guard: N400 may have gaps
    if not valid.all():
        print(f"[warn] dropping {int((~valid).sum())} words with NaN N400")
    y = y[valid]
    n = len(y)
    print(f"N400 words: {n}", flush=True)

    cols = {c: [] for c in PREDS}
    cols.update({c + "_sd": [] for c in PREDS})
    for key, scol, layer, dim in MODELS:
        print(f"\n=== {key} (D={dim}) ===", flush=True)
        word_sfl = np.asarray(df_noinit[[scol, "logfreq", "length"]], dtype=np.float64)[filtr][valid]
        act_all = data_loader.load_activations(ACT_FILE[key], dim, is_brnn=False)[0]
        emb = get_act_eye(act_all, layer, dim, available)
        assert len(emb) == len(df_noinit), f"emb/df_noinit misaligned: {len(emb)} vs {len(df_noinit)}"
        emb = emb[filtr][valid]
        del act_all

        m = {"base": R.score_matrix(word_sfl, y, f"UCL_n400_matched_{key}_base"),
             "act": R.score_matrix(emb, y, f"UCL_n400_matched_{key}"),
             "all": R.score_matrix(np.hstack([word_sfl, emb]), y, f"UCL_n400_matched_{key}_all")}
        rff_stat, all_stat = R.score_word_rff(
            word_sfl, y, dim, emb=emb,
            name_rff=f"UCL_n400_matched_{key}_sflrff", name_all=f"UCL_n400_matched_{key}_allmatched")
        m["sfl_rff"], m["all_matched"] = rff_stat, all_stat
        for c in PREDS:
            cols[c].append(m[c][0]); cols[c + "_sd"].append(m[c][1])
        print({c: round(m[c][0], 4) for c in PREDS}, flush=True)

    df_results = pd.DataFrame({c: cols[c] for c in
                               ["base", "base_sd", "act", "act_sd", "all", "all_sd",
                                "sfl_rff", "sfl_rff_sd", "all_matched", "all_matched_sd"]},
                              index=ROW_INDEX)

    def pname(key, pred):
        return f"UCL_n400_matched_{key}{SUF[pred]}"
    for a, b in itertools.combinations(PREDS, 2):
        df_results[f"p__{a}__{b}"] = [R.significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]
    for col, (a, b) in {"p_base_emb": ("act", "base"), "p_all_base": ("all", "base"),
                        "p_all_emb": ("all", "act")}.items():
        df_results[col] = [R.significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]

    df_results.to_csv("results/results_UCL_N400_matched.csv", index=False)
    print("\nWROTE results/results_UCL_N400_matched.csv")
    print(df_results[PREDS].round(4))

if __name__ == "__main__":
    main()
