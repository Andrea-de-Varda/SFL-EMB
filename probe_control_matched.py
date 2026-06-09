#!/usr/bin/env python3
"""
Tuckute2024 (control_fmri) SFL-vs-EMB with the dimensionality-matched predictors.
Sentence-level fMRI (2000 sentences, language network), 10-fold SEQUENTIAL KFold
(unchanged from the original control_encoding.py; no passages -> groups=None).

Predictors per model (D = embedding dim): base (SFL-3 mean), act (EMB-D mean),
all (SFL^EMB), emb_sum (EMB-D sum), sfl_rff (per-word SFL -> RBFSampler(D) -> SUM
per sentence), all_matched (emb_sum ^ sfl_rff). See rff_matched.py for the
leakage-free RFF machinery and the scoring/significance (identical to
utils.test_model_Ridge / significance_testing).

Overwrites control_fmri/results/control_results.csv, keeping the base/act/all +
p_base_emb/p_all_base/p_all_emb columns plotting.py reads, plus matched columns
and all-pairs permutation p-values.
"""
import sys
import itertools
from os import chdir
from pathlib import Path as _Path
import numpy as np
import pandas as pd

_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())

MODELS = [("gpt", "s_GPT", 12, 768), ("small", "s_small", 12, 768),
          ("med", "s_med", 24, 1024), ("large", "s_large", 36, 1280),
          ("xl", "s_xl", 48, 1600)]
ACT_FILE = {"gpt": "activations_gpt.json", "small": "activations_gpt2_small.json",
            "med": "activations_gpt2_medium.json", "large": "activations_gpt2_large.json",
            "xl": "activations_gpt2_xl.json"}
ROW_INDEX = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
PREDS = ["base", "act", "all", "emb_sum", "sfl_rff", "all_matched"]
SUF = {"base": "_base", "act": "", "all": "_all",
       "emb_sum": "_embsum", "sfl_rff": "_sflrff", "all_matched": "_allmatched"}

def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    import rff_matched as R
    chdir(str(PROJECT_ROOT / "control_fmri"))
    import neurox.data.loader as data_loader

    # y: per-sentence language-network response, groupby-sorted (== sentences.txt
    # == activations == SFL.csv order). Identical to the original control_encoding.py.
    control = pd.read_csv("data/brain-lang-data_participant_20230728.csv")
    df_control = (control[control["roi"] == "lang_LH_netw"]
                  .groupby("sentence").agg({"response_target": "mean", "cond": "max"}))
    y = df_control["response_target"].to_numpy(dtype=np.float64)
    n = len(y)
    df = pd.read_csv("SFL.csv")
    wsi = df["sentnum"].to_numpy(dtype=np.int64)
    assert wsi.min() == 0 and wsi.max() == n - 1, f"sentnum {wsi.min()}..{wsi.max()} != 0..{n-1}"

    def pool_sfl(col):
        return np.array([list(df[df.sentnum == s][[col, "len", "freq"]].mean()) for s in range(n)])

    cols = {c: [] for c in PREDS}
    cols.update({c + "_sd": [] for c in PREDS})
    for key, scol, layer, dim in MODELS:
        print(f"\n=== {key} (D={dim}) ===", flush=True)
        act = data_loader.load_activations(ACT_FILE[key], dim, is_brnn=False)[0]
        emb_mean = R.get_act(act, layer, dim, "mean")
        emb_sum = R.get_act(act, layer, dim, "sum")
        del act
        base = pool_sfl(scol)
        allcat = np.hstack([base, emb_mean])
        word_sfl = df[[scol, "len", "freq"]].to_numpy(dtype=np.float64)

        m = {"base": R.score_matrix(base, y, f"control_{key}_base"),
             "act": R.score_matrix(emb_mean, y, f"control_{key}"),
             "all": R.score_matrix(allcat, y, f"control_{key}_all"),
             "emb_sum": R.score_matrix(emb_sum, y, f"control_{key}_embsum")}
        m["sfl_rff"], m["all_matched"] = R.score_sfl_rff_sentence(
            word_sfl, wsi, y, emb_sum, dim, groups=None,
            name_rff=f"control_{key}_sflrff", name_all=f"control_{key}_allmatched")
        for c in PREDS:
            cols[c].append(m[c][0]); cols[c + "_sd"].append(m[c][1])
        print({c: round(m[c][0], 4) for c in PREDS}, flush=True)

    df_results = pd.DataFrame({c: cols[c] for c in
                               ["base", "base_sd", "act", "act_sd", "all", "all_sd",
                                "emb_sum", "emb_sum_sd", "sfl_rff", "sfl_rff_sd",
                                "all_matched", "all_matched_sd"]}, index=ROW_INDEX)

    def pname(key, pred):
        return f"control_{key}{SUF[pred]}"
    for a, b in itertools.combinations(PREDS, 2):
        df_results[f"p__{a}__{b}"] = [R.significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]
    for col, (a, b) in {"p_base_emb": ("act", "base"), "p_all_base": ("all", "base"),
                        "p_all_emb": ("all", "act")}.items():
        df_results[col] = [R.significance_testing(pname(k, a), pname(k, b)) for k, *_ in MODELS]

    df_results.to_csv("results/control_results.csv", index=False)
    print("\nWROTE control_fmri/results/control_results.csv")
    print(df_results[PREDS].round(4))

if __name__ == "__main__":
    main()
