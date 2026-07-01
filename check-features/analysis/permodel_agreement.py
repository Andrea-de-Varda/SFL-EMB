#!/usr/bin/env python3
"""
POINT 1: do the EMB predicted-response vectors from DIFFERENT LMs agree with each
other more than EMB agrees with SFL+RFF? If EMB models converge on a common signal
that SFL+RFF does not share, that signal is the (semantic) content unique to EMB.

For each dataset we compute leakage-free OUT-OF-FOLD predicted responses (same folds
/ gamma as the reported pipeline) for EACH of the 5 LMs, for two predictors:
  EMB      = LM last layer, mean-pooled (act)   SFL+RFF = SFL(3)->RBFSampler(D)
Then per dataset:
  EMB-EMB       = mean pairwise Pearson r among the 5 EMB prediction vectors
  SFLRFF-SFLRFF = mean pairwise r among the 5 SFL+RFF vectors
  EMB-SFLRFF    = mean r between EMB and SFL+RFF (same LM)  and (all 25 cross pairs)

Needs neurox + all-model activations (NOT available locally; N400 activations are
cluster-only) -> run on Engaging:  sbatch check-features/analysis/slurm_permodel.sh
Writes permodel_predictions_<ds>.csv and permodel_agreement.csv (+ prints the table).
"""
import sys, itertools
from os import chdir
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import rff_matched as R                       # noqa: E402

# (key, activation file, surprisal-col in the sentence SFL.csv, N400 surprisal-col, layer, dim)
MODELS = [("gpt",   "activations_gpt.json",          "s_GPT",   "s_GPT",       12, 768),
          ("small", "activations_gpt2_small.json",   "s_small", "s_GPT2_small", 12, 768),
          ("med",   "activations_gpt2_medium.json",  "s_med",   "s_GPT2_med",   24, 1024),
          ("large", "activations_gpt2_large.json",   "s_large", "s_GPT2_large", 36, 1280),
          ("xl",    "activations_gpt2_xl.json",      "s_xl",    "s_GPT2_xl",    48, 1600)]


def get_layer(vec, layer, dim):
    return np.asarray(vec[layer * dim:(layer + 1) * dim])


def oof_linear(X, y, groups):
    pred = np.full(len(y), np.nan)
    for test in R.make_folds(len(y), groups, 10):
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        pred[test] = R.fit_ridge(X[train], y[train]).predict(X[test])
    return pred


def oof_sfl_rff_sentence(word_sfl, wsi, y, n_units, groups, D):
    pred = np.full(n_units, np.nan)
    for test in R.make_folds(n_units, groups, 10):
        train = np.setdiff1d(np.arange(n_units), test, assume_unique=True)
        g = R._select_gamma_sentence(word_sfl, wsi, y, train, groups, D)
        imp, sca, rff = R.fit_sfl_rff(word_sfl, wsi, train, g, D)
        xtr = R.words_to_units(word_sfl, wsi, train, imp, sca, rff, "sum")
        xte = R.words_to_units(word_sfl, wsi, test, imp, sca, rff, "sum")
        pred[test] = R.fit_ridge(xtr, y[train]).predict(xte)
    return pred


def oof_sfl_rff_word(word_sfl, y, D):
    n = len(y); pred = np.full(n, np.nan)
    for test in R.make_folds(n, None, 10):
        train = np.setdiff1d(np.arange(n), test, assume_unique=True)
        g = R._select_gamma_word(word_sfl, y, train, D)
        X = R._rff_word_features(word_sfl, train, g, D)
        pred[test] = R.fit_ridge(X[train], y[train]).predict(X[test])
    return pred


def per_model(dataset):
    """Return {'emb': {model: vec}, 'sflrff': {model: vec}, 'y': y}."""
    import neurox.data.loader as dl
    emb, sflrff = {}, {}
    if dataset in ("pereira", "control"):
        if dataset == "pereira":
            chdir(str(ROOT / "pereira_1"))
            meta = pd.read_csv("pereira_averaged_with_passage.csv")
            y = meta["EffectSize"].to_numpy(float); groups = meta["passage_group"].to_numpy()
        else:
            chdir(str(ROOT / "control_fmri"))
            ctl = pd.read_csv("data/brain-lang-data_participant_20230728.csv")
            y = (ctl[ctl.roi == "lang_LH_netw"].groupby("sentence")
                 .agg({"response_target": "mean"})["response_target"].to_numpy(float))
            groups = None
        sfl = pd.read_csv("SFL.csv"); wsi = sfl["sentnum"].to_numpy(int); n = len(y)
        for key, af, scol, _, layer, dim in MODELS:
            act_list = dl.load_activations(af, dim, is_brnn=False)[0]
            act = R.get_act(act_list, layer, dim, "mean"); del act_list
            emb[key] = oof_linear(np.asarray(act), y, groups)
            wsfl = sfl[[scol, "len", "freq"]].to_numpy(float)
            sflrff[key] = oof_sfl_rff_sentence(wsfl, wsi, y, n, groups, dim)
            print(f"  {dataset} {key} done", flush=True)
        return dict(emb=emb, sflrff=sflrff, y=y)
    elif dataset == "n400":
        chdir(str(ROOT))
        da = pd.read_csv("UCL_reading_all.csv")
        dni = da.dropna(subset=["s_GPT2_small"])
        filtr = np.array([("." not in w) and ("," not in w) for w in dni["word"]])
        y = np.asarray(dni["N400"], float)[filtr]; valid = ~np.isnan(y); y = y[valid]
        available = sorted(set(da.sent_nr - 1))

        def per_word(act_list, layer, dim):
            out = [get_layer(v, layer, dim) for i in available for v in act_list[i][1:]]
            return np.array(out)
        for key, af, _, scol, layer, dim in MODELS:
            act_list = dl.load_activations(af, dim, is_brnn=False)[0]
            e = per_word(act_list, layer, dim)[filtr][valid]; del act_list
            emb[key] = oof_linear(e, y, None)
            wsfl = np.asarray(dni[[scol, "logfreq", "length"]], float)[filtr][valid]
            sflrff[key] = oof_sfl_rff_word(wsfl, y, dim)
            print(f"  n400 {key} done", flush=True)
        return dict(emb=emb, sflrff=sflrff, y=y)


def summarize(name, d):
    keys = [m[0] for m in MODELS]
    E = np.array([d["emb"][k] for k in keys])
    S = np.array([d["sflrff"][k] for k in keys])
    emb_pairs = [pearsonr(E[i], E[j])[0] for i, j in itertools.combinations(range(5), 2)]
    sfl_pairs = [pearsonr(S[i], S[j])[0] for i, j in itertools.combinations(range(5), 2)]
    es_same = [pearsonr(E[i], S[i])[0] for i in range(5)]
    es_cross = [pearsonr(E[i], S[j])[0] for i in range(5) for j in range(5)]
    # save per-model predictions for reuse
    cols = {f"emb_{k}": d["emb"][k] for k in keys}
    cols.update({f"sflrff_{k}": d["sflrff"][k] for k in keys}); cols["y"] = d["y"]
    pd.DataFrame(cols).to_csv(HERE / f"permodel_predictions_{name}.csv", index=False)
    return dict(dataset=name,
                EMB_EMB=np.mean(emb_pairs), EMB_EMB_min=np.min(emb_pairs), EMB_EMB_max=np.max(emb_pairs),
                SFLRFF_SFLRFF=np.mean(sfl_pairs),
                EMB_SFLRFF_sameLM=np.mean(es_same), EMB_SFLRFF_cross=np.mean(es_cross))


def main():
    which = sys.argv[1:] or ["pereira", "control", "n400"]
    rows = []
    for ds in which:
        print(f"\n=== {ds} ===", flush=True)
        rows.append(summarize(ds, per_model(ds)))
    out = pd.DataFrame(rows).round(3)
    out.to_csv(HERE / "permodel_agreement.csv", index=False)
    print("\n" + "=" * 78)
    print("POINT 1 — predicted-response agreement (Pearson r)")
    print("=" * 78)
    print(out.to_string(index=False))
    print("\nEMB_EMB = mean pairwise r among the 5 EMB vectors; EMB_SFLRFF_sameLM = mean r")
    print("between EMB and SFL+RFF within an LM. If EMB_EMB >> EMB_SFLRFF, the EMB models")
    print("share a common signal that SFL+RFF does not capture.")


if __name__ == "__main__":
    main()
