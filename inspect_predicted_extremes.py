#!/usr/bin/env python3
"""
Interpretability: which items are PREDICTED to elicit the highest / lowest brain or
ERP response, under each of three encoding predictors, and do those predictors
encode the same information?

  act      EMB (D), mean-over-subword pooling      (the original embedding readout)
  emb_sum  EMB (D), sum-over-(sub)word pooling
  sfl_rff  SFL(3) -> RBFSampler(D)  (dimensionality-matched SFL)

For each dataset (Pereira, Tuckute/control, UCL-N400) and one LM (GPT-2 XL, D=1600)
we compute leakage-free OUT-OF-FOLD predictions: each item is predicted by the model
trained on the folds that exclude it, using the SAME folds / gamma selection /
sum-pooling as the reported scoring functions in rff_matched (so the predicted
values are consistent with the published r's). Then per predictor we list the
top-10 / bottom-10 items by predicted response (with text + context), and quantify
agreement: Spearman/Pearson between the three predicted-value vectors, top/bottom-10
set overlap, and correlation of each predicted vector with item length / mean
surprisal / mean log-frequency (to see WHAT each predictor tracks).

Usage (neurox env, on the cluster -- xl activations are large; N400 needs the
sum-pooled activations produced by `probe_UCL_N400_matched.py` first):
    python inspect_predicted_extremes.py            # all three
    python inspect_predicted_extremes.py pereira    # one

Writes results_extremes/<dataset>_xl_{report.md,predictions.csv}.
"""
import sys
from os import chdir, path, makedirs
from pathlib import Path as _Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
sys.path.insert(0, str(PROJECT_ROOT))
import rff_matched as R                       # noqa: E402

LM = ("xl", 48, 1600)                         # GPT-2 XL: key, last-layer index, dim
PREDS = ["act", "emb_sum", "sfl_rff"]
OUTDIR = str(PROJECT_ROOT / "results_extremes")
N_TOP = 10


def get_layer(vec, layer, dim):
    return np.asarray(vec[layer * dim:(layer + 1) * dim])


# ---------------------------------------------------------------- OOF predictors
def oof_linear(X, y, groups, n_splits=10):
    """Cross-validated predictions of a linear ridge on feature matrix X."""
    pred = np.full(len(y), np.nan)
    for test in R.make_folds(len(y), groups, n_splits):
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        pred[test] = R.fit_ridge(X[train], y[train]).predict(X[test])
    return pred


def oof_sfl_rff_sentence(word_sfl, wsi, y, n_units, groups, D, n_splits=10):
    """OOF preds mirroring rff_matched.score_sfl_rff_sentence (nested-gamma, sum-pool)."""
    pred = np.full(n_units, np.nan)
    allu = np.arange(n_units)
    for test in R.make_folds(n_units, groups, n_splits):
        train = np.setdiff1d(allu, test, assume_unique=True)
        g = R._select_gamma_sentence(word_sfl, wsi, y, train, groups, D)
        imp, sca, rff = R.fit_sfl_rff(word_sfl, wsi, train, g, D)
        x_tr = R.words_to_units(word_sfl, wsi, train, imp, sca, rff, "sum")
        x_te = R.words_to_units(word_sfl, wsi, test, imp, sca, rff, "sum")
        pred[test] = R.fit_ridge(x_tr, y[train]).predict(x_te)
    return pred


def oof_sfl_rff_word(word_sfl, y, D, n_splits=10):
    """OOF preds mirroring rff_matched.score_word_rff (nested-gamma, per-word RFF)."""
    n = len(y)
    pred = np.full(n, np.nan)
    for test in R.make_folds(n, None, n_splits):
        train = np.setdiff1d(np.arange(n), test, assume_unique=True)
        g = R._select_gamma_word(word_sfl, y, train, D)
        X = R._rff_word_features(word_sfl, train, g, D)
        pred[test] = R.fit_ridge(X[train], y[train]).predict(X[test])
    return pred


# ----------------------------------------------------------------- data loaders
def load_pereira(key, layer, dim):
    import neurox.data.loader as dl
    chdir(str(PROJECT_ROOT / "pereira_1"))
    meta = pd.read_csv("pereira_averaged_with_passage.csv")
    y = meta["EffectSize"].to_numpy(float)
    groups = meta["passage_group"].to_numpy()
    text = meta["Sentence"].astype(str).to_numpy()
    sfl = pd.read_csv("SFL.csv")
    wsi = sfl["sentnum"].to_numpy(int)
    word_sfl = sfl[[f"s_{key}", "len", "freq"]].to_numpy(float)
    # per-sentence covariates (mean over words) for interpretation
    cov = sfl.groupby("sentnum").agg({f"s_{key}": "mean", "len": "mean", "freq": "mean"})
    covars = {"surprisal": cov[f"s_{key}"].to_numpy(), "length": cov["len"].to_numpy(),
              "logfreq": cov["freq"].to_numpy()}
    act_list = dl.load_activations(f"activations_gpt2_{key}.json", dim, is_brnn=False)[0]
    act = np.array([get_layer(s[1:].mean(0), layer, dim) for s in act_list])
    emb_sum = np.array([get_layer(s[1:].sum(0), layer, dim) for s in act_list])
    del act_list
    return dict(name="pereira", level="sentence", y=y, groups=groups, text=text, context=None,
                act=act, emb_sum=emb_sum, word_sfl=word_sfl, wsi=wsi, n_units=len(y),
                D=dim, covars=covars, unit="EffectSize")


def load_control(key, layer, dim):
    import neurox.data.loader as dl
    chdir(str(PROJECT_ROOT / "control_fmri"))
    ctl = pd.read_csv("data/brain-lang-data_participant_20230728.csv")
    dc = ctl[ctl["roi"] == "lang_LH_netw"].groupby("sentence").agg({"response_target": "mean"})
    y = dc["response_target"].to_numpy(float)
    text = dc.index.astype(str).to_numpy()          # sentence strings, same order as y
    sfl = pd.read_csv("SFL.csv")
    wsi = sfl["sentnum"].to_numpy(int)
    word_sfl = sfl[[f"s_{key}", "len", "freq"]].to_numpy(float)
    cov = sfl.groupby("sentnum").agg({f"s_{key}": "mean", "len": "mean", "freq": "mean"})
    covars = {"surprisal": cov[f"s_{key}"].to_numpy(), "length": cov["len"].to_numpy(),
              "logfreq": cov["freq"].to_numpy()}
    act_list = dl.load_activations(f"activations_gpt2_{key}.json", dim, is_brnn=False)[0]
    act = np.array([get_layer(s[1:].mean(0), layer, dim) for s in act_list])
    emb_sum = np.array([get_layer(s[1:].sum(0), layer, dim) for s in act_list])
    del act_list
    return dict(name="control", level="sentence", y=y, groups=None, text=text, context=None,
                act=act, emb_sum=emb_sum, word_sfl=word_sfl, wsi=wsi, n_units=len(y),
                D=dim, covars=covars, unit="BOLD (lang LH)")


def load_n400(key, layer, dim):
    import neurox.data.loader as dl
    chdir(str(PROJECT_ROOT))
    scol = {"xl": "s_GPT2_xl"}[key]
    df_all = pd.read_csv("UCL_reading_all.csv")
    df_noinit = df_all.dropna(subset=["s_GPT2_small"])
    filtr = np.array([("." not in w) and ("," not in w) for w in df_noinit["word"]])
    y = np.asarray(df_noinit["N400"], float)[filtr]
    valid = ~np.isnan(y)
    y = y[valid]
    available = set(df_all.sent_nr - 1)

    def per_word(act_list):
        out = []
        for i in sorted(available):
            for v in act_list[i][1:]:
                out.append(get_layer(v, layer, dim))
        return np.array(out)

    a_list = dl.load_activations(f"activations_gpt2_{key}.json", dim, is_brnn=False)[0]
    act = per_word(a_list)[filtr][valid]; del a_list
    s_list = dl.load_activations(f"activations_gpt2_{key}_sum.json", dim, is_brnn=False)[0]
    emb_sum = per_word(s_list)[filtr][valid]; del s_list
    word_sfl = np.asarray(df_noinit[[scol, "logfreq", "length"]], float)[filtr][valid]
    text = df_noinit["word"].astype(str).to_numpy()[filtr][valid]
    context = df_noinit["sentence"].astype(str).to_numpy()[filtr][valid]
    covars = {"surprisal": word_sfl[:, 0], "logfreq": word_sfl[:, 1], "length": word_sfl[:, 2]}
    return dict(name="n400", level="word", y=y, groups=None, text=text, context=context,
                act=act, emb_sum=emb_sum, word_sfl=word_sfl, wsi=np.arange(len(y)),
                n_units=len(y), D=dim, covars=covars, unit="N400 amplitude")


LOADERS = {"pereira": load_pereira, "control": load_control, "n400": load_n400}


# --------------------------------------------------------------------- analysis
def fmt(s, n=70):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def analyze(D):
    name, key, layer, dim = D["name"], LM[0], LM[1], LM[2]
    y, groups = D["y"], D["groups"]
    preds = {"act": oof_linear(D["act"], y, groups),
             "emb_sum": oof_linear(D["emb_sum"], y, groups)}
    if D["level"] == "sentence":
        preds["sfl_rff"] = oof_sfl_rff_sentence(D["word_sfl"], D["wsi"], y, D["n_units"], groups, dim)
    else:
        preds["sfl_rff"] = oof_sfl_rff_word(D["word_sfl"], y, dim)

    # predictions CSV
    out = {"text": D["text"], "y": y}
    if D["context"] is not None:
        out["context"] = D["context"]
    for p in PREDS:
        out[f"pred_{p}"] = preds[p]
    df = pd.DataFrame(out)
    df.to_csv(f"{OUTDIR}/{name}_{key}_predictions.csv", index=False)

    L = [f"# Predicted-response extremes — {name} (GPT-2 {key}, D={dim})",
         f"n items = {len(y)}; level = {D['level']}; response unit = {D['unit']}\n"]

    # how well each predicted vector tracks the actual response (sanity)
    L.append("## Predicted vs. actual response (Pearson r)")
    for p in PREDS:
        L.append(f"- {p}: r = {pearsonr(preds[p], y)[0]:.3f}")

    # agreement between predictors
    L.append("\n## Do the predictors agree? (correlation of predicted values across items)")
    L.append("Spearman (lower triangle) / Pearson (upper):")
    L.append("```")
    L.append("           " + "".join(f"{p:>10}" for p in PREDS))
    for i, a in enumerate(PREDS):
        row = [f"{a:>10}"]
        for j, b in enumerate(PREDS):
            if i == j:
                row.append(f"{'-':>10}")
            elif j > i:
                row.append(f"{pearsonr(preds[a], preds[b])[0]:>10.3f}")
            else:
                row.append(f"{spearmanr(preds[a], preds[b])[0]:>10.3f}")
        L.append("".join(row))
    L.append("```")

    # what does each predictor track? (correlation with SFL covariates)
    L.append("\n## What does each predicted vector track? (Pearson r with item covariates)")
    L.append("```")
    cov = D["covars"]
    cn = [c for c in ("surprisal", "length", "logfreq") if c in cov]
    L.append("           " + "".join(f"{c:>11}" for c in cn))
    for p in PREDS:
        row = [f"{p:>10}"] + [f"{pearsonr(preds[p], cov[c])[0]:>11.3f}" for c in cn]
        L.append("".join(row))
    L.append("```")

    # top/bottom-10 set overlap between predictors
    def topset(p, high):
        order = np.argsort(preds[p])
        return set((order[-N_TOP:] if high else order[:N_TOP]).tolist())
    for high, lab in [(True, "TOP"), (False, "BOTTOM")]:
        L.append(f"\n## {lab}-{N_TOP} set overlap (count shared items)")
        L.append("```")
        L.append("           " + "".join(f"{p:>10}" for p in PREDS))
        for a in PREDS:
            row = [f"{a:>10}"]
            for b in PREDS:
                row.append(f"{len(topset(a, high) & topset(b, high)):>10}")
            L.append("".join(row))
        L.append("```")

    # the actual lists
    for p in PREDS:
        order = np.argsort(preds[p])
        for high, lab in [(True, "HIGHEST"), (False, "LOWEST")]:
            idx = order[::-1][:N_TOP] if high else order[:N_TOP]
            L.append(f"\n### {p} — {lab} {N_TOP} predicted")
            for rank, ii in enumerate(idx, 1):
                ctx = f"   | ctx: {fmt(D['context'][ii], 60)}" if D["context"] is not None else ""
                txt = fmt(D["text"][ii])
                L.append(f'{rank:>2}. pred={preds[p][ii]:+.3f}  y={y[ii]:+.3f}  "{txt}"{ctx}')

    report = "\n".join(L)
    with open(f"{OUTDIR}/{name}_{key}_report.md", "w") as h:
        h.write(report + "\n")
    print(report)
    print(f"\nWROTE {OUTDIR}/{name}_{key}_report.md  and  _{key}_predictions.csv")


def main():
    makedirs(OUTDIR, exist_ok=True)
    which = sys.argv[1:] or ["pereira", "control", "n400"]
    for name in which:
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
        D = LOADERS[name](*LM)
        analyze(D)


if __name__ == "__main__":
    main()
