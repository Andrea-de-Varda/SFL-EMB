#!/usr/bin/env python3
"""Interpretability: which items are PREDICTED to elicit the highest / lowest brain
or ERP response under each encoding predictor, and do the predictors agree?

  act      EMB (D), mean-over-sub-word pooling            (the embedding readout)
  emb_sum  EMB (D), sum pooling (over words / sub-words)
  sfl_rff  SFL(3) -> RBFSampler(D)                         (dimensionality-matched SFL)

For each dataset (Pereira, Tuckute/control, UCL-N400) and GPT-2 XL (D=1600) we
compute leakage-free out-of-fold predictions with the same folds / RFF as the main
pipeline (src/fit, src/rff), then list the top/bottom-N predicted items and quantify
agreement (Spearman/Pearson between predicted vectors, top/bottom-N overlap, and
correlation of each predicted vector with item length / surprisal / log-frequency).

Self-contained: embeddings + surprisal are recomputed ex novo. Writes
results_extremes/<dataset>_xl_{report.md,predictions.csv}.

Usage:  python inspect_predicted_extremes.py [pereira|control|n400 ...]
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT
from src import extract, fit, rff, behavioral
from src.brain import _wsi
from src.models import MODELS

MODEL = os.environ.get("SFL_MODEL", "gpt2_xl")   # paper uses GPT-2 XL
D = MODELS[MODEL]["dim"]
PREDS = ["act", "emb_sum", "sfl_rff"]
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
OUTDIR = PROJECT_ROOT / "results_extremes"
CACHE = PROJECT_ROOT / "cache"
DATA = PROJECT_ROOT / "data"
N_TOP = 10


# ----------------------------------------------------------------- data loaders
def load_pereira():
    m = pd.read_csv(DATA / "pereira/pereira_averaged_with_passage.csv")
    return dict(name="pereira", level="sentence", unit="EffectSize",
                sentences=[str(s).split() for s in m["Sentence"]],
                y=m["EffectSize"].to_numpy(float), groups=m["passage_group"].to_numpy(),
                text=m["Sentence"].astype(str).to_numpy(), context=None)


def load_control():
    ctl = pd.read_csv(DATA / "tuckute/brain-lang-data_participant_20230728.csv")
    g = ctl[ctl["roi"] == "lang_LH_netw"].groupby("sentence", sort=True)["response_target"].mean()
    return dict(name="control", level="sentence", unit="BOLD (lang LH)",
                sentences=[str(s).split() for s in g.index], y=g.to_numpy(float),
                groups=None, text=g.index.astype(str).to_numpy(), context=None)


def load_n400():
    d = pd.read_csv(DATA / "UCL_reading_all.csv")
    sents, n400, words, ctx = [], [], [], []
    for sn in sorted(d["sent_nr"].unique()):
        s = d[d["sent_nr"] == sn].sort_values("word_pos")
        sents.append(list(s["word"]))
        ni = s.iloc[1:]
        n400.extend(ni["N400"].to_numpy())
        words.extend(ni["word"].astype(str))
        ctx.extend(ni["sentence"].astype(str))
    n400 = np.array(n400, float)
    mask = np.array([("." not in w) and ("," not in w) for w in words]) & np.isfinite(n400)
    return dict(name="n400", level="word", unit="N400 amplitude", sentences=sents,
                y=n400, groups=None, text=np.array(words), context=np.array(ctx), row_mask=mask)


LOADERS = {"pereira": load_pereira, "control": load_control, "n400": load_n400}


# ----------------------------------------------------------------- feature build
def prepare(D_):
    """Extract embeddings/SFL ex novo and build act / emb_sum / sfl_rff inputs."""
    sents = D_["sentences"]
    wsi = _wsi(sents)
    freqs, lens = behavioral.freq_length(sents)
    emb_mean, surp = extract.extract_words(MODEL, sents, drop_first=True,
                                           cache=CACHE / f"extremes_{D_['name']}_mean.npz")
    word_sfl = np.column_stack([surp, freqs, lens])
    n_units = len(D_["y"]) if D_["level"] == "word" and "row_mask" not in D_ else None

    if D_["level"] == "sentence":
        units = np.arange(len(sents))
        act = rff.pool_words_to_units(emb_mean, wsi, units, "mean")
        emb_sum = rff.pool_words_to_units(emb_mean, wsi, units, "sum")
        covars = {c: rff.pool_words_to_units(v[:, None], wsi, units, "mean").ravel()
                  for c, v in zip(("surprisal", "length", "logfreq"),
                                  (surp, lens, freqs))}
        rff_build = rff.sentence_rff_build_fn(word_sfl, wsi, units, D)
        return dict(act=act, emb_sum=emb_sum, covars=covars, rff_build=rff_build,
                    word_sfl=None)
    # word level (n400): unit == word; emb_sum = sum-over-sub-words
    emb_sumw, _ = extract.extract_words(MODEL, sents, drop_first=True, pool="sum",
                                        cache=CACHE / f"extremes_{D_['name']}_sum.npz")
    mask = D_.get("row_mask", np.ones(len(word_sfl), bool))
    covars = {"surprisal": surp[mask], "length": lens[mask], "logfreq": freqs[mask]}
    return dict(act=emb_mean[mask], emb_sum=emb_sumw[mask], covars=covars,
                rff_build=rff.rff_build_fn(word_sfl[mask], D), word_sfl=word_sfl[mask])


# --------------------------------------------------------------------- analysis
def fmt(s, n=70):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def analyze(D_):
    F = prepare(D_)
    # apply the same row mask to the metadata for word-level datasets
    if D_["level"] == "word":
        m = D_.get("row_mask", np.ones(len(D_["y"]), bool))
        y, text = D_["y"][m], D_["text"][m]
        context = D_["context"][m] if D_["context"] is not None else None
    else:
        y, text, context = D_["y"], D_["text"], D_["context"]
    folds = fit.make_folds(len(y), groups=D_["groups"], n_splits=10)

    preds = {"act": fit.oof_predict(y, folds, X=F["act"], backend=BACKEND),
             "emb_sum": fit.oof_predict(y, folds, X=F["emb_sum"], backend=BACKEND),
             "sfl_rff": fit.oof_predict(y, folds, build_fn=F["rff_build"], backend=BACKEND)}

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {"text": text, "y": y}
    if context is not None:
        out["context"] = context
    for p in PREDS:
        out[f"pred_{p}"] = preds[p]
    pd.DataFrame(out).to_csv(OUTDIR / f"{D_['name']}_xl_predictions.csv", index=False)

    L = [f"# Predicted-response extremes — {D_['name']} ({MODEL}, D={D})",
         f"n items = {len(y)}; level = {D_['level']}; response = {D_['unit']}\n",
         "## Predicted vs. actual response (Pearson r)"]
    for p in PREDS:
        L.append(f"- {p}: r = {pearsonr(preds[p], y)[0]:.3f}")

    L.append("\n## Do the predictors agree? Spearman (lower) / Pearson (upper)")
    L.append("```")
    L.append("          " + "".join(f"{p:>10}" for p in PREDS))
    for i, a in enumerate(PREDS):
        row = [f"{a:>10}"]
        for j, b in enumerate(PREDS):
            row.append(f"{'-':>10}" if i == j else
                       (f"{pearsonr(preds[a], preds[b])[0]:>10.3f}" if j > i else
                        f"{spearmanr(preds[a], preds[b])[0]:>10.3f}"))
        L.append("".join(row))
    L.append("```")

    L.append("\n## What does each predicted vector track? (Pearson r with covariates)")
    L.append("```")
    cn = ["surprisal", "length", "logfreq"]
    L.append("          " + "".join(f"{c:>11}" for c in cn))
    for p in PREDS:
        L.append(f"{p:>10}" + "".join(f"{pearsonr(preds[p], F['covars'][c])[0]:>11.3f}" for c in cn))
    L.append("```")

    def topset(p, high):
        o = np.argsort(preds[p])
        return set((o[-N_TOP:] if high else o[:N_TOP]).tolist())
    for high, lab in [(True, "TOP"), (False, "BOTTOM")]:
        L.append(f"\n## {lab}-{N_TOP} set overlap (shared items)")
        L.append("```")
        L.append("          " + "".join(f"{p:>10}" for p in PREDS))
        for a in PREDS:
            L.append(f"{a:>10}" + "".join(f"{len(topset(a, high) & topset(b, high)):>10}" for b in PREDS))
        L.append("```")

    for p in PREDS:
        o = np.argsort(preds[p])
        for high, lab in [(True, "HIGHEST"), (False, "LOWEST")]:
            idx = o[::-1][:N_TOP] if high else o[:N_TOP]
            L.append(f"\n### {p} — {lab} {N_TOP} predicted")
            for rank, ii in enumerate(idx, 1):
                c = f"   | ctx: {fmt(context[ii], 60)}" if context is not None else ""
                L.append(f'{rank:>2}. pred={preds[p][ii]:+.3f}  y={y[ii]:+.3f}  "{fmt(text[ii])}"{c}')

    report = "\n".join(L)
    (OUTDIR / f"{D_['name']}_xl_report.md").write_text(report + "\n")
    print(report)
    print(f"\nWROTE {OUTDIR}/{D_['name']}_xl_report.md  and  _xl_predictions.csv")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["pereira", "control", "n400"]):
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        analyze(LOADERS[name]())
