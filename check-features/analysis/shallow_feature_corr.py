#!/usr/bin/env python3
"""
POINTS 2-3: correlation between the PREDICTED brain/ERP response (EMB, SFL+RFF;
GPT-2-XL) and shallow / semantic features.
  Pereira  -> sentence length
  Tuckute  -> the 14 semantic+form ratings (rated subset)
  UCL-N400 -> surprisal / log-frequency / length
Uses only the saved XL predictions (results_extremes/*_xl_predictions.csv) + the
feature CSVs -> no activations needed. Writes SHALLOW_FEATURE_CORR.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RES = ROOT / "results_extremes"
OUT = []


def w(s=""):
    print(s); OUT.append(s)


def r(a, b):
    return pearsonr(a, b)[0]


def main():
    # ---- Pereira: sentence length ----
    p = pd.read_csv(RES / "pereira_xl_predictions.csv")
    L = p.text.map(lambda s: len(str(s).split()))
    w("## Pereira — corr(prediction, sentence length)   n=%d" % len(p))
    w("| predictor | r with length |\n|---|---|")
    w(f"| EMB | {r(p.pred_act, L):+.3f} |")
    w(f"| SFL+RFF | {r(p.pred_sfl_rff, L):+.3f} |")

    # ---- Tuckute: 14 ratings ----
    c = pd.read_csv(RES / "control_xl_predictions.csv")
    feat = (pd.read_csv(ROOT / "check-features/features_semantics.csv")
            .merge(pd.read_csv(ROOT / "check-features/features_syntax.csv"), on="sentence"))
    cf = c.merge(feat, left_on="text", right_on="sentence", how="inner")
    SEM = ["rating_imageability_mean", "rating_physical_mean", "rating_places_mean",
           "rating_others_thoughts_mean", "rating_valence_mean", "rating_arousal_mean",
           "rating_sense_mean"]
    FORM = ["avg_word_length", "avg_word_freq_zipf", "rating_frequency_mean",
            "rating_conversational_mean", "rating_gram_mean", "log-prob-pcfg_mean", "dep_length_mean"]
    w("\n## Tuckute — corr(prediction, feature)   rated subset n=%d" % len(cf))
    w("| feature | group | EMB | SFL+RFF |\n|---|---|---|---|")
    for grp, cols in [("semantic", SEM), ("form", FORM)]:
        for f in cols:
            w(f"| {f} | {grp} | {r(cf.pred_act, cf[f]):+.3f} | "
              f"{r(cf.pred_sfl_rff, cf[f]):+.3f} |")

    # ---- N400: surprisal / logfreq / length ----
    n = pd.read_csv(RES / "n400_xl_predictions.csv")
    da = pd.read_csv(ROOT / "data/UCL_reading_all.csv")
    dni = da.dropna(subset=["s_GPT2_small"])
    filtr = np.array([("." not in x) and ("," not in x) for x in dni["word"]])
    yv = np.asarray(dni["N400"], float)[filtr]
    valid = ~np.isnan(yv)
    cov = dni[["s_GPT2_xl", "logfreq", "length"]].to_numpy(float)[filtr][valid]
    assert len(cov) == len(n), (len(cov), len(n))
    w("\n## UCL-N400 — corr(prediction, covariate)   n=%d" % len(n))
    w("| covariate | EMB | SFL+RFF |\n|---|---|---|")
    for nm, v in [("surprisal", cov[:, 0]), ("log-frequency", cov[:, 1]), ("length", cov[:, 2])]:
        w(f"| {nm} | {r(n.pred_act, v):+.3f} | {r(n.pred_sfl_rff, v):+.3f} |")

    (HERE / "SHALLOW_FEATURE_CORR.md").write_text(
        "# Predictions vs. shallow / semantic features (GPT-2-XL)\n\n" + "\n".join(OUT) + "\n")
    print(f"\nwrote {HERE/'SHALLOW_FEATURE_CORR.md'}")


if __name__ == "__main__":
    main()
