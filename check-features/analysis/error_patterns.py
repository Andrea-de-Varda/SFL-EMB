#!/usr/bin/env python3
"""
Error / difference patterns between EMB and SFL+RFF on the brain over/under-prediction
runs. Both reach similar predictivity but rank items differently -- WHERE and on WHAT
kind of item do they diverge?

For each dataset we score every item by the disagreement
    d = z(EMB_sum prediction) - z(SFL+RFF prediction)
Large positive d = EMB predicts high response where SFL+RFF does not (EMB-favored);
large negative d = SFL+RFF-favored. We print the extreme items of each and quantify
what distinguishes them (length; and for Tuckute, the human semantic/form ratings).

Run:  python check-features/analysis/error_patterns.py   (writes ERROR_PATTERNS.md)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEM = ["rating_imageability_mean", "rating_physical_mean", "rating_places_mean",
       "rating_others_thoughts_mean", "rating_valence_mean", "rating_arousal_mean",
       "rating_sense_mean"]
FORM = ["avg_word_length", "avg_word_freq_zipf", "rating_gram_mean", "dep_length_mean"]
OUT = []


def w(s=""):
    print(s); OUT.append(s)


def wc(s):                       # word count of a string
    return len(str(s).split())


def analyze(key, name, ctx_col=None):
    c = pd.read_csv(ROOT / "results_extremes" / f"{key}_xl_predictions.csv")
    c["d"] = zscore(c["pred_emb_sum"]) - zscore(c["pred_sfl_rff"])
    c["nwords"] = c["text"].map(wc)
    c["nchars"] = c["text"].str.len()
    q = 0.15
    hi = c[c.d >= c.d.quantile(1 - q)]           # EMB-favored
    lo = c[c.d <= c.d.quantile(q)]               # SFL+RFF-favored
    w(f"\n{'='*78}\n{name}  (n={len(c)}, unit={'word' if ctx_col else 'sentence'})\n{'='*78}")
    w(f"predictivity: EMB_sum r={pearsonr(c.pred_emb_sum, c.y)[0]:.3f}  "
      f"SFL+RFF r={pearsonr(c.pred_sfl_rff, c.y)[0]:.3f}  "
      f"agreement Spearman={pd.Series(c.pred_emb_sum).corr(c.pred_sfl_rff, 'spearman'):.3f}")
    # what surface variable predicts the disagreement?
    unit = "word length (chars)" if ctx_col else "sentence length (words)"
    lenvar = c["nchars"] if ctx_col else c["nwords"]
    w(f"corr(disagreement d, {unit}) = {pearsonr(c.d, lenvar)[0]:+.3f}   "
      f"[EMB-favored {'len' if ctx_col else 'wc'}={ (lo['nchars'] if ctx_col else hi['nwords']).mean():.1f}"
      f" ... ]")
    w(f"  mean {unit}:  EMB-favored={ (hi['nchars'] if ctx_col else hi['nwords']).mean():.2f}"
      f"   SFL+RFF-favored={ (lo['nchars'] if ctx_col else lo['nwords']).mean():.2f}")

    # Tuckute: attach the human ratings and compare the two groups
    if key == "control":
        sem = pd.read_csv(ROOT / "check-features/features_semantics.csv")
        syn = pd.read_csv(ROOT / "check-features/features_syntax.csv")
        feat = sem.merge(syn, on="sentence")
        cf = c.merge(feat, left_on="text", right_on="sentence", how="inner")
        cf_hi = cf[cf.d >= cf.d.quantile(1 - q)]; cf_lo = cf[cf.d <= cf.d.quantile(q)]
        w(f"\n  rated-sentence subset n={len(cf)}. Group means (EMB-favored vs SFL+RFF-favored):")
        w(f"  {'feature':26s}{'EMB-fav':>9}{'RFF-fav':>9}{'Δ(EMB-RFF)':>12}")
        for f in SEM + FORM:
            a, b = cf_hi[f].mean(), cf_lo[f].mean()
            tag = "  <-- semantic" if f in SEM else ""
            w(f"  {f:26s}{a:>9.2f}{b:>9.2f}{a-b:>12.2f}{tag}")

    # the actual items
    def show(sub, title, n=8):
        w(f"\n  {title}:")
        for _, r in sub.sort_values("d", ascending=(title.startswith("SFL")))[:n].iterrows():
            extra = f"  | {str(r[ctx_col])[:64]}" if ctx_col else ""
            w(f"    d={r.d:+.2f} y={r.y:+.2f}  \"{str(r.text)[:66]}\"{extra}")
    show(hi, "EMB-favored (EMB predicts high, SFL+RFF low)")
    show(lo, "SFL+RFF-favored (SFL+RFF predicts high, EMB low)")


def main():
    analyze("control", "TUCKUTE (control fMRI, lang-LH BOLD)")
    analyze("pereira", "PEREIRA (fMRI EffectSize)")
    analyze("n400", "UCL-N400 (ERP amplitude)", ctx_col="context")
    (HERE / "ERROR_PATTERNS.md").write_text("# Error / difference patterns: EMB vs SFL+RFF\n" +
                                            "\n".join(OUT) + "\n")
    print(f"\nwrote {HERE/'ERROR_PATTERNS.md'}")


if __name__ == "__main__":
    main()
