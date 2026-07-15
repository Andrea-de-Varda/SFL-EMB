#!/usr/bin/env python3
"""Appendix A4: layer sweep (GPT-2 355M).

Design (as reported in the SI):
  * Held-IN datasets pick the best EMB layer: UCL_ET for eye-tracking, UCL_SPR for
    self-paced reading (the latter with i, i-1, i-2 spillover on SFL and EMB, exactly
    as in the main analysis).
  * The winning eye-tracking layer is then evaluated on the held-OUT eye-tracking
    datasets (Provo, Meco-en, ZuCo-2) against the layer-independent SFL baseline.

Reports mean and SD over the 10 outer folds, i.e. the numbers quoted in the SI.
Same estimator/folds as the main pipeline; SFL needs no banded ridge (single block).

Writes appendix/layer_sweep.csv (rows = dataset x layer; cols = emb_r, emb_sd,
sfl_r, sfl_sd) and prints the numbers the SI sentences quote.

    python appendix/layer_sweep.py            # full sweep (GPU)
    python appendix/layer_sweep.py --report   # re-print from the existing csv
"""
import os
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
OUT = PROJECT_ROOT / "appendix/layer_sweep.csv"

warnings.filterwarnings("ignore", message="Solving ridge is slower")


# --------------------------------------------------------------------- loaders
# Each returns (sentences, y, mask_or_sent_ids). Mirrors the matching probe_*.py.
def load_ucl_et():
    d = pd.read_csv(DATA / "UCL_reading_all.csv")
    sents, y = [], []
    for sn in sorted(d["sent_nr"].unique()):
        s = d[d["sent_nr"] == sn].sort_values("word_pos")
        sents.append(list(s["word"]))
        y.extend(s.iloc[1:]["RTfirstpass"].to_numpy())
    return sents, np.array(y, float), None


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
    return sents, np.array(y, float), None


def load_meco():
    """Mirrors probe_meco.py (first-pass = firstrun.dur)."""
    meco = pd.read_csv(DATA / "Meco/Meco_l1.csv", index_col=None).iloc[:, 1:]
    meco = meco.dropna(subset=["ia"])
    meco = meco[~meco["subid"].isin(["ee_09", "ee_22", "ru_8"])]
    meco = meco[meco["lang"] == "en"].copy()
    meco["ianum"] = meco["ianum"].astype(int)
    meco["firstrun.dur"] = meco["firstrun.dur"].fillna(0)
    meco["uniq"] = meco["trialid"].astype(str) + "-" + meco["ianum"].astype(str)
    wordmap = meco.groupby(["trialid", "ianum"])["ia"].first()
    agg = meco.groupby("uniq")["firstrun.dur"].mean()
    sents, y = [], []
    for trialid in sorted(meco["trialid"].unique()):
        ianums = sorted(meco[meco["trialid"] == trialid]["ianum"].unique())
        sents.append([wordmap[(trialid, ia)] for ia in ianums])
        y.extend(agg.loc[f"{trialid}-{ia}"] for ia in ianums[1:])
    return sents, np.array(y, float), None


def load_zuco():
    """Mirrors probe_zuco.py (first-pass = gd); NaN-gaze rows dropped."""
    df = pd.read_csv(DATA / "ZuCo/ET_data_aggregated.csv")
    sents, y = [], []
    for idx in sorted(df["idx"].unique()):
        s = df[df["idx"] == idx].sort_values("word_idx")
        sents.append(list(s["word"]))
        y.extend(s[s["word_idx"] != 0]["gd"].to_numpy())
    y = np.array(y, float)
    return sents, y, ~np.isnan(y)


def load_ucl_spr():
    """Mirrors probe_ucl_spr.py; third value is sent_ids (for spillover)."""
    d = pd.read_csv(DATA / "UCL_reading_all.csv")
    sents, y, sids = [], [], []
    for i, sn in enumerate(sorted(d["sent_nr"].unique())):
        s = d[d["sent_nr"] == sn].sort_values("word_pos")
        sents.append(list(s["word"]))
        ni = s.iloc[1:]
        y.extend(ni["RT"].to_numpy())
        sids.extend([i] * len(ni))
    return sents, np.array(y, float), np.array(sids)


# ---------------------------------------------------------------------- sweeps
def sweep_word(name, loader, rows):
    """Word-level (no spillover) sweep: every layer's EMB vs the SFL baseline."""
    sents, y, mask = loader()
    if mask is None:
        mask = np.ones(len(y), bool)
    freqs, lens = behavioral.freq_length(sents)
    _, surp = extract.extract_words(MODEL, sents, drop_first=True,
                                    cache=PROJECT_ROOT / f"cache/lsweep_{name}_{MODEL}.npz")
    X_sfl = np.column_stack([surp, freqs, lens])[mask]
    y_m = y[mask]
    folds = fit.make_folds(len(y_m), n_splits=10)
    s_r, s_sd, _, _ = fit.score(X=X_sfl, y=y_m, folds=folds, warn_boundary=False)
    layers = extract.extract_all_layers(MODEL, sents, drop_first=True)
    for li, emb in sorted(layers.items()):
        e_r, e_sd, _, _ = fit.score(X=emb[mask], y=y_m, folds=folds, warn_boundary=False)
        rows.append(dict(dataset=name, layer=li, emb_r=e_r, emb_sd=e_sd,
                         sfl_r=s_r, sfl_sd=s_sd))
        print(f"  {name:<8} layer {li:2d}: EMB {e_r:.4f} (SD {e_sd:.4f})   SFL {s_r:.4f}",
              flush=True)


def sweep_spill(name, loader, rows):
    """SPR sweep: SFL and EMB both carry i, i-1, i-2 spillover (as in the main run)."""
    sents, y, sids = loader()
    freqs, lens = behavioral.freq_length(sents)
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
    for li, emb in sorted(layers.items()):
        e_r, e_sd, _, _ = fit.score(X=_spill(emb, keep)[finite], y=y_m, folds=folds,
                                    warn_boundary=False)
        rows.append(dict(dataset=name, layer=li, emb_r=e_r, emb_sd=e_sd,
                         sfl_r=s_r, sfl_sd=s_sd))
        print(f"  {name:<8} layer {li:2d}: EMB {e_r:.4f} (SD {e_sd:.4f})   SFL {s_r:.4f}",
              flush=True)


def report(df):
    et_in = df[df.dataset == "UCL_ET"].set_index("layer")
    best_et = int(et_in["emb_r"].idxmax())
    print(f"\nbest held-in eye-tracking layer (UCL_ET): {best_et} "
          f"(EMB r = {et_in.loc[best_et, 'emb_r']:.4f})")
    if (df.dataset == "UCL_SPR").any():
        spr = df[df.dataset == "UCL_SPR"].set_index("layer")
        print(f"best held-in SPR layer (UCL_SPR): {int(spr['emb_r'].idxmax())} "
              f"(last layer available = {int(spr.index.max())})")
    print(f"\nheld-out eye-tracking datasets at layer {best_et}:")
    for ds in ["Provo", "Meco-en", "ZuCo-2"]:
        sub = df[(df.dataset == ds) & (df.layer == best_et)]
        if not len(sub):
            continue
        r = sub.iloc[0]
        print(f"  {ds:<8} SFL r = {r.sfl_r:.4f} (SD {r.sfl_sd:.4f}); "
              f"EMB r = {r.emb_r:.4f} (SD {r.emb_sd:.4f})  -> "
              f"{'SFL' if r.sfl_r > r.emb_r else 'EMB'} wins")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(pd.read_csv(OUT))
        sys.exit()
    rows = []
    for name, loader in [("UCL_ET", load_ucl_et), ("Provo", load_provo),
                         ("Meco-en", load_meco), ("ZuCo-2", load_zuco)]:
        print(f"=== {name} ===", flush=True)
        sweep_word(name, loader, rows)
    print("=== UCL_SPR (spillover) ===", flush=True)
    sweep_spill("UCL_SPR", load_ucl_spr, rows)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    report(df)
    print(f"\nwrote {OUT}")
