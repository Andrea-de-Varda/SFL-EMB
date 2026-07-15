#!/usr/bin/env python3
"""Layer-matched EMB vs SFL: is the layer-wise EMB advantage real, or an artefact of
pinning SFL to the FINAL layer's surprisal?

The A4 layer sweep compares layer-L embeddings against an SFL predictor whose
surprisal always comes from the last layer. That is not a like-for-like comparison:
it lets EMB move up and down the network while SFL is frozen at the top. Here both
predictors are read out at the SAME layer:

    EMB_L   layer-L hidden state (mean-pooled over sub-words), as in the main analyses
    SFL_L   [surprisal_L, log-frequency, length]

where surprisal_L is the **logit lens** readout: layer L's hidden state passed through
the model's own final layer norm and (weight-tied) unembedding
(nostalgebraist 2020; Geva et al. 2022). Frequency and length are layer-independent.

Correctness: GPT2Model already applies ln_f to the last hidden state, so ln_f is
applied only to layers < last. The last layer therefore reproduces the model's true
surprisal exactly -- asserted at runtime (--selftest).

One model (GPT-2 355M), all 13 datasets, every layer. Runs locally on one GPU.

    python appendix/layerwise_logitlens.py --selftest        # validate the lens only
    python appendix/layerwise_logitlens.py                   # all datasets (resumes)
    python appendix/layerwise_logitlens.py --datasets Provo,UCL_SPR
    python appendix/layerwise_logitlens.py --plot            # figures from the csv
"""
import argparse
import os
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT
from src import fit, rff, behavioral
from src.behavioral import _spill
from src.brain import _wsi
from src.extract import Extractor
from src.models import MODELS

MODEL = "gpt2_medium"
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "appendix/layerwise_logitlens.csv"
SURP_CACHE = PROJECT_ROOT / "cache/layerwise_surp"
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")

warnings.filterwarnings("ignore", message="Solving ridge is slower")


# =========================================================================== #
# dataset specs. kind: word | spill | sentence | tr
# =========================================================================== #
def sp_provo():
    p = pd.read_csv(DATA / "Provo/Provo_Corpus-Eyetracking_Data.csv")
    p["IA_FIRST_RUN_DWELL_TIME"] = p["IA_FIRST_RUN_DWELL_TIME"].fillna(0)
    p = p.groupby(["Sentence_Number", "Word_In_Sentence_Number", "Word", "Text_ID"],
                  as_index=False, sort=False).agg(
        {"IA_FIRST_RUN_DWELL_TIME": "mean", "Text_ID": "mean"})
    first = [s.split("\t")[2][1:-2].split()[0]
             for s in open(DATA / "Provo/sentences.txt").readlines()]
    d_first = {n + 1: first[n] for n in range(55)}
    sents, y = [], []
    for t in sorted(set(p["Text_ID"])):
        tmp = p[p["Text_ID"] == t]
        sents.append([d_first[t]] + list(tmp["Word"]))
        y.extend(tmp["IA_FIRST_RUN_DWELL_TIME"])
    return dict(kind="word", sentences=sents, y=np.array(y, float), group="behavior")


def sp_meco():
    m = pd.read_csv(DATA / "Meco/Meco_l1.csv", index_col=None).iloc[:, 1:]
    m = m.dropna(subset=["ia"])
    m = m[~m["subid"].isin(["ee_09", "ee_22", "ru_8"])]
    m = m[m["lang"] == "en"].copy()
    m["ianum"] = m["ianum"].astype(int)
    m["firstrun.dur"] = m["firstrun.dur"].fillna(0)
    m["uniq"] = m["trialid"].astype(str) + "-" + m["ianum"].astype(str)
    wordmap = m.groupby(["trialid", "ianum"])["ia"].first()
    agg = m.groupby("uniq")["firstrun.dur"].mean()
    sents, y = [], []
    for t in sorted(m["trialid"].unique()):
        ia = sorted(m[m["trialid"] == t]["ianum"].unique())
        sents.append([wordmap[(t, i)] for i in ia])
        y.extend(agg.loc[f"{t}-{i}"] for i in ia[1:])
    return dict(kind="word", sentences=sents, y=np.array(y, float), group="behavior")


def sp_zuco():
    df = pd.read_csv(DATA / "ZuCo/ET_data_aggregated.csv")
    sents, y = [], []
    for idx in sorted(df["idx"].unique()):
        s = df[df["idx"] == idx].sort_values("word_idx")
        sents.append(list(s["word"]))
        y.extend(s[s["word_idx"] != 0]["gd"].to_numpy())
    y = np.array(y, float)
    return dict(kind="word", sentences=sents, y=y, mask=~np.isnan(y), group="behavior")


def _ucl(col):
    d = pd.read_csv(DATA / "UCL_reading_all.csv")
    sents, y, sids = [], [], []
    for i, sn in enumerate(sorted(d["sent_nr"].unique())):
        s = d[d["sent_nr"] == sn].sort_values("word_pos")
        sents.append(list(s["word"]))
        ni = s.iloc[1:]
        y.extend(ni[col].to_numpy())
        sids.extend([i] * len(ni))
    return sents, np.array(y, float), np.array(sids)


def sp_ucl_et():
    s, y, _ = _ucl("RTfirstpass")
    return dict(kind="word", sentences=s, y=y, group="behavior")


def sp_ucl_spr():
    s, y, sid = _ucl("RT")
    return dict(kind="spill", sentences=s, y=y, sent_ids=sid, group="behavior")


def sp_ucl_n400():
    d = pd.read_csv(DATA / "UCL_reading_all.csv")
    sents, y, w = [], [], []
    for sn in sorted(d["sent_nr"].unique()):
        s = d[d["sent_nr"] == sn].sort_values("word_pos")
        sents.append(list(s["word"]))
        ni = s.iloc[1:]
        y.extend(ni["N400"].to_numpy())
        w.extend(ni["word"].astype(str))
    y = np.array(y, float)
    mask = np.array([("." not in x) and ("," not in x) for x in w]) & np.isfinite(y)
    return dict(kind="word", sentences=sents, y=y, mask=mask, group="brain")


def sp_brown():
    sents = [ln.split() for ln in (DATA / "corpora/bnc_brown.txt").read_text().splitlines()]
    y = pd.read_csv(DATA / "brown_surprisal.csv")["psychometric"].to_numpy(float)
    sid = np.array([s for s, sent in enumerate(sents) for _ in sent[1:]])
    return dict(kind="spill", sentences=sents, y=y, sent_ids=sid, group="behavior")


def sp_natstor_spr():
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
    sid = np.array([s for s, sent in enumerate(sents) for _ in sent[1:]])
    return dict(kind="spill", sentences=sents, y=y, sent_ids=sid, group="behavior")


def sp_maze():
    import pyreadr
    mz = pyreadr.read_r(DATA / "NaturalStories/natstor_maze.rds")[None]
    mz = mz[mz["native"].isin(['ENG', 'ENGLISH', 'English', 'english'])].copy()
    msent = [ln.split() for ln in
             (DATA / "corpora/maze_sentences.txt").read_text().splitlines()]
    recover = {" ".join(w): i for i, w in enumerate(msent)}
    mz["sent_n"] = mz["sentence"].map(recover)
    att = mz.groupby("subject", observed=True)["correct"].apply(lambda c: (c == "yes").mean() > 0.8)
    err, pi, pc = [], 0, True
    for wn, cor in zip(mz["word_num"].to_numpy(), (mz["correct"] == "yes").to_numpy()):
        err.append((0 if (cor and pc) else 1) if wn - pi == 1 else (0 if cor else 1))
        pi, pc = wn, cor
    mz["error"] = err
    mz["is_attentive"] = mz["subject"].map(att)
    mz = mz[(mz["type"] != "practice") & (mz["rt"] < 5000) & (mz["rt"] > 100)
            & (mz["is_attentive"]) & (mz["word_num"] != 0) & (mz["error"] == 0)]
    mz = mz.dropna(subset=["sent_n"]).copy()
    mz["sent_n"] = mz["sent_n"].astype(int)
    agg = mz.groupby(["sent_n", "word_num"], as_index=False, observed=True).agg({"rt": "mean"})
    rt_map = {(int(r.sent_n), int(r.word_num)): float(r.rt) for r in agg.itertuples()}
    present = sorted(agg["sent_n"].unique())
    sents = [msent[s] for s in present]
    y, mask = [], []
    for s in present:
        for pos in range(1, len(msent[s])):
            y.append(rt_map.get((s, pos), np.nan))
            mask.append((s, pos) in rt_map)
    return dict(kind="word", sentences=sents, y=np.array(y, float),
                mask=np.array(mask), group="behavior")


def sp_pereira():
    m = pd.read_csv(DATA / "pereira/pereira_averaged_with_passage.csv")
    return dict(kind="sentence", sentences=[str(s).split() for s in m["Sentence"]],
                y=m["EffectSize"].to_numpy(float),
                groups=m["passage_group"].to_numpy(), group="brain")


def sp_control():
    df = pd.read_csv(DATA / "tuckute/brain-lang-data_participant_20230728.csv")
    g = df[df["roi"] == "lang_LH_netw"].groupby("sentence", sort=True)["response_target"].mean()
    return dict(kind="sentence", sentences=[str(s).split() for s in g.index],
                y=g.to_numpy(float), groups=None, group="brain")


def sp_wehbe():
    FM = DATA / "wehbe"
    SUBJ = ["F", "H", "I", "J", "K", "L", "M", "N"]
    roi = np.load(FM / "HP_subj_roi_inds.npy", allow_pickle=True).tolist()
    y = np.mean([np.load(FM / f"data_subject_{p}.npy")[:, roi[p]["all"]].mean(axis=1)
                 for p in SUBJ], axis=0)
    runs = np.load(FM / "runs_fmri.npy")
    keep = np.concatenate([np.where(runs == r)[0][20:-15] for r in [1, 2, 3, 4]])
    words = [str(w) for w in np.load(FM / "words_fmri.npy", allow_pickle=True)]
    time = np.load(FM / "time_fmri.npy")
    tw = np.load(FM / "time_words_fmri.npy")[1:]
    word_tr = np.array([np.where(e > time)[0][-1] if np.any(e > time) else 0 for e in tw])
    return dict(kind="tr", loso=False, group="brain",
                stories=[{"words": words, "word_tr": word_tr, "n_tr": len(time),
                          "y": y, "keep": keep}])


def sp_natstor_fmri():
    NF = DATA / "natstor_fmri"
    with open(NF / "response/d_natstor", "rb") as h:
        d = pickle.load(h)
    NAME = {"1": "boar", "2": "aqua", "3": "matchstickseller", "4": "kingofbirds",
            "5": "elvis", "6": "mrsticky", "7": "highschool", "9": "tulips", "10": "tree"}
    stories = []
    for sid in ["1", "2", "3", "4", "5", "6", "7", "9", "10"]:
        y = np.asarray(d[NAME[sid]], float)
        df = pd.read_csv(NF / "transcribed" / f"{sid}.csv")
        words = [str(w) for w in df["text"]]
        ends = df["end"].to_numpy()[1:]
        time = np.arange(0, len(y) * 2, 2)
        wtr = np.array([np.where(e > time)[0][-1] if np.any(e > time) else 0 for e in ends])
        stories.append({"words": words, "word_tr": wtr, "n_tr": len(y), "y": y})
    return dict(kind="tr", loso=True, group="brain", stories=stories)


SPECS = {
    "Provo": sp_provo, "Meco-en": sp_meco, "ZuCo-2": sp_zuco, "UCL_ET": sp_ucl_et,
    "UCL_SPR": sp_ucl_spr, "Brown": sp_brown, "NatStor_SPR": sp_natstor_spr,
    "NatStor_Maze": sp_maze,
    "UCL_N400": sp_ucl_n400, "Pereira2018": sp_pereira, "Tuckute2024": sp_control,
    "NatStor_fMRI": sp_natstor_fmri, "Wehbe2014": sp_wehbe,
}
ORDER = list(SPECS)


# =========================================================================== #
# extraction (all layers: embeddings + logit-lens surprisal)
# =========================================================================== #
def extract_all(ex, sentences, chunked=False):
    """-> ({layer: (N,D) float32}, {layer: (N,) float64}) over non-initial words."""
    E, S = None, None
    for words in sentences:
        w = [str(x) for x in words]
        if chunked:
            ctx = getattr(ex.lm.config, "n_positions", 1024)
            import math
            ns = max(2, math.ceil(len(w) / max(64, ctx // 2)))
            de = ex.chunked_word_features_all(w, n_splits=ns, overlap=50)
            ds = ex.chunked_word_surprisal_all(w, n_splits=ns, overlap=50)
        else:
            de = ex.word_embeddings_all(w)
            ds = ex.word_surprisal_all(w)
        if E is None:
            E = {k: [] for k in de}; S = {k: [] for k in ds}
        for k in de:
            E[k].append(de[k][1:].astype(np.float32))
            S[k].append(ds[k][1:])
    return ({k: np.vstack(v) for k, v in E.items()},
            {k: np.concatenate(v) for k, v in S.items()})


def get_features(name, spec, ex):
    """All-layer word features + the layer-independent freq/length, per dataset."""
    if spec["kind"] == "tr":
        Es, Ss, frs, lns = [], [], [], []
        for st in spec["stories"]:
            e, s = extract_all(ex, [st["words"]], chunked=True)
            f, l = behavioral.freq_length([st["words"]])
            Es.append(e); Ss.append(s); frs.append(f); lns.append(l)
        return Es, Ss, frs, lns
    E, S = extract_all(ex, spec["sentences"], chunked=False)
    fr, ln = behavioral.freq_length(spec["sentences"])
    return E, S, fr, ln


# =========================================================================== #
# per-layer design matrices, mirroring the main pipeline for each geometry
# =========================================================================== #
def run_dataset(name, spec, ex, done):
    E, S, fr, ln = get_features(name, spec, ex)
    rows = []
    kind = spec["kind"]

    if kind in ("word", "spill", "sentence"):
        layers = sorted(E)
    else:
        layers = sorted(E[0])

    for li in layers:
        if (name, li) in done:
            continue
        if kind == "word":
            mask = spec.get("mask", np.ones(len(spec["y"]), bool))
            y = spec["y"][mask]
            folds = fit.make_folds(len(y), n_splits=10)
            Xe = E[li][mask]
            Xs = np.column_stack([S[li], fr, ln])[mask]

        elif kind == "spill":
            sid = spec["sent_ids"]
            keep = np.array([i for i in range(len(spec["y"]))
                             if i >= 2 and sid[i] == sid[i - 1] == sid[i - 2]])
            sfl = _spill(np.column_stack([S[li], fr, ln]), keep)
            yk = spec["y"][keep]
            ok = np.isfinite(yk) & np.isfinite(sfl).all(1)
            y = yk[ok]
            folds = fit.make_folds(len(y), n_splits=10)
            Xe = _spill(E[li], keep)[ok]
            Xs = sfl[ok]

        elif kind == "sentence":
            wsi = _wsi(spec["sentences"])
            units = np.arange(len(spec["sentences"]))
            y = spec["y"]
            folds = fit.make_folds(len(y), groups=spec["groups"], n_splits=10)
            Xe = rff.pool_words_to_units(E[li], wsi, units, "mean")
            Xs = rff.pool_words_to_units(np.column_stack([S[li], fr, ln]), wsi, units, "mean")

        else:  # tr
            ye, ys = [], []
            for si, st in enumerate(spec["stories"]):
                keep = np.asarray(st.get("keep", np.arange(st["n_tr"])), int)
                ye.append(rff.aggregate_to_tr(E[si][li], st["word_tr"], st["n_tr"], "mean")[keep])
                ys.append(rff.aggregate_to_tr(np.column_stack([S[si][li], fr[si], ln[si]]),
                                              st["word_tr"], st["n_tr"], "mean")[keep])
            Xe, Xs = np.vstack(ye), np.vstack(ys)
            y = np.concatenate([np.asarray(s["y"], float) for s in spec["stories"]])
            if spec["loso"]:
                b = np.cumsum([0] + [len(np.asarray(s.get("keep", np.arange(s["n_tr"])), int))
                                     for s in spec["stories"]])
                folds = [np.arange(b[i], b[i + 1]) for i in range(len(spec["stories"]))]
            else:
                folds = fit.make_folds(len(y), n_splits=10)

        r_e = fit.score(X=Xe, y=y, folds=folds, backend=BACKEND, warn_boundary=False)
        r_s = fit.score(X=Xs, y=y, folds=folds, backend=BACKEND, warn_boundary=False)
        rows.append(dict(dataset=name, group=spec["group"], layer=li,
                         emb_r=r_e[0], emb_sd=r_e[1], sfl_r=r_s[0], sfl_sd=r_s[1],
                         n=len(y)))
        print(f"  {name:<13} layer {li:2d}: EMB {r_e[0]:.4f}  SFL_lens {r_s[0]:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(
            OUT, mode="a", header=not OUT.exists(), index=False)
        rows = []
    del E, S


# =========================================================================== #
def selftest():
    ex = Extractor(MODEL)
    s = "The chef cooked the meal in the kitchen yesterday evening".split()
    real = ex.word_surprisal(s)
    lens = ex.word_surprisal_all(s)
    d = np.nanmax(np.abs(real - lens[max(lens)]))
    assert d < 1e-4, f"logit lens != true surprisal at the last layer (max diff {d})"
    print(f"[selftest] last-layer logit lens == model surprisal (max diff {d:.2e})  OK")
    print(f"[selftest] layers returned: {len(lens)} (0..{max(lens)})")
    for li in [0, 1, 12, max(lens)]:
        print(f"           layer {li:2d}: mean surprisal {np.nanmean(lens[li]):7.3f} nats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ORDER))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.plot:
        from layerwise_plot import plot_all
        return plot_all(pd.read_csv(OUT))
    selftest()
    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT)
        done = set(zip(d.dataset, d.layer))
        print(f"resuming: {len(done)} (dataset, layer) cells already done")
    ex = Extractor(MODEL)
    for name in a.datasets.split(","):
        print(f"=== {name} ===", flush=True)
        run_dataset(name, SPECS[name](), ex, done)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
