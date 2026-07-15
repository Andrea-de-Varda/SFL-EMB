#!/usr/bin/env python3
"""Appendix A5: measurement reliability vs sample size (progressive split-half).

For each dataset we take, per item, the vector of measurements across participants
(reading time / ERP / BOLD). At sample size ``n`` we draw two disjoint halves of ``n``
observations per item, correlate the item means across the halves, and average over
random draws -> reliability ``R(n)``. Curves are grouped by modality (eye-tracking,
self-paced reading, N400, Maze, fMRI-sentences, fMRI-stories) and saved to
``figures/reliability/``.

Runs locally (conda ``analysis``). The heavy raw arrays it reads (Wehbe .npy,
NatStor-fMRI timeseries, Tuckute csv, Pereira parquet) are ``.gitignore``d but must be
present under ``data/``. Split-half results are cached so re-plotting is instant.

    python appendix/reliability_and_size.py            # compute (cached) + plot
    python appendix/reliability_and_size.py --replot   # re-plot from cache only
"""
import argparse
import os
import pickle
import random
import re
import sys
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT_ROOT

DATA = PROJECT_ROOT / "data"
CORP = DATA / "corpora"
FIGDIR = PROJECT_ROOT / "figures" / "reliability"
CACHE = PROJECT_ROOT / "cache" / "reliab_size.pkl"


# --------------------------------------------------------------------------- core
def progressive_splithalf(nested_list, n_iterations=10):
    """R(n) for n = 1..max_item_len/2: correlate item means of two disjoint n-halves."""
    shortest = max(len(s) for s in nested_list)
    out = []
    for n_samples in tqdm(range(1, int(shortest / 2) + 1), total=int(shortest / 2)):
        rs = []
        for it in range(n_iterations):
            a, b = [], []
            for sub in nested_list:
                use = min(n_samples, floor(len(sub) / 2))
                random.seed(it)
                random.shuffle(sub)
                h1 = np.mean(sub[:use]) if sub[:use] else np.nan
                h2 = np.mean(sub[use:use * 2]) if sub[use:use * 2] else np.nan
                if not np.isnan(h1) and not np.isnan(h2):
                    a.append(h1)
                    b.append(h2)
            if a and b and not any(np.isnan(a + b)):
                rs.append(pearsonr(a, b)[0])
        if rs:
            out.append([n_samples * 2, np.mean(rs), np.std(rs)])
    return pd.DataFrame(out, columns=["n", "m", "sd"])


def _sent_words(path):
    return [re.sub("\n", "", s).split() for s in open(path).readlines()]


def _create_word_pos(sentences):
    word_pos, sent_nr = [], []
    for snum, sent in enumerate(sentences):
        for tnum, _ in enumerate(sent):
            word_pos.append(tnum + 1)
            sent_nr.append(snum)
    return word_pos, sent_nr


# --------------------------------------------------------------------- per dataset
# Each loader returns a nested list: one sublist per item, holding that item's
# measurements across participants (the input to progressive_splithalf).

def load_natstor_spr():
    df = pd.read_csv(DATA / "NaturalStories/NaturalStories_RAW.tsv", sep="\t")
    df = df[df["correct"] == 6]
    pos = _create_word_pos(_sent_words(CORP / "natural_stories.txt"))
    g = df.groupby(["item", "zone", "word"], as_index=False).agg({"RT": list})
    g["word_pos"], g["sent_nr"] = pos
    return g[g["word_pos"] > 3]["RT"].tolist()


def load_brown():
    df = pd.read_csv(DATA / "Brown/brown_spr_RAW.csv")
    pos = _create_word_pos(_sent_words(CORP / "bnc_brown.txt"))
    g = df.groupby(["text_id", "text_pos", "word"], as_index=False).agg({"time": list})
    g["word_pos"], g["sent_nr"] = pos
    return g[g["word_pos"] > 3]["time"].tolist()


def load_ucl_et():
    eye = pd.read_csv(DATA / "UCL_ET_CORPUS/eyetracking.RT.txt", sep="\t")
    eye = eye[eye.word_pos > 1]                     # exclude sentence-initial word
    g = eye.groupby(["sent_nr", "word_pos"], as_index=False).agg({"RTfirstfix": list})
    return g["RTfirstfix"].tolist(), set(eye["sent_nr"])


def load_ucl_spr(keep_sent_nr):
    spr = pd.read_csv(DATA / "UCL_ET_CORPUS/selfpacedreading.RT.txt", sep="\t")
    spr = spr[spr.sent_nr.isin(keep_sent_nr)]
    spr = spr[spr.word_pos > 3]
    g = spr.groupby(["sent_nr", "word_pos"], as_index=False).agg({"RT": list})
    return g["RT"].tolist()


def load_ucl_n400():
    mat = scipy.io.loadmat(DATA / "ERP_Frank/stimuli_erp.mat")
    out = []
    for idx, sentence in enumerate(mat["sentences"]):
        words = [t[0] for t in sentence[0][0]]
        ERP = mat["ERP"][idx][0]
        artefact = mat["artefact"][idx][0]
        ERP[np.where(artefact == 1)] = np.nan
        for i in range(1, len(words)):              # drop sentence-initial word
            if ("." in words[i]) or ("," in words[i]):
                continue
            out.append(ERP[i][:, 2].tolist())       # channel 2 = N400 window
    return out


def load_provo():
    p = pd.read_csv(DATA / "Provo/Provo_Corpus-Eyetracking_Data.csv")
    p["IA_FIRST_RUN_DWELL_TIME"] = p["IA_FIRST_RUN_DWELL_TIME"].fillna(0)
    g = p.groupby(["Word_Unique_ID", "Word_Number", "Sentence_Number",
                   "Word_In_Sentence_Number", "Word"], as_index=False
                  ).agg({"IA_FIRST_RUN_DWELL_TIME": list})
    return g["IA_FIRST_RUN_DWELL_TIME"].tolist()


def load_maze():
    maze = pyreadr_read(DATA / "NaturalStories/natstor_maze.rds")
    maze = maze[maze["native"].isin(["ENG", "ENGLISH", "English", "english"])]
    sents = _sent_words(CORP / "maze_sentences.txt")
    recover = {" ".join(l): idx for idx, l in enumerate(sents)}
    maze["sent_n"] = maze.sentence.map(recover)
    part_ok = {s: (len(t[t["correct"] == "yes"]) / len(t) > .8)
               for s, t in maze.groupby("subject", observed=True)}
    error, prev_idx, prev_cor = [], 0, True
    for _, row in tqdm(maze.iterrows(), total=len(maze), desc="maze error"):
        cur_idx, cur_cor = row["word_num"], row["correct"] == "yes"
        error.append(0 if ((cur_cor and prev_cor) if cur_idx - prev_idx == 1 else cur_cor) else 1)
        prev_idx, prev_cor = cur_idx, cur_cor
    maze["error"] = error
    maze["is_attentive"] = maze["subject"].map(part_ok)
    maze = maze[(maze["type"] != "practice") & (maze["rt"] < 5000) & (maze["rt"] > 100)
                & (maze["is_attentive"]) & (maze["word_num"] != 0) & (maze["error"] == 0)]
    sent_len = {idx: len(l) for idx, l in enumerate(sents)}
    rt = []
    for sn in tqdm(sorted(set(maze.sent_n.dropna())), desc="maze items"):
        temp = maze[maze.sent_n == sn]
        for n in range(sent_len[int(sn)]):
            row = temp[temp.word_num == n]
            if not row.empty and n != 0:
                rt.append(row["rt"].tolist())
    return [r for r in rt if len(r) > 1]


def load_meco():
    meco = pd.read_csv(DATA / "Meco/Meco_l1.csv", index_col=None).iloc[:, 1:]
    meco = meco[meco["ia"].notna()]
    for c in ("trialid", "itemid", "ianum"):
        meco[c] = meco[c].astype(int)
    meco["unique"] = meco.trialid.astype(str) + "-" + meco.ianum.astype(str) + "-" + meco.lang
    meco = meco[~meco["subid"].isin(["ee_09", "ee_22", "ru_8"])]
    meco["firstrun.dur"] = meco["firstrun.dur"].fillna(0)
    uniques = []
    for textid in sorted(set(meco.trialid)):
        temp = meco[(meco["lang"] == "en") & (meco["trialid"] == textid)]
        ids = []
        for i in range(1, int(max(temp.ianum)) + 1):
            u = list(set(temp[temp.ianum == i].unique))
            if u:
                ids.append(u[0])
        uniques.append(ids)
    rt = []
    for ids in tqdm(uniques, desc="meco"):
        for uniq in ids[1:]:                        # drop sentence-initial item
            temp = meco[meco.unique == uniq]
            rt.append(temp["firstrun.dur"].fillna(0).tolist())
    return rt


def load_zuco():
    out = pd.read_csv(DATA / "ZuCo/ET_data.csv").fillna(0)
    out["unique"] = out["idx"].astype(str) + "-" + out["word_idx"].astype(str)
    out = out[out.word_idx != 0]
    return [out[out.unique == u]["gd"].tolist() for u in tqdm(sorted(set(out["unique"])), desc="zuco")]


def load_pereira():
    df = pd.read_parquet(DATA / "pereira/Pereira.parquet.gzip",
                         columns=["UID", "Sentence", "ROI", "EffectSize"])
    lang = ["Lang_LH_IFGorb", "Lang_LH_IFG", "Lang_LH_MFG", "Lang_LH_AntTemp", "Lang_LH_PostTemp"]
    pereira = df[df.ROI.isin(lang)].copy()
    pereira["ROI"] = pereira["ROI"].cat.remove_unused_categories()
    g = (pereira.groupby(["UID", "ROI", "Sentence"], observed=True).agg({"EffectSize": "mean"})
         .groupby(["UID", "Sentence"], observed=True).agg({"EffectSize": "mean"})
         .groupby(["Sentence"], observed=True).agg({"EffectSize": list}))
    data = g["EffectSize"].tolist()
    return [[x for x in row if not (isinstance(x, float) and np.isnan(x))] for row in data]


def load_wehbe():
    parts = "F H I J K L M N".split()
    arrs = [np.load(DATA / f"wehbe/data_subject_{p}.npy") for p in parts]
    roi = np.load(DATA / "wehbe/HP_subj_roi_inds.npy", allow_pickle=True).tolist()
    n_items = arrs[0].shape[0]
    data = [[] for _ in range(n_items)]
    for arr, name in zip(arrs, parts):
        m = arr[:, roi[name]["all"]].mean(axis=1)
        for i, v in enumerate(m):
            data[i].append(v)
    return data


def load_natstor_fmri():
    ns = pd.read_csv(DATA / "natstor_fmri/NaturalStories_TimeSeries_20220615.csv")
    groups = {
        "kingofbirds": ["StoriesAud_4_kingofbirds_N", "Stories_ToM_4_kingofbirds_N", "Stories_4_kingofbirds_N"],
        "elvis": ["StoriesAud_5_elvis_T", "Stories_ToM_5_elvis_T", "Stories_5_elvis_T", "ISCmega_elvis"],
        "aqua": ["Stories_ToM_2_aqua_T", "Stories_2_aqua_T", "StoriesAud_2_aqua_T"],
        "tulips": ["Stories_9_tulips_T"],
        "boar": ["Stories_ToM_1_boar_T", "Stories_1_boar_T", "StoriesAud_1_boar_T"],
        "tree": ["ISCmega_tree"],
        "highschool": ["Stories_ToM_7_highschool_N", "Stories_7_highschool_N"],
        "matchstickseller": ["Stories_3_matchstickseller_N"],
        "mrsticky": ["Stories_6_mrsticky_N"]}
    lang = ["Lang_LH_AntTemp", "Lang_LH_IFG", "Lang_LH_IFGorb", "Lang_LH_MFG", "Lang_LH_PostTemp"]
    tr_cols = ["T_" + str(n) for n in range(1, 227)]
    elvis_cols = ["T_" + str(n) for n in range(1, 152)]
    data = []
    for grp, stories in groups.items():
        temp = ns[(ns.Story.isin(stories)) & (ns.ROI.isin(lang))].copy().dropna(axis=1, how="all")
        cols = elvis_cols if grp == "elvis" else tr_cols
        ts = np.array([temp[temp.UID == p].filter(cols).mean(axis=0).tolist()
                       for p in temp.UID.unique()])
        data.extend(ts.T.tolist())                  # (items, parts) -> per-item across parts
    return data


def load_control():
    ctl = pd.read_csv(DATA / "tuckute/brain-lang-data_participant_20230728.csv")
    ctl = ctl[ctl["roi"] == "lang_LH_netw"]
    return [ctl[ctl.sentence == s]["response_target"].tolist()
            for s in tqdm(sorted(set(ctl["sentence"])), desc="control")]


def pyreadr_read(path):
    import pyreadr
    return pyreadr.read_r(path)[None]


# --------------------------------------------------------------------- compute all
def compute_all():
    R = {}
    R["provo"] = progressive_splithalf(load_provo())
    R["meco"] = progressive_splithalf(load_meco())
    R["zuco"] = progressive_splithalf(load_zuco())
    eye, keep = load_ucl_et()
    R["eye"] = progressive_splithalf(eye)
    R["spr"] = progressive_splithalf(load_ucl_spr(keep))
    R["natstor"] = progressive_splithalf(load_natstor_spr())
    R["brown"] = progressive_splithalf(load_brown())
    R["maze"] = progressive_splithalf(load_maze())
    R["n400"] = progressive_splithalf(load_ucl_n400())
    R["control"] = progressive_splithalf(load_control())
    R["pereira"] = progressive_splithalf(load_pereira())
    R["natstor_fmri"] = progressive_splithalf(load_natstor_fmri())
    R["hp"] = progressive_splithalf(load_wehbe())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(R, fh)
    print(f"cached split-half results -> {CACHE}")
    return R


# --------------------------------------------------------------------- plotting
def plot_group(fname, dfs, names, title="", themax=0, ylim=None, shifts=None):
    """Paper-style reliability curves: R vs sample size, one labelled line per dataset."""
    plt.figure(figsize=(4, 3), dpi=300)
    for idx, (out_all, name) in enumerate(zip(dfs, names)):
        plt.errorbar(out_all["n"], out_all["m"], yerr=out_all["sd"], fmt="-",
                     capsize=0, linewidth=2, elinewidth=2, markeredgewidth=0, alpha=.8)
        newmax = out_all["n"].max()
        if shifts is not None:
            plt.text(newmax + shifts[idx][0], out_all["m"].iloc[-1] + shifts[idx][1], name, fontsize=12)
        else:
            plt.text(newmax + 0.05 * newmax, out_all["m"].iloc[-1], name, fontsize=12)
            themax = max(themax, newmax)
    plt.xlim(None, max(1, themax + 0.3 * themax))
    plt.ylim(None, ylim)
    plt.xlabel("Sample size")
    plt.ylabel("R")
    plt.title(title, fontsize=18)
    plt.grid(True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        plt.savefig(FIGDIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"wrote figures/reliability/{fname}.svg/.png")


def make_figures(R):
    plot_group("eyetracking", [R["eye"], R["provo"], R["meco"], R["zuco"]],
               ["UCL$_{ET}$", "Provo", "MECO-en", "ZuCo-2"], title="Eye-tracking",
               shifts=[[3, -.05], [3, 0], [-22, +.04], [-19, .05]], themax=90, ylim=1)
    plot_group("spr", [R["natstor"], R["brown"], R["spr"]],
               ["NatStor$_{SPR}$", "Brown", "UCL$_{SPR}$"], title="Self-paced reading",
               themax=90, ylim=0.76)
    plot_group("n400", [R["n400"]], ["UCL$_{N400}$"], title="N400",
               shifts=[[1, .01]], themax=24.5)
    plot_group("maze", [R["maze"]], ["NatStor$_{Maze}$"], title="Maze", themax=9)
    plot_group("fmri_sentences", [R["control"], R["pereira"]], ["Control", "Pereira"],
               title="fMRI, sentences")
    plot_group("fmri_stories", [R["natstor_fmri"], R["hp"]],
               ["NatStor$_{fMRI}$", "Harry Potter"], title="fMRI, stories", themax=84)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="plot from cache; do not recompute")
    args = ap.parse_args()
    if args.replot:
        with open(CACHE, "rb") as fh:
            R = pickle.load(fh)
    else:
        R = compute_all()
    make_figures(R)


if __name__ == "__main__":
    main()
