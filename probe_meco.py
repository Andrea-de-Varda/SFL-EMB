#!/usr/bin/env python3
"""Meco-en eye-tracking (A1 first-pass + A3 first-fixation / total-fixation).

Word-level, no spillover. Sentences are reconstructed from the behavioral data
(the English MeCo texts); each text's words in reading order INCLUDE the
sentence-initial word (for left context) and it is dropped when aligning features
to the behavioral rows. Unfixated words -> 0 before averaging over participants.

Env vars: SFL_MODELS (comma-sep subset, default all), SFL_BACKEND (himalaya|sklearn).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT
from src import behavioral
from src.models import MODEL_ORDER

MODELS_TO_RUN = os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CACHE, RESULTS = PROJECT_ROOT / "cache", PROJECT_ROOT / "results"

MEASURES = {"fpass": "firstrun.dur", "ffix": "firstfix.dur", "totfix": "dur"}

meco = pd.read_csv(PROJECT_ROOT / "data/Meco/Meco_l1.csv", index_col=None).iloc[:, 1:]
meco = meco.dropna(subset=["ia"])
meco = meco[~meco["subid"].isin(["ee_09", "ee_22", "ru_8"])]   # wrong itemid assignment
meco = meco[meco["lang"] == "en"].copy()
meco["ianum"] = meco["ianum"].astype(int)
for col in MEASURES.values():
    meco[col] = meco[col].fillna(0)                            # unfixated -> 0
meco["uniq"] = meco["trialid"].astype(str) + "-" + meco["ianum"].astype(str)

# word string per (text, position) and participant-mean of each measure per word
wordmap = meco.groupby(["trialid", "ianum"])["ia"].first()
agg = meco.groupby("uniq")[list(MEASURES.values())].mean()

sentences, targ = [], {t: [] for t in MEASURES}
for trialid in sorted(meco["trialid"].unique()):
    ianums = sorted(meco[meco["trialid"] == trialid]["ianum"].unique())
    sentences.append([wordmap[(trialid, ia)] for ia in ianums])   # incl. first word
    for ia in ianums[1:]:                                          # drop sentence-initial
        row = agg.loc[f"{trialid}-{ia}"]
        for t, col in MEASURES.items():
            targ[t].append(row[col])

TARGETS = {t: np.array(v, dtype=float) for t, v in targ.items()}
print(f"Meco-en: {len(TARGETS['fpass'])} words across {len(sentences)} texts")

behavioral.run_word_dataset("meco", sentences, TARGETS, MODELS_TO_RUN,
                            results_dir=str(RESULTS), cache_dir=str(CACHE), backend=BACKEND)
