#!/usr/bin/env python3
"""UCL_N400 EEG (brain). Word-level, no spillover.

Target is the precomputed N400 amplitude (mean over 12 centro-parietal electrodes,
300-500 ms, artefact-excluded, participant-averaged) from UCL_reading_all.csv.
Following the original analysis, words adjacent to punctuation (containing "." or
",") are dropped for wrap-up effects, along with the sentence-initial word and any
NaN-N400 rows.

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

d = pd.read_csv(PROJECT_ROOT / "UCL_reading_all.csv")
sentences, n400, words_ni = [], [], []
for sn in sorted(d["sent_nr"].unique()):
    s = d[d["sent_nr"] == sn].sort_values("word_pos")
    sentences.append(list(s["word"]))
    ni = s.iloc[1:]                                   # non-initial words
    n400.extend(ni["N400"].to_numpy())
    words_ni.extend(ni["word"].astype(str))

n400 = np.array(n400, dtype=float)
# drop punctuation-adjacent (wrap-up) words and NaN-N400 rows
mask = np.array([("." not in w) and ("," not in w) for w in words_ni]) & np.isfinite(n400)
print(f"UCL_N400: {int(mask.sum())} words across {len(sentences)} sentences")

behavioral.run_word_dataset("ucl_n400", sentences, {"n400": n400}, MODELS_TO_RUN,
                            results_dir=str(RESULTS), cache_dir=str(CACHE),
                            backend=BACKEND, row_mask=mask)
