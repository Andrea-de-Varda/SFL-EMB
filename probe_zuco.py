#!/usr/bin/env python3
"""ZuCo-2 eye-tracking (A1 first-pass + A3 first-fixation / total-fixation).

Word-level, no spillover. The shipped ET_data_aggregated.csv already contains ONLY
the 349 natural-reading sentences (the 390 task-specific ones were dropped upstream)
and is participant-averaged. Targets: fp = gaze duration (`gd`), ff = `ff`,
tf = `tt`; the sentence-initial word (word_idx 0) is dropped and rows whose gaze
duration is NaN are removed uniformly across features and all targets.

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

df = pd.read_csv(PROJECT_ROOT / "main_data/ZuCo/ET_data_aggregated.csv")
MEASURES = {"fpass": "gd", "ffix": "ff", "totfix": "tt"}

sentences, targ = [], {t: [] for t in MEASURES}
for idx in sorted(df["idx"].unique()):
    sent = df[df["idx"] == idx].sort_values("word_idx")
    sentences.append(list(sent["word"]))                    # incl. word_idx 0
    noninit = sent[sent["word_idx"] != 0]                   # drop sentence-initial
    for t, col in MEASURES.items():
        targ[t].extend(noninit[col].to_numpy())

TARGETS = {t: np.array(v, dtype=float) for t, v in targ.items()}
mask = ~np.isnan(TARGETS["fpass"])                          # drop NaN-gaze words uniformly
print(f"ZuCo-2: {int(mask.sum())} words ({len(sentences)} NR sentences)")

behavioral.run_word_dataset("zuco", sentences, TARGETS, MODELS_TO_RUN,
                            results_dir=str(RESULTS), cache_dir=str(CACHE),
                            backend=BACKEND, row_mask=mask)
