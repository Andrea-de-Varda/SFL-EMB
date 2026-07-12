#!/usr/bin/env python3
"""UCL eye-tracking (A1 first-pass; + first-fixation / go-past for A3).

Word-level, no spillover. Sentences and targets come from UCL_reading_all.csv
(built once from the UCL corpus + Frank ERP stimuli). Each sentence's word list
includes the sentence-initial word (for context); it is dropped when aligning
features to the behavioral rows.

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

# fpass = first-pass gaze (A1 target); ffix/gopast = A3 extra fixation measures
MEASURES = {"fpass": "RTfirstpass", "ffix": "RTfirstfix", "totfix": "RTgopast"}

d = pd.read_csv(PROJECT_ROOT / "UCL_reading_all.csv")
sentences, targ = [], {t: [] for t in MEASURES}
for sn in sorted(d["sent_nr"].unique()):
    s = d[d["sent_nr"] == sn].sort_values("word_pos")
    sentences.append(list(s["word"]))                 # incl. sentence-initial word
    ni = s.iloc[1:]                                    # drop sentence-initial
    for t, col in MEASURES.items():
        targ[t].extend(ni[col].to_numpy())

TARGETS = {t: np.array(v, dtype=float) for t, v in targ.items()}
print(f"UCL_ET: {len(TARGETS['fpass'])} words across {len(sentences)} sentences")

behavioral.run_word_dataset("ucl_et", sentences, TARGETS, MODELS_TO_RUN,
                            results_dir=str(RESULTS), cache_dir=str(CACHE), backend=BACKEND)
