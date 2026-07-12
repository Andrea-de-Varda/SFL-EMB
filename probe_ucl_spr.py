#!/usr/bin/env python3
"""UCL self-paced reading (A1). Word-level with i, i-1, i-2 spillover on SFL & EMB.

Uses only the 205 sentences shared with UCL_ET (already the content of
UCL_reading_all.csv). Target is the per-word SPR reading time (`RT`).

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
sentences, target, sent_ids = [], [], []
for i, sn in enumerate(sorted(d["sent_nr"].unique())):
    s = d[d["sent_nr"] == sn].sort_values("word_pos")
    sentences.append(list(s["word"]))
    ni = s.iloc[1:]                                    # non-initial words
    target.extend(ni["RT"].to_numpy())
    sent_ids.extend([i] * len(ni))

target = np.array(target, dtype=float)
print(f"UCL_SPR: {len(target)} non-initial words across {len(sentences)} sentences")

behavioral.run_spill_dataset("ucl_spr", sentences, target, sent_ids, MODELS_TO_RUN,
                             results_dir=str(RESULTS), cache_dir=str(CACHE), backend=BACKEND)
