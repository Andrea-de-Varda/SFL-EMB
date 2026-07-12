#!/usr/bin/env python3
"""Brown self-paced reading (A1). Word-level with i, i-1, i-2 spillover on both
SFL and EMB.

The per-word reading time (`psychometric`, already participant-aggregated) is read
from the cached data/brown_surprisal.csv, whose rows are exactly the non-initial
words of corpora/bnc_brown.txt in order; embeddings/surprisal are recomputed from
those sentences. Spillover and the 5-condition fit are handled by the shared driver.

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

sentences = [ln.split() for ln in
             open(PROJECT_ROOT / "data/corpora/bnc_brown.txt").read().splitlines()]
target = pd.read_csv(PROJECT_ROOT / "data/brown_surprisal.csv")["psychometric"].to_numpy()
sent_ids = [s for s, sent in enumerate(sentences) for _ in sent[1:]]  # non-initial words
print(f"Brown: {len(target)} non-initial words across {len(sentences)} sentences")

behavioral.run_spill_dataset("brown", sentences, target, sent_ids, MODELS_TO_RUN,
                             results_dir=str(RESULTS), cache_dir=str(CACHE), backend=BACKEND)
