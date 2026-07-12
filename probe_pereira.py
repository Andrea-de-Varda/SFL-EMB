#!/usr/bin/env python3
"""Pereira2018 sentence-level fMRI (A1). Passage-grouped 10-fold CV.

Experiments 2 & 3 only (627 sentences, pre-filtered in the averaged CSV). The
response (EffectSize) is already averaged over the 5 language ROIs / participants.
Cross-validation holds out whole passages (GroupKFold on `passage_group`) so
sentences from the same passage never split across train/test.

Env vars: SFL_MODELS (comma-sep subset, default all), SFL_BACKEND (himalaya|sklearn).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT
from src import brain
from src.models import MODEL_ORDER

MODELS_TO_RUN = os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CACHE = PROJECT_ROOT / "cache"
RESULTS = PROJECT_ROOT / "pereira_1/results"

meta = pd.read_csv(PROJECT_ROOT / "pereira_1/pereira_averaged_with_passage.csv")
sentences = [str(s).split() for s in meta["Sentence"]]
y = meta["EffectSize"].to_numpy()
groups = meta["passage_group"].to_numpy()
assert meta["passage_group"].notna().all()
print(f"Pereira2018: {len(sentences)} sentences, {meta['passage_group'].nunique()} passages")

brain.run_sentence_dataset("pereira", sentences, y, MODELS_TO_RUN,
                           results_dir=str(RESULTS), cache_dir=str(CACHE),
                           groups=groups, backend=BACKEND)
