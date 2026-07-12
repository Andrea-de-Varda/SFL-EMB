#!/usr/bin/env python3
"""Tuckute2024 sentence-level fMRI (A1). 10-fold sequential CV.

The response (`response_target`) is averaged over sentences within the LH language
network (roi == "lang_LH_netw"). ~2000 sentences.

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
RESULTS = PROJECT_ROOT / "control_fmri/results"

df = pd.read_csv(PROJECT_ROOT / "control_fmri/data/brain-lang-data_participant_20230728.csv")
df = df[df["roi"] == "lang_LH_netw"]
g = df.groupby("sentence", sort=True)["response_target"].mean()   # one response per sentence
sentences = [str(s).split() for s in g.index]
y = g.to_numpy()
print(f"Tuckute2024: {len(sentences)} sentences")

brain.run_sentence_dataset("control", sentences, y, MODELS_TO_RUN,
                           results_dir=str(RESULTS), cache_dir=str(CACHE), backend=BACKEND)
