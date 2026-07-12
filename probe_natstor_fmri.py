#!/usr/bin/env python3
"""NatStor_fMRI TR-level fMRI (A1). Leave-one-story-out CV (9 stories).

The brain response (already 5-language-ROI / top-voxel / participant averaged and
trimmed) is loaded from response/d_natstor. Word onsets (Whisper `end` times) map
each word to its 2 s TR; per-word features are pooled to TRs and concatenated with
4 lagged TR blocks (BOLD delay), within each story. Embeddings/surprisal are
recomputed from the story text with the shared extractor.

Env vars: SFL_MODELS (comma-sep subset, default all), SFL_BACKEND (himalaya|sklearn).
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT
from src import brain
from src.models import MODEL_ORDER

MODELS_TO_RUN = os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CACHE = PROJECT_ROOT / "cache"
RESULTS = PROJECT_ROOT / "results"
NF = PROJECT_ROOT / "data/natstor_fmri"

with open(NF / "response/d_natstor", "rb") as h:
    d = pickle.load(h)

STORY_NAME = {"1": "boar", "2": "aqua", "3": "matchstickseller", "4": "kingofbirds",
              "5": "elvis", "6": "mrsticky", "7": "highschool", "9": "tulips", "10": "tree"}
STORY_IDS = ["1", "2", "3", "4", "5", "6", "7", "9", "10"]

stories = []
for sid in STORY_IDS:
    y = np.asarray(d[STORY_NAME[sid]], dtype=float)
    n_tr = len(y)
    df = pd.read_csv(NF / "transcribed" / f"{sid}.csv")
    words = [str(w) for w in df["text"]]                 # full text (first kept for context)
    ends = df["end"].to_numpy()[1:]                      # non-initial words (extract drops first)
    time = np.arange(0, n_tr * 2, 2)
    word_tr = np.array([np.where(e > time)[0][-1] if np.any(e > time) else 0 for e in ends])
    stories.append({"words": words, "word_tr": word_tr, "n_tr": n_tr, "y": y})
print(f"NatStor_fMRI: {len(stories)} stories, {sum(s['n_tr'] for s in stories)} TRs")

brain.run_tr_dataset("natstorfmri", stories, MODELS_TO_RUN, results_dir=str(RESULTS),
                     cache_dir=str(CACHE), loso=True, backend=BACKEND)
