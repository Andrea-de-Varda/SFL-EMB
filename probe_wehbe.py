#!/usr/bin/env python3
"""Wehbe2014 (Harry Potter) TR-level fMRI (A1). 10-fold sequential CV.

Anatomical (AAL) language ROIs — no functional localizer exists for this dataset —
so the response is the participant-averaged mean over the "all" language-region
voxel set. Words are presented at a fixed 0.5 s each; each word's onset maps to its
2 s TR. Per-word features are pooled to TRs + 4 lagged blocks (BOLD delay); the
first 20 and last 15 TRs of each of the 4 runs are trimmed (1351 -> 1211 TRs).
Embeddings/surprisal are recomputed from the stimulus words with the shared
extractor.

Env vars: SFL_MODELS (comma-sep subset, default all), SFL_BACKEND (himalaya|sklearn).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT
from src import brain
from src.models import MODEL_ORDER

MODELS_TO_RUN = os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CACHE = PROJECT_ROOT / "cache"
RESULTS = PROJECT_ROOT / "results"
FM = PROJECT_ROOT / "data/wehbe"

SUBJECTS = ["F", "H", "I", "J", "K", "L", "M", "N"]

# brain response: mean over "all" language voxels, then over participants -> (1211,)
# The averaged response is shipped in the repo (wehbe_lang_avg.npy); the ~2 GB
# per-subject arrays are only needed to regenerate it.
AVG = FM / "wehbe_lang_avg.npy"
if AVG.exists():
    y = np.load(AVG)
else:
    roi = np.load(FM / "HP_subj_roi_inds.npy", allow_pickle=True).tolist()
    y = np.mean([np.load(FM / f"data_subject_{p}.npy")[:, roi[p]["all"]].mean(axis=1)
                 for p in SUBJECTS], axis=0)
    np.save(AVG, y)

# per-run [20:-15] trim of the (lagged) TR predictors -> the 1211 kept TR rows
runs = np.load(FM / "runs_fmri.npy")
keep = np.concatenate([np.where(runs == r)[0][20:-15] for r in [1, 2, 3, 4]])
assert len(keep) == len(y), f"{len(keep)} kept TRs vs {len(y)} responses"

# word -> TR via fixed 0.5 s/word onsets
words = [str(w) for w in np.load(FM / "words_fmri.npy", allow_pickle=True)]
time = np.load(FM / "time_fmri.npy")            # (1351,) TR onset times
time_words = np.load(FM / "time_words_fmri.npy")  # (5176,) word onset times
ends = time_words[1:]                            # non-initial words (extract drops first)
word_tr = np.array([np.where(e > time)[0][-1] if np.any(e > time) else 0 for e in ends])

story = {"words": words, "word_tr": word_tr, "n_tr": len(time), "y": y, "keep": keep}
print(f"Wehbe2014: {len(y)} TRs (from {len(time)} pre-trim), {len(words)} words")

brain.run_tr_dataset("HP", [story], MODELS_TO_RUN, results_dir=str(RESULTS),
                     cache_dir=str(CACHE), loso=False, backend=BACKEND)
