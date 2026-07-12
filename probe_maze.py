#!/usr/bin/env python3
"""NatStor_Maze (A1). Word-level, NO spillover (high temporal resolution).

Applies the Boyce et al. (2023) maze exclusion criteria before averaging RT over
participants. Because many non-initial words are excluded, a row mask selects the
words that survive; the shared word-level driver handles the rest.

Env vars: SFL_MODELS (comma-sep subset, default all), SFL_BACKEND (himalaya|sklearn).
"""
import os
import sys

import numpy as np
import pandas as pd
import pyreadr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT
from src import behavioral
from src.models import MODEL_ORDER

MODELS_TO_RUN = os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CACHE, RESULTS = PROJECT_ROOT / "cache", PROJECT_ROOT / "results"

# --- load + map sentences to file order (defines sent_n / word alignment) ------
maze = pyreadr.read_r(PROJECT_ROOT / "main_data/NaturalStories/natstor_maze.rds")[None]
maze = maze[maze["native"].isin(['ENG', 'ENGLISH', 'English', 'english'])].copy()

maze_sent = [line.split() for line in
             open(PROJECT_ROOT / "corpora/maze_sentences.txt").read().splitlines()]
recover = {" ".join(words): i for i, words in enumerate(maze_sent)}
maze["sent_n"] = maze["sentence"].map(recover)

# --- Boyce exclusions -----------------------------------------------------------
attentive = maze.groupby("subject")["correct"].apply(lambda c: (c == "yes").mean() > 0.8)

# per-word error flag: a word (and the word after an error) is flagged
error, prev_idx, prev_cor = [], 0, True
for wn, cor in zip(maze["word_num"].to_numpy(), (maze["correct"] == "yes").to_numpy()):
    if wn - prev_idx == 1:
        error.append(0 if (cor and prev_cor) else 1)
    else:
        error.append(0 if cor else 1)
    prev_idx, prev_cor = wn, cor
maze["error"] = error
maze["is_attentive"] = maze["subject"].map(attentive)

maze = maze[(maze["type"] != "practice") & (maze["rt"] < 5000) & (maze["rt"] > 100)
            & (maze["is_attentive"]) & (maze["word_num"] != 0) & (maze["error"] == 0)]
maze = maze.dropna(subset=["sent_n"]).copy()
maze["sent_n"] = maze["sent_n"].astype(int)

# participant-mean RT per surviving word
agg = maze.groupby(["sent_n", "word_num"], as_index=False).agg({"rt": "mean"})
rt_map = {(int(r.sent_n), int(r.word_num)): float(r.rt) for r in agg.itertuples()}

# --- build sentences + target/mask aligned to the non-initial-word flattening ---
present = sorted(agg["sent_n"].unique())
sentences = [maze_sent[s] for s in present]
target, mask = [], []
for s in present:
    for pos in range(1, len(maze_sent[s])):        # non-initial words
        target.append(rt_map.get((s, pos), np.nan))
        mask.append((s, pos) in rt_map)
target, mask = np.array(target, dtype=float), np.array(mask)
print(f"NatStor_Maze: {int(mask.sum())} words across {len(present)} sentences")

behavioral.run_word_dataset("maze", sentences, {"maze": target}, MODELS_TO_RUN,
                            results_dir=str(RESULTS), cache_dir=str(CACHE),
                            backend=BACKEND, row_mask=mask)
