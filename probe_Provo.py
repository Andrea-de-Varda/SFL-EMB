#!/usr/bin/env python3
"""Provo eye-tracking (A1 first-pass + A3 first-fixation / total-fixation).

Word-level, no spillover. For each of the 5 models runs the 5 conditions
(SFL, EMB, SFL+EMB, SFL_RFF, SFL_RFF+EMB) via the shared pipeline.

Data wrangling preserved from the original probe_ET_Provo.py:
  * aggregate the eye-tracking measures per word (mean over participants);
  * the corpus CSV omits each passage's first word -> recover it from the
    stimuli file and prepend it, so surprisal/embeddings have left context;
  * drop that sentence-initial word again when aligning to the behavioral rows.

Env vars: SFL_MODELS (comma-sep subset, default all), SFL_BACKEND (himalaya|sklearn).
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT
from src import behavioral
from src.models import MODEL_ORDER

MODELS_TO_RUN = os.environ.get("SFL_MODELS", ",".join(MODEL_ORDER)).split(",")
BACKEND = os.environ.get("SFL_BACKEND", "himalaya")
CACHE = PROJECT_ROOT / "cache"
RESULTS = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------- #
# behavioral data
# --------------------------------------------------------------------------- #
provo = pd.read_csv(PROJECT_ROOT / "main_data/Provo/Provo_Corpus-Eyetracking_Data.csv")
provo = provo[['Text_ID', 'Word_Unique_ID', 'Word_Number', 'Sentence_Number',
               'Word_In_Sentence_Number', 'Word', 'Word_Cleaned', 'Word_Length',
               'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_DWELL_TIME']]
for c in ['IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_DWELL_TIME']:
    provo[c] = provo[c].fillna(0)  # unfixated words -> 0 (paper convention)
provo = provo.groupby(
    ["Sentence_Number", "Word_In_Sentence_Number", "Word", "Text_ID"],
    as_index=False, sort=False).agg(
        {'IA_FIRST_FIXATION_DURATION': "mean", 'IA_FIRST_RUN_DWELL_TIME': "mean",
         'IA_DWELL_TIME': "mean", 'Word_Number': "mean", 'Sentence_Number': "mean",
         'Word_Unique_ID': "max", 'Word_In_Sentence_Number': "mean", 'Word': "max",
         'Word_Cleaned': "max", 'Word_Length': "mean", 'Text_ID': "mean"})

# the CSV omits each passage's first word -> recover from the stimuli file
first_word = [s.split("\t")[2][1:-2].split()[0]
              for s in open(PROJECT_ROOT / "main_data/Provo/sentences.txt").readlines()]
d_first = {num + 1: first_word[num] for num in range(55)}

# frequency (log SUBTLEX count)
freq_tbl = pd.read_excel(PROJECT_ROOT / "main_data/subtlex.xlsx")
_f = {row.Word: np.log(row.FREQcount) for _, row in freq_tbl.iterrows()}
_minf = min(_f.values())


def get_f(word):
    word = re.sub(r'[\.\,\:\-\?\!\)\(\"]', "", word)
    return _f.get(word, _minf)


ff, fp, tf, sentences = [], [], [], []
for text in sorted(set(provo["Text_ID"])):
    temp = provo[provo["Text_ID"] == text]
    ff.extend(temp["IA_FIRST_FIXATION_DURATION"])
    fp.extend(temp["IA_FIRST_RUN_DWELL_TIME"])   # first-pass gaze duration (A1 target)
    tf.extend(temp["IA_DWELL_TIME"])
    sentences.append([d_first[text]] + list(temp["Word"]))  # first word + CSV words

TARGETS = {"fpass": np.array(fp), "ffix": np.array(ff), "totfix": np.array(tf)}
print(f"Provo: {len(fp)} words across {len(sentences)} passages")

behavioral.run_word_dataset("provo", sentences, TARGETS, MODELS_TO_RUN,
                            results_dir=str(RESULTS), cache_dir=str(CACHE),
                            backend=BACKEND)
