#!/usr/bin/env python3
"""NatStor self-paced reading (A1). Word-level with i, i-1, i-2 spillover on both
SFL and EMB.

The stories (data/natstor.tsv) are split into sentences at sentence-final
punctuation so spillover never crosses a sentence boundary; the target is the
per-word geometric-mean item RT. Non-finite RTs are dropped by the driver.

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
CACHE, RESULTS = PROJECT_ROOT / "cache", PROJECT_ROOT / "results"

END = re.compile(r'[\.!?](?:["\')\]]*)$')   # sentence-final punctuation

df = pd.read_csv(PROJECT_ROOT / "data/natstor.tsv", sep="\t").sort_values(["item", "zone"])

sentences, sent_rts = [], []
for item in sorted(df["item"].unique()):
    t = df[df["item"] == item]
    cur_w, cur_r = [], []
    for w, r in zip(t["word"].astype(str), t["gmeanItemRT"]):
        cur_w.append(w)
        cur_r.append(r)
        if END.search(w):
            sentences.append(cur_w); sent_rts.append(cur_r); cur_w, cur_r = [], []
    if cur_w:
        sentences.append(cur_w); sent_rts.append(cur_r)

target = np.array([r for rts in sent_rts for r in rts[1:]], dtype=float)  # non-initial words
sent_ids = [s for s, sent in enumerate(sentences) for _ in sent[1:]]
print(f"NatStor_SPR: {len(target)} non-initial words across {len(sentences)} sentences")

behavioral.run_spill_dataset("natstor_spr", sentences, target, sent_ids, MODELS_TO_RUN,
                             results_dir=str(RESULTS), cache_dir=str(CACHE), backend=BACKEND)
