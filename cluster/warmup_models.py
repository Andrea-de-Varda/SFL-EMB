#!/usr/bin/env python3
"""Pre-download the 5 language models into the HuggingFace cache.

Run once before fanning the pipeline out across GPUs so that parallel array tasks
don't each try to download the same model at the same time. Downloading needs no
GPU. Idempotent: already-cached models return immediately.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import MODELS

from transformers import AutoModelForCausalLM, AutoTokenizer

for key, cfg in MODELS.items():
    hf = cfg["hf"]
    print(f"warming up {key} ({hf}) ...", flush=True)
    AutoTokenizer.from_pretrained(hf, add_prefix_space=(cfg["sep"] == "Ġ"))
    AutoModelForCausalLM.from_pretrained(hf)   # covers both surprisal + hidden states
print("all 5 models cached.")
