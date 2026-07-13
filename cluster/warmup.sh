#!/bin/bash
# One-time setup on Engaging: ensure the two extra deps are in the active env
# (modularity-conda already has torch + transformers), then pre-download the 5
# models. flock-guarded by the SLURM scripts so it runs once across parallel tasks.
set -e
cd "$(dirname "$0")/.."
python -c "import himalaya, pyreadr" 2>/dev/null || pip install -q himalaya pyreadr
python cluster/warmup_models.py
