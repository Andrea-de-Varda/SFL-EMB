#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=sfl_emb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Full SFL-EMB pipeline on MIT Engaging, single GPU. Submit with:
#   sbatch cluster/slurm_engaging.sh
# or a subset:
#   sbatch cluster/slurm_engaging.sh probe_pereira.py probe_wehbe.py
#
# Reuses the existing `modularity-conda` env (see cluster/env.sh). For 3-GPU
# parallel runs use cluster/slurm_engaging_array.sh instead.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source cluster/env.sh

# one-time: add himalaya/pyreadr if missing + download the 5 models
flock "$HF_HOME/.sfl_warmup.lock" bash cluster/warmup.sh

bash cluster/run_all.sh "$@"
