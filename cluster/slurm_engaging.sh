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
# Full SFL-EMB pipeline on MIT Engaging (GPU). The GPU is used for embedding /
# surprisal extraction and for himalaya ridge fitting. Submit with:
#   sbatch cluster/slurm_engaging.sh
# or a subset:
#   sbatch cluster/slurm_engaging.sh probe_pereira.py probe_wehbe.py

source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate sfl-emb

# HuggingFace cache + token (extraction downloads the 5 models once)
export HF_HOME=/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# export HF_TOKEN=...   # only needed for gated models (GPT/GPT-2 are public)

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

bash cluster/run_all.sh "$@"
