#!/bin/bash
# Engaging environment setup. Reuses the existing `modularity-conda` env (already
# has torch + transformers); himalaya + pyreadr are added on first run by warmup.sh.
# Includes the finicky conda double-activate (activate -> deactivate -> activate)
# that this env needs to come up reliably in a batch shell.
#
# `source`d by cluster/slurm_engaging*.sh.

source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate modularity-conda
conda deactivate
conda activate modularity-conda

export HF_HOME=${HF_HOME:-/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new}
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_OFFLOAD_DIR=${HF_OFFLOAD_DIR:-/orcd/data/evelina9/001/USERS/devar_ag/offload}
mkdir -p "$HF_OFFLOAD_DIR" "$HF_HOME"
# export HF_TOKEN=...   # only needed for gated models; GPT / GPT-2 are public
