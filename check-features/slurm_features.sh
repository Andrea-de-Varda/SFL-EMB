#!/bin/bash
# Predict 14 sentence features (semantic ratings + form/syntax) from SFL / EMB / SFL_RFF.
# CPU only -- loads the already-extracted control_fmri activations (no model forward
# pass), but the XL JSON is ~12 GB so request plenty of RAM. 14 features x 5 LMs of
# nested-gamma RFF is the time-dominant part -> generous wall time.
#   cd /orcd/data/evelina9/001/USERS/devar_ag/SFL-EMB && sbatch check-features/slurm_features.sh
#SBATCH -p pi_evelina9
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH --job-name=sfl_features
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate neurox

export TOKENIZERS_PARALLELISM=false
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
python check-features/predict_features.py
