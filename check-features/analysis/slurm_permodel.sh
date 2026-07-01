#!/bin/bash
# POINT 1: per-model predicted-response agreement (EMB vs EMB across LMs, EMB vs SFL+RFF).
# CPU only (loads already-extracted activations); XL is large so request memory.
#   cd /orcd/data/evelina9/001/USERS/devar_ag/SFL-EMB && sbatch check-features/analysis/slurm_permodel.sh
#SBATCH -p pi_evelina9
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH --job-name=sfl_permodel
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
python check-features/analysis/permodel_agreement.py
