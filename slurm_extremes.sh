#!/bin/bash
# Predicted-response extremes / predictor-agreement analysis (GPT-2 XL).
# CPU only -- it just LOADS the already-extracted activations (no model forward pass),
# but the XL activation JSONs are large so request plenty of RAM.
#   cd /orcd/data/evelina9/001/USERS/devar_ag/SFL-EMB && sbatch slurm_extremes.sh
#   one dataset:  sbatch slurm_extremes.sh pereira       (pereira|control|n400)
# NOTE: n400 needs activations_gpt2_xl_sum.json, produced by the n400 matched job
#       (sbatch slurm_run_matched.sh n400) -- run that first for the n400 case.
#SBATCH -p pi_evelina9
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 8:00:00
#SBATCH --job-name=sfl_extremes
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
python inspect_predicted_extremes.py "$@"
