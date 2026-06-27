#!/bin/bash
# Re-run ONLY Wehbe2014/HP (it crashed earlier on a missing openpyxl). HP is
# CPU-only -- it loads precomputed HP_emb activations, no model run / no GPU.
#   cd /orcd/data/evelina9/001/USERS/devar_ag/SFL-EMB && sbatch slurm_HP.sh
#SBATCH -p pi_evelina9
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 20:00:00
#SBATCH --job-name=sfl_HP
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate neurox
pip install -q openpyxl          # idempotent: HP rebuilds freq from subtlex.xlsx (read_excel)
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
bash run_matched_brain.sh HP     # overwrites fMRI/results/all_results.csv
