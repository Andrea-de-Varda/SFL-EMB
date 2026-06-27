#!/bin/bash
# Symmetric-RFF analysis (EMB & SFL both RFF'd) -> results_rffsym/ (temp).
# SUBMIT WITH sbatch -- never run on the login node:
#   cd /orcd/data/evelina9/001/USERS/devar_ag/SFL-EMB && sbatch slurm_rffsym.sh
# One dataset only:  sbatch slurm_rffsym.sh pereira     (pereira|control|n400|natstorfmri|HP)
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --job-name=sfl_rffsym
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# NOTE: a GPU is not actually needed here (all activations are precomputed/loaded);
# the --gres line just guarantees scheduling on pi_evelina9. Drop it if CPU-only is allowed.

source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate neurox
python -c "import openpyxl" 2>/dev/null || pip install -q openpyxl   # needed for HP
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
bash run_rffsym.sh "$1"     # empty arg = all five; or pass one dataset
