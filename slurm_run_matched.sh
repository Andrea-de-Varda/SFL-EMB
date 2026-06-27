#!/bin/bash
# Run all four matched-brain analyses on Engaging. One GPU node: n400 uses the GPU
# (embedding extraction), the other three are CPU-bound on the same node.
#   cd /orcd/data/evelina9/001/USERS/devar_ag/SFL-EMB && sbatch slurm_run_matched.sh
#   one dataset (e.g. re-run HP after `pip install openpyxl`): sbatch slurm_run_matched.sh HP
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --job-name=sfl_matched
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate neurox

export HF_HOME=/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
bash run_matched_brain.sh "$1"      # empty = all four; or pass one (control|n400|natstorfmri|HP)
