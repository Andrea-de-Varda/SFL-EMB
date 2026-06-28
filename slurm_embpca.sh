#!/bin/bash
# EMB PCA-down analysis -> results_embpca/. SUBMIT WITH sbatch (never the login node):
#   cd /orcd/data/evelina9/001/USERS/devar_ag/SFL-EMB && sbatch slurm_embpca.sh
#   one dataset:  sbatch slurm_embpca.sh pereira
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --job-name=sfl_embpca
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# GPU not actually needed (activations are precomputed); --gres just guarantees scheduling.

source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate neurox
python -c "import openpyxl" 2>/dev/null || pip install -q openpyxl
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
bash run_embpca.sh "$1"
