#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH -t 00:20:00
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --job-name=sfl_recompute
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# CPU-only (no GPU): recompute the significance contrasts from the saved per-fold
# prediction pickles WITHOUT refitting, then regenerate all figures. Use after adding
# a contrast to conditions.CONTRASTS (e.g. p_sflrffemb_sflrff). Submit with:
#   sbatch cluster/slurm_recompute.sh
#
# Requires results/*_predictions from a prior pipeline run to be present on disk. If
# they were cleaned, re-run the pipeline instead (cluster/slurm_engaging_array.sh).

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source cluster/env.sh

python recompute_significance.py
python figures/plot_results.py
