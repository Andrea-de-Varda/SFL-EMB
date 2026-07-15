#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=sfl_layerwise
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Layer-matched EMB vs SFL_lens (appendix/layerwise_logitlens.py) across 4 GPUs.
# Each array task owns a disjoint bucket of datasets, and each dataset writes its
# OWN csv (appendix/layerwise/<dataset>.csv), so tasks never share a writer.
# Completed (dataset, layer) cells are skipped, so re-submitting resumes.
#
#   sbatch cluster/slurm_layerwise.sh
#
# When all tasks are done, merge + plot LOCALLY after pulling (not on the login node):
#   python appendix/layerwise_logitlens.py --merge
#   python appendix/layerwise_plot.py
#
# Buckets are cost-balanced: the spillover datasets (SFL and EMB both carry i, i-1,
# i-2 -> 3x1024 dims) and the TR datasets (4 lagged blocks -> 4x1024) dominate, so
# they are spread one per task. Provo/Meco-en/ZuCo-2/UCL_ET were already computed
# locally and are listed only so a fresh clone still reproduces everything -- they
# are skipped in seconds if their csvs are present.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source cluster/env.sh

BUCKET_0=(NatStor_SPR UCL_N400 Provo)
BUCKET_1=(Brown NatStor_fMRI Meco-en)
BUCKET_2=(UCL_SPR Wehbe2014 Pereira2018 ZuCo-2)
BUCKET_3=(NatStor_Maze Tuckute2024 UCL_ET)

case "$SLURM_ARRAY_TASK_ID" in
  0) BUCKET=("${BUCKET_0[@]}") ;;
  1) BUCKET=("${BUCKET_1[@]}") ;;
  2) BUCKET=("${BUCKET_2[@]}") ;;
  3) BUCKET=("${BUCKET_3[@]}") ;;
  *) echo "unexpected array id $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

# one-time: deps + model download, serialized across tasks by a lock
echo "[task $SLURM_ARRAY_TASK_ID] ensuring deps + models are ready ..."
flock "$HF_HOME/.sfl_warmup.lock" bash cluster/warmup.sh

DS=$(IFS=,; echo "${BUCKET[*]}")
echo "[task $SLURM_ARRAY_TASK_ID] datasets: $DS"
python -u appendix/layerwise_logitlens.py --datasets "$DS"
echo "[task $SLURM_ARRAY_TASK_ID] done."
