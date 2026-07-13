#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=sfl_emb
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Run the full SFL-EMB pipeline across 3 GPUs in parallel (one SLURM array task per
# GPU). Datasets are statically partitioned into 3 cost-balanced buckets, so each
# dataset runs on exactly one GPU -> no re-runs, no clobber (results/ and cache/ are
# keyed per dataset). Each task writes its own log: logs/sfl_emb_<jobid>_<0|1|2>.out
#
#   sbatch cluster/slurm_engaging_array.sh
#
# The flock-guarded warmup installs himalaya/pyreadr (if missing) and downloads the 5
# models once: the first task to reach it does the work while the others block
# briefly, then find everything ready. Single race-free command.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source cluster/env.sh

# --- cost-balanced buckets (heaviest datasets split across GPUs) ---
BUCKET_0=(probe_maze.py        probe_wehbe.py        probe_ucl_et.py  probe_ucl_n400.py)
BUCKET_1=(probe_natstor_spr.py probe_natstor_fmri.py probe_meco.py    probe_ucl_spr.py)
BUCKET_2=(probe_brown.py       probe_zuco.py         probe_pereira.py probe_tuckute.py  probe_Provo.py)

case "$SLURM_ARRAY_TASK_ID" in
  0) BUCKET=("${BUCKET_0[@]}") ;;
  1) BUCKET=("${BUCKET_1[@]}") ;;
  2) BUCKET=("${BUCKET_2[@]}") ;;
  *) echo "unexpected array id $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

# one-time setup, serialized across tasks by a lock
echo "[task $SLURM_ARRAY_TASK_ID] ensuring deps + 5 models are ready ..."
flock "$HF_HOME/.sfl_warmup.lock" bash cluster/warmup.sh

echo "[task $SLURM_ARRAY_TASK_ID] running ${#BUCKET[@]} datasets: ${BUCKET[*]}"
bash cluster/run_all.sh "${BUCKET[@]}"
echo "[task $SLURM_ARRAY_TASK_ID] done."
