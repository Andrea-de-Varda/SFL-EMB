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
# A flock-guarded warmup downloads the 5 models once (the first task to reach it
# downloads; the others block briefly, then find them cached), avoiding a
# concurrent-download race. Single command; no dependency chaining needed.

source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate sfl-emb

export HF_HOME=${HF_HOME:-/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new}
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$HF_HOME"

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

# --- one-time model download, serialized across tasks by a lock ---
echo "[task $SLURM_ARRAY_TASK_ID] ensuring the 5 models are cached ..."
flock "$HF_HOME/.sfl_warmup.lock" python cluster/warmup_models.py

echo "[task $SLURM_ARRAY_TASK_ID] running ${#BUCKET[@]} datasets: ${BUCKET[*]}"
bash cluster/run_all.sh "${BUCKET[@]}"
echo "[task $SLURM_ARRAY_TASK_ID] done."
