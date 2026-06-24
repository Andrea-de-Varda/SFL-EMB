#!/bin/bash
# Run the dimensionality-matched SFL-vs-EMB analyses for the 4 brain datasets.
# These are HEAVY (load activation dumps up to ~7.6 GB + nested grouped CV) and
# need the `neurox` env. Run on the cluster or locally when the laptop is free.
#
# Pereira was done separately (probe_Pereira_passages.py).
#
# --- local / interactive ---
#   bash run_matched_brain.sh                 # all four, sequentially
#   bash run_matched_brain.sh control         # just one (control|natstorfmri|HP|n400)
#
# --- SLURM (Engaging) sketch: 1 CPU job, no GPU (activations are precomputed) ---
#   #!/bin/bash
#   #SBATCH -p pi_evelina9
#   #SBATCH -c 4
#   #SBATCH --mem=64G
#   #SBATCH -t 12:00:00
#   #SBATCH --job-name=sfl_matched
#   #SBATCH --output=logs/%x_%j.out
#   ... module/conda setup ...
#   bash run_matched_brain.sh
set -e
# Activate the `neurox` env BEFORE calling this script (laptop: conda activate neurox;
# Engaging: see slurm_run_matched.sh / ENGAGING.md). We don't hardcode a conda path.
python -c "import neurox" 2>/dev/null || { echo "ERROR: activate the 'neurox' env first"; exit 1; }
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")"

declare -A SCRIPTS=(
  [control]=probe_control_matched.py
  [natstorfmri]=probe_natstorfmri_matched.py
  [HP]=probe_HP_matched.py
  [n400]=probe_UCL_N400_matched.py
)
ORDER=(control n400 natstorfmri HP)   # lightest -> heaviest

run_one() { echo "=== $(date '+%H:%M:%S') running $1 ==="; python -u "$1"; }

if [ -n "$1" ]; then
  run_one "${SCRIPTS[$1]:?unknown dataset: use control|natstorfmri|HP|n400}"
else
  for k in "${ORDER[@]}"; do run_one "${SCRIPTS[$k]}"; done
fi
echo "=== done ==="
