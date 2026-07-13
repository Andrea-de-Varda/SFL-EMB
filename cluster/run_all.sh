#!/bin/bash
# Run the SFL-EMB pipeline: each dataset x 5 models x 5 conditions.
# Writes results/<dataset>_..._results.csv and per-fold prediction files.
# A failing dataset is reported but does NOT stop the rest (no set -e).
#
# Usage:
#   bash cluster/run_all.sh                        # all 13 datasets, all 5 models
#   SFL_MODELS=gpt2_small bash cluster/run_all.sh  # quick single-model smoke run
#   bash cluster/run_all.sh probe_pereira.py probe_wehbe.py   # a subset
cd "$(dirname "$0")/.."

PROBES=("$@")
if [ ${#PROBES[@]} -eq 0 ]; then
  PROBES=(
    probe_Provo.py probe_meco.py probe_zuco.py probe_ucl_et.py     # eye-tracking (A1+A3)
    probe_ucl_spr.py probe_brown.py probe_natstor_spr.py           # self-paced reading
    probe_maze.py                                                   # maze
    probe_ucl_n400.py                                               # EEG N400
    probe_pereira.py probe_tuckute.py                               # sentence fMRI
    probe_wehbe.py probe_natstor_fmri.py                            # TR fMRI
  )
fi

failed=()
for p in "${PROBES[@]}"; do
  echo "=================== $p ==================="
  if python -u "$p"; then
    echo "--- $p OK"
  else
    echo "!!! $p FAILED (rc=$?) — continuing with the rest"
    failed+=("$p")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "DONE with failures: ${failed[*]}"
  exit 1
fi
echo "All done. Results in results/"
