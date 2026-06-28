#!/bin/bash
# Dimensionality-REDUCTION analysis: EMB reduced via PCA DOWN to the SFL dimensionality
# (complement of run_rffsym.sh which expands SFL up). Writes to results_embpca/ (TEMP).
# Activate neurox BEFORE running. HP needs openpyxl. SUBMIT VIA sbatch (slurm_embpca.sh),
# never on the login node.
#   bash run_embpca.sh            # all five
#   bash run_embpca.sh pereira    # one (pereira|control|n400|natstorfmri|HP)
set -e
python -c "import neurox" 2>/dev/null || { echo "ERROR: activate the 'neurox' env first"; exit 1; }
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")"
run() {
  case "$1" in
    pereira|control|n400) python -u probe_rffsym_sentence.py "$1" pca ;;
    natstorfmri|HP)       python -u probe_rffsym_tr.py "$1" pca ;;
    *) echo "unknown: $1"; exit 1 ;;
  esac
}
if [ -n "$1" ]; then run "$1"; else for d in control n400 pereira natstorfmri HP; do echo "== $d =="; run "$d"; done; fi
echo "=== done -> results_embpca/ ==="
