#!/bin/bash
# Symmetric-RFF exploratory analysis (EMB and SFL BOTH routed through RFF, same gamma
# policy, same D). Writes to results_rffsym/ (TEMP -- main results/ untouched).
# Activate the neurox env BEFORE running. HP needs openpyxl (pip install openpyxl).
#   bash run_rffsym.sh                 # all five
#   bash run_rffsym.sh pereira         # one (pereira|control|n400|natstorfmri|HP)
set -e
python -c "import neurox" 2>/dev/null || { echo "ERROR: activate the 'neurox' env first"; exit 1; }
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")"
run() {
  case "$1" in
    pereira|control|n400) python -u probe_rffsym_sentence.py "$1" ;;
    natstorfmri|HP)       python -u probe_rffsym_tr.py "$1" ;;
    *) echo "unknown: $1"; exit 1 ;;
  esac
}
if [ -n "$1" ]; then run "$1"; else for d in control n400 pereira natstorfmri HP; do echo "== $d =="; run "$d"; done; fi
echo "=== done -> results_rffsym/ ==="
