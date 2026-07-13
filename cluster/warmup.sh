#!/bin/bash
# One-time setup on Engaging: ensure the pipeline's Python deps are present in the
# active env (modularity-conda already has torch + transformers + numpy/pandas), then
# pre-download the 5 models. flock-guarded by the SLURM scripts so it runs once across
# parallel tasks.
set -e
cd "$(dirname "$0")/.."

# install any missing dependency (idempotent; a no-op if all present)
python - <<'PY'
import importlib, subprocess, sys
need = []
for mod, pip_name in [("himalaya", "himalaya"), ("pyreadr", "pyreadr"),
                      ("openpyxl", "openpyxl"), ("sklearn", "scikit-learn"),
                      ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("tqdm", "tqdm")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pip_name)
if need:
    print("installing missing deps:", need, flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
else:
    print("all deps present.")
PY

python cluster/warmup_models.py
