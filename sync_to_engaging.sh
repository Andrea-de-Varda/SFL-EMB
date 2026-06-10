#!/bin/bash
# Push the gitignored data needed by the matched-brain analyses from the laptop to
# Engaging. The CODE is on GitHub (git clone on the cluster) -- this only moves the
# large, gitignored inputs (~45 GB). Run FROM the laptop.
#
#   bash sync_to_engaging.sh            # default remote (Andrea's evlab dir)
#   bash sync_to_engaging.sh user@host:/path/to/SFL-EMB
#
# UCL-N400 embeddings are NOT transferred -- probe_UCL_N400_matched.py extracts
# them on the cluster (needs a GPU). To also re-run Pereira on the cluster, add
# pereira_1/{activations_*.json,Pereira.parquet.gzip}.
set -e
REMOTE="${1:-engaging:/orcd/data/evelina9/001/USERS/devar_ag/SFL-EMB}"
cd "$(dirname "$0")"
rsync -avhP --relative --exclude='sem_*' \
  control_fmri/activations_gpt.json control_fmri/activations_gpt2_small.json \
  control_fmri/activations_gpt2_medium.json control_fmri/activations_gpt2_large.json \
  control_fmri/activations_gpt2_xl.json control_fmri/data \
  natstor_fmri/embeddings natstor_fmri/response/d_natstor natstor_fmri/transcribed \
  fMRI/HP_emb/activations_gpt.json fMRI/HP_emb/activations_gpt2_small.json \
  fMRI/HP_emb/activations_gpt2_medium.json fMRI/HP_emb/activations_gpt2_large.json \
  fMRI/HP_emb/activations_gpt2_xl.json \
  fMRI/fMRI fMRI/SFL fMRI/sentences.txt \
  "$REMOTE/"
echo "done."
