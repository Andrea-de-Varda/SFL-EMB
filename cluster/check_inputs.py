#!/usr/bin/env python3
"""Preflight: verify every input the pipeline reads is present before running.

Most inputs are tracked and arrive with the clone; two sets are gitignored (too
large for GitHub) and must be supplied manually on a fresh checkout:
  - data/wehbe/data_subject_*.npy   (Wehbe fMRI subject arrays, ~2 GB)
  - data/tuckute/brain-lang-data_participant_*.csv   (Tuckute response table)

Exits 0 if everything is present, 1 (listing what's missing) otherwise.
Run from the repo root:  python cluster/check_inputs.py
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# (path or glob, is_gitignored/must-supply-manually)
REQUIRED = [
    ("data/subtlex.xlsx", False),
    ("data/Provo/Provo_Corpus-Eyetracking_Data.csv", False),
    ("data/Provo/sentences.txt", False),
    ("data/Meco/Meco_l1.csv", False),
    ("data/ZuCo/ET_data_aggregated.csv", False),
    ("data/UCL_reading_all.csv", False),
    ("data/NaturalStories/natstor_maze.rds", False),
    ("data/corpora/bnc_brown.txt", False),
    ("data/corpora/maze_sentences.txt", False),
    ("data/brown_surprisal.csv", False),
    ("data/natstor.tsv", False),
    ("data/pereira/pereira_averaged_with_passage.csv", False),
    ("data/natstor_fmri/response/d_natstor", False),
    ("data/natstor_fmri/transcribed/1.csv", False),
    ("data/wehbe/words_fmri.npy", False),
    ("data/wehbe/HP_subj_roi_inds.npy", False),
    ("data/wehbe/runs_fmri.npy", False),
    ("data/wehbe/time_fmri.npy", False),
    ("data/wehbe/time_words_fmri.npy", False),
    ("data/tuckute/brain-lang-data_participant_*.csv", True),   # gitignored
    ("data/wehbe/data_subject_F.npy", True),                    # gitignored (+ H..N below)
]
# the other 7 Wehbe subject arrays
REQUIRED += [(f"data/wehbe/data_subject_{s}.npy", True) for s in "HIJKLMN"]

missing_tracked, missing_supply, ok = [], [], 0
for pat, ignored in REQUIRED:
    hit = glob.glob(pat)
    if hit and all(os.path.getsize(h) > 0 for h in hit):
        ok += 1
    else:
        (missing_supply if ignored else missing_tracked).append(pat)

print(f"{ok}/{len(REQUIRED)} required inputs present.")
if missing_tracked:
    print("\nMISSING tracked inputs (should have come with the clone — bad checkout?):")
    for p in missing_tracked:
        print("   ", p)
if missing_supply:
    print("\nMISSING gitignored inputs — copy these onto the cluster manually:")
    for p in missing_supply:
        print("   ", p)
    print("   (from the OLD clone: fMRI/fMRI/data_subject_*.npy -> data/wehbe/ ; "
          "control_fmri/data/brain-lang-data_*.csv -> data/tuckute/ ; or rsync from the laptop)")

if missing_tracked or missing_supply:
    sys.exit(1)
print("\nAll inputs present — ready to run.")
