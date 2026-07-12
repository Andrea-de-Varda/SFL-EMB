# SFL-EMB

Code and data for **de Varda et al., _"Behavioral and Brain Responses to Language
Reflect Different Levels of Linguistic Representation."_**

The paper compares two families of per-word predictors as models of human reading
and brain responses:

- **SFL** — three scalars per word: **s**urprisal (from a language model),
  log-**f**requency (SUBTLEX), and word **l**ength.
- **EMB** — the model's final-layer hidden state for the word (768–1600 dims).

**Main finding:** SFL predicts *reading behavior* better; EMB predicts *brain
responses* better. Behavior is a compressed readout; the brain retains richer
representational content. A random-Fourier-feature (RFF) expansion of SFL tests how
much of EMB's brain advantage is dimensionality/pooling rather than content.

Five models are used (same model for surprisal and embeddings within an analysis):
`openai-gpt`, `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`.

---

## Repository layout

```
src/            shared pipeline (import as a package)
  extract.py      transformers embeddings + surprisal (mean/sum over sub-words)
  fit.py          ridge encoding; himalaya (GPU, default) or sklearn backend
  rff.py          random-Fourier-feature SFL expansion + pooling helpers
  conditions.py   the 5-condition runner + permutation significance
  behavioral.py   word-level & spillover drivers (eye-tracking, SPR, maze, EEG)
  brain.py        sentence-level & TR-level drivers (fMRI)
  models.py       the 5 models (dims, last layer, tokenizer separators)
  significance.py paired permutation test on fold-level r
  paths.py        repo-root resolution
  prep/           one-off data-prep (passage labels, whisper timings, TR series)
probe_*.py      one script per dataset (13 total)
appendix/       layer sweep (A4) and reliability (A5) analyses
figures/        plotting scripts + rendered figures
cluster/        run_all.sh, slurm_engaging.sh, ENGAGING.md
data/           all curated input data (see below)
results/        all outputs (<dataset>_..._results.csv + per-fold predictions)
paper/          manuscript + PAPER_SUMMARY.md
```

`data/` holds every input: the behavioral corpora (`Provo/`, `Meco/`, `ZuCo/`,
`UCL_ET_CORPUS/`, `Brown/`, `NaturalStories/`, `ERP_Frank/`), `subtlex.xlsx`,
`corpora/` (sentence lists), and the brain datasets under
`data/{pereira,tuckute,natstor_fmri,wehbe}/`. Large raw arrays (fMRI subject `.npy`,
audio, the Tuckute table) are `.gitignore`d — back them up externally.

---

## The pipeline

Every dataset runs the **5 predictor conditions**:

1. `SFL` · 2. `EMB` · 3. `SFL+EMB` (`all`) · 4. `SFL_RFF` · 5. `SFL_RFF+EMB`

where `SFL_RFF` projects SFL through random Fourier features to `D` = the model's
hidden size (dimensionality-matched to EMB), with the RBF bandwidth set by the
median heuristic and everything fit on the training fold only (leakage-safe).

**Fitting** (`src/fit.py`): ridge regression, **10-fold sequential** outer CV
(GroupKFold by passage for Pereira; leave-one-story-out for NatStor-fMRI), alpha
chosen by inner `KFold(5)` over a log grid, per-fold `StandardScaler` on X and y
with `fit_intercept=False`. The **himalaya** GPU backend is the default and agrees
with the **sklearn** backend to ~1e-4. Significance is a paired permutation test on
the fold-level r values.

Extraction is **ex novo**: `src/extract.py` runs the model for both embeddings
(mean over sub-word tokens) and surprisal (sum over sub-word tokens) every run,
caching to `cache/` (git-ignored, regenerable). No precomputed activations are used.

---

## Running it

```bash
conda env create -f environment.yml && conda activate sfl-emb

# one dataset, one model (quick check)
SFL_MODELS=gpt2_small python probe_pereira.py

# everything (13 datasets x 5 models x 5 conditions)
bash cluster/run_all.sh
```

Env vars: `SFL_MODELS` (comma-separated subset of
`gpt,gpt2_small,gpt2_medium,gpt2_large,gpt2_xl`; default all) and `SFL_BACKEND`
(`himalaya` default, or `sklearn`).

On MIT Engaging: `sbatch cluster/slurm_engaging.sh` (single GPU), or
`sbatch cluster/slurm_engaging_array.sh` to run across **3 GPUs in parallel** — a
SLURM job array that statically partitions the 13 datasets into cost-balanced
buckets (one GPU each, per-task logs, no re-runs since results/cache are per-dataset).
A `flock`-guarded warmup downloads the 5 models once. See `cluster/ENGAGING.md`.

Results land in `results/`: one `results/<dataset>_<target>_results.csv` per
target (rows = models; columns = the 5 conditions' mean r + sd + permutation
p-values), plus per-fold r vectors `results/<prefix>_<condition>_predictions`.

---

## Datasets (13)

Behavioral (8): Provo, Meco-en, ZuCo-2, UCL-ET, UCL-SPR, Brown, NatStor-SPR,
NatStor-Maze. Brain (5): UCL-N400 (EEG), Wehbe2014 & NatStor-fMRI (TR fMRI),
Pereira2018 & Tuckute2024 (sentence fMRI). See `paper/PAPER_SUMMARY.md` for the
per-dataset sample sizes, measures, and preprocessing rules; the probe sample sizes
reproduce that table.
