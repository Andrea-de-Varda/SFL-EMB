# SFL-EMB

Code and data for **de Varda et al., _"Behavioral and Brain Responses to Language
Reflect Different Levels of Linguistic Representation."_**

The study compares two families of per-word predictors as models of human reading
and brain responses:

- **SFL** — three scalars per word: **s**urprisal (from a language model),
  log-**f**requency (SUBTLEX), and word **l**ength.
- **EMB** — the language model's final-layer hidden state for the word (768–1600
  dimensions).

Across 8 behavioral and 5 neural datasets, SFL predicts *reading behavior* better
while EMB predicts *brain responses* better: behavior reflects a compressed,
effort-based readout, whereas the brain retains richer representational content.

Five models are used, the same model supplying both surprisal and embeddings within
an analysis: `openai-gpt`, `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`.

---

## Installation

```bash
conda env create -f environment.yml
conda activate sfl-emb
```

The GPU ridge backend (`himalaya`) needs CUDA; without a GPU, pass
`SFL_BACKEND=sklearn` (below) to run everything on CPU.

## Quick start

```bash
# one dataset, one model (fast smoke test)
SFL_MODELS=gpt2_small python probe_pereira.py

# one dataset, all five models
python probe_pereira.py

# every dataset x 5 models x 5 conditions
bash cluster/run_all.sh
```

Two environment variables control every entry point:

- `SFL_MODELS` — comma-separated subset of
  `gpt,gpt2_small,gpt2_medium,gpt2_large,gpt2_xl` (default: all five).
- `SFL_BACKEND` — `himalaya` (GPU, default) or `sklearn` (CPU). The two agree to
  ~1e-4.

---

## Repository layout

```
src/                importable pipeline package
  extract.py          transformers embeddings + surprisal (per-word, sub-word aligned)
  fit.py              ridge encoding, cross-validation, banded ridge
  rff.py              random-Fourier-feature SFL expansion + word/TR pooling
  conditions.py       the 5-condition runner
  significance.py     paired permutation test on fold-level r
  behavioral.py       word-level & spillover drivers (eye-tracking, SPR, maze, EEG)
  brain.py            sentence-level & TR-level drivers (fMRI)
  decompose.py        surprisal / frequency / length decomposition
  models.py           the 5 models (dims, last layer, tokenizer separators)
  paths.py            repo-root resolution
  prep/               one-off data prep (passage labels, Whisper timings, TR series)

probe_*.py            one driver per dataset (13); reads data/, writes results/
inspect_predicted_extremes.py   per-item predicted-response inspection (GPT-2-XL)

appendix/            supplementary analyses (layer sweep, reliability, RFF controls)
check-features/      what SFL_RFF vs EMB encode (meaning vs surface form)
figures/             plotting + Overleaf figure sync + numeric-audit scripts
cluster/             SLURM scripts + environment setup for MIT Engaging
data/                all curated input data (see Datasets)
results/             fitted outputs (CSV summaries + per-fold predictions)
paper/               manuscript summary
```

`data/` holds every input: the behavioral corpora (`Provo/`, `Meco/`, `ZuCo/`,
`UCL_ET_CORPUS/`, `Brown/`, `NaturalStories/`, `ERP_Frank/`), `subtlex.xlsx`,
`corpora/` (sentence lists), and the brain datasets under
`data/{pereira,tuckute,natstor_fmri,wehbe}/`. Large raw arrays (fMRI subject `.npy`,
audio, the Tuckute response table) are `.gitignore`d — keep them backed up
externally. Embeddings and surprisal are recomputed from the raw text on every run
and cached under `cache/` (git-ignored, regenerable); no precomputed activations are
stored.

---

## The pipeline

Each dataset is fit under **5 predictor conditions**:

| condition | predictor |
|---|---|
| `sfl` | surprisal, log-frequency, length |
| `emb` | final-layer embedding (D dims) |
| `all` | `sfl` ⌢ `emb` (banded ridge) |
| `sfl_rff` | SFL projected into D random Fourier features |
| `sfl_rff_emb` | `sfl_rff` ⌢ `emb` |

`sfl_rff` matches EMB in dimensionality: SFL is projected into `D` = the model's
hidden size via random Fourier features (an RBF-kernel approximation), with the RBF
bandwidth set by the median heuristic. Scaler, bandwidth and projection are fit on
the training fold only (leakage-safe).

**Fitting** (`src/fit.py`). Ridge regression with 10-fold cross-validation. The ridge
penalty α is chosen per training fold by inner 5-fold CV over a log-spaced grid.
Predictors and responses are standardized within each training fold (train-fit
statistics applied to the test fold), so the penalty acts uniformly across features.
Predictivity is the Pearson *r* between held-out predictions and the measured
response, averaged across folds.

**Cross-validation schemes.** Sequential (unshuffled) 10-fold everywhere, except:

- **Pereira2018** — leave whole passages out (`GroupKFold` on passage id), so
  sentences from one passage never split across train and test.
- **NatStor-fMRI** — leave one story out (9 folds).

**Significance** (`src/significance.py`). A paired permutation test on the per-fold
*r* values: for two conditions scored on the same folds, enumerate all 2ⁿ sign-flips
of the paired fold differences and report the fraction at least as extreme as
observed.

### Predictor construction

- **Embeddings** are mean-pooled over a word's sub-word tokens (and, for
  sentence/TR datasets, over the words in the unit).
- **Surprisal** is summed over a word's sub-word tokens; frequency and length are
  word-level.
- **Spillover.** Self-paced-reading datasets concatenate the predictors for words
  *i*, *i*−1, *i*−2 (both SFL and EMB); the eye-tracking and maze datasets use word
  *i* only.
- **TR-level fMRI** (Wehbe2014, NatStor-fMRI) pools per-word predictors into 2 s TRs
  and concatenates four lagged TR blocks for the hemodynamic delay.
- **RFF pooling.** For datasets aggregated above the word (all fMRI), the RFF
  features are summed within each unit, while EMB and the SFL base are mean-pooled.

### Banded ridge for `all`

A single global α on the concatenation of the 3 SFL scalars with the 768–1600 EMB
dimensions over-shrinks the few strong SFL coefficients in order to control the large
EMB block, so `all` could otherwise score below `sfl` alone. `all` therefore uses
**banded ridge** — a separate penalty per feature block (Nunez-Elizalde 2019; Dupré
la Tour 2022), with the between-block penalty ratio chosen on a deterministic grid by
inner CV (`src/fit.py:_banded_fit_predict`). `sfl_rff_emb` is two equal-width D-dim
blocks, so a single α is already well-behaved there.

### Outputs

Per target, `results/<dataset>_<target>_results.csv` has one row per model and
columns for each condition's mean *r* and SD plus the permutation *p*-values for the
key contrasts. The per-fold *r* vectors are written alongside as
`results/<prefix>_<condition>_predictions` (pickled lists), which the significance
and plotting code read.

---

## Supplementary analyses

- **`src/decompose.py`** (invoked by the drivers) — predictivity of surprisal,
  frequency and length individually and in every subset, for the SFL decomposition
  figure.
- **`appendix/layer_sweep.py`** — EMB predictivity by layer for the behavioral
  held-in/held-out layer analysis (GPT-2-medium).
- **`appendix/layerwise_logitlens.py`** — layer-matched EMB-vs-SFL comparison across
  all datasets and layers; `layerwise_significance.py` runs the paired permutation
  test per (dataset, layer) and `layerwise_plot.py` renders the per-dataset panels.
  Heavy — run on the cluster (`cluster/slurm_layerwise.sh`); see `cluster/ENGAGING.md`.
- **`appendix/reliability_and_size.py`** — split-half reliability of each dataset as
  a function of sample size (CPU, from the raw data; `--replot` re-uses the cache).
- **`appendix/pereira_length_control.py`** — what SFL_RFF and EMB add on top of
  sentence length in Pereira2018.
- **`check-features/`** — probes what SFL, SFL_RFF and EMB encode by predicting
  hand-annotated sentence features (`predict_features.py`) and correlating model
  predictions with those features (`analysis/`).

---

## Figures and manuscript

The manuscript lives in a separate Overleaf/LaTeX project. The figure scripts here
generate the assets and copy them in:

- **`figures/plot_results.py`** — the main per-dataset bar figures, the SFL_RFF
  variants, the SFL decomposition, and the model-size boxplots.
- **`figures/tikz_diagram_widths.py`** — line widths for the SFL-vs-EMB summary
  diagram (proportional to the per-dataset *r* gap).
- **`figures/make_legend_rff.py`** — standalone legend for the SFL_RFF figure.
- **`figures/sync_to_overleaf.py`** — copies every generated figure into the Overleaf
  tree under the name each `\includegraphics` expects, with a checksum audit that no
  referenced image is missing and no mapping is stale.
- **`figures/verify_paper_numbers.py`** — re-derives every fitted number reported in
  the manuscript from `results/` and checks it against the LaTeX source.

---

## Datasets (13)

Behavioral (8): Provo, Meco-en, ZuCo-2, UCL-ET (eye-tracking); UCL-SPR, Brown,
NatStor-SPR (self-paced reading); NatStor-Maze (maze).
Neural (5): UCL-N400 (EEG); Wehbe2014, NatStor-fMRI (TR-level fMRI); Pereira2018,
Tuckute2024 (sentence-level fMRI).

Each dataset has one `probe_<name>.py` driver. See `paper/PAPER_SUMMARY.md` for
per-dataset sample sizes, measures and preprocessing rules.

---

## Running on the cluster (MIT Engaging)

```bash
sbatch cluster/slurm_engaging.sh            # all datasets, single GPU
sbatch cluster/slurm_engaging_array.sh      # 3 GPUs in parallel (job array)
sbatch cluster/slurm_layerwise.sh           # layer-matched analysis (4-GPU array)
```

The array scripts partition the datasets into cost-balanced buckets, one per GPU,
with per-task logs and per-dataset outputs (so re-submitting resumes rather than
recomputes). A `flock`-guarded warmup installs any missing dependencies and downloads
the five models once. Details in `cluster/ENGAGING.md`.
