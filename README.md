# SFL-EMB

### Using language models to relate behavioral and brain responses in human language comprehension

Code and data for **de Varda, Berzak, Fedorenko & Levy**. A single analytic framework
for asking a question that behavioral and neural language research usually answer
separately: **do reading behavior and brain activity encode the same information?**

<p align="center">
  <img src="docs/hero.png" width="640" alt="SFL predicts behavior; EMB predicts the brain — dumbbell across 13 datasets">
</p>

We compare two families of per-word predictors, both derived from the *same* language
model, across **8 behavioral and 5 neural datasets**:

- 🟢 **SFL** — three interpretable scalars per word: **s**urprisal, log-**f**requency,
  word **l**ength. They reflect the classic "effort" account of processing difficulty.
- 🔵 **EMB** — the model's final-layer embedding for the word (768–1600 dims): rich,
  contextual representation which includes linguistic meaning.

> **TL;DR:** SFL predicts **reading behavior** better; EMB predicts **brain
> responses** better. Behavior looks like a *compressed, effort-based readout* of
> language processing, while the brain retains far more of the high-dimensional
> representational content. The dissociation holds even on the same sentences read
> under both modalities (the UCL and NatStories corpora), so it reflects the measures
> themselves, not the stimuli.

Because every dataset runs through one identical pipeline — same models, same ridge
estimator, same cross-validation and significance test — this repo is also a general
tool for **comparing behavioral and neural data** and asking
which representations each modality is sensitive to.

---

## The pipeline

```mermaid
flowchart LR
    W["word w_i<br/>in context"] --> S["SFL<br/>surprisal · logfreq · length<br/>3 dims"]
    W --> E["EMB<br/>LM final-layer state<br/>768–1600 dims"]
    S --> R["ridge regression<br/>10-fold CV"]
    E --> R
    R --> BEH["reading behavior<br/>(eye-tracking, SPR, maze)"]
    R --> BRN["brain response<br/>(EEG, fMRI)"]
```

Each dataset is fit under **five predictor conditions**, so we can read off both
absolute predictivity and what each predictor adds over the other:

| condition | predictor | encodes |
|---|---|---|
| `sfl` | surprisal, log-freq, length | effort |
| `emb` | LM embedding (D dims) | full representation |
| `all` | `sfl` ⌢ `emb` | effort+representation |
| `sfl_rff` | SFL → D random Fourier features | dimensionality expanded in SFL |
| `sfl_rff_emb` | `sfl_rff` ⌢ `emb` | dimensionality expanded SFL + representation |

---

## Datasets (13)

Curated from published resources; a subset (UCL, NatStories) has **both** behavioral
and neural recordings on the same sentences. `N` is the number of data points
(words / TRs / sentences) after preprocessing; reliability is split-half.

| | Dataset | Measure | Modality | Unit | N | Split-half *r* | Driver |
|---|---|---|---|--:|--:|--:|---|
| 🟢 | **Provo** | Eye-tracking | reading | word | 2,659 | 0.93 | `probe_Provo.py` |
| 🟢 | **Meco-en** | Eye-tracking | reading | word | 2,096 | 0.87 | `probe_meco.py` |
| 🟢 | **ZuCo-2** | Eye-tracking | reading | word | 6,461 | 0.76 | `probe_zuco.py` |
| 🟢 | **UCL-ET** | Eye-tracking | reading | word | 1,726 | 0.84 | `probe_ucl_et.py` |
| 🟢 | **UCL-SPR** | Self-paced reading | reading | word | 1,316 | 0.22 | `probe_ucl_spr.py` |
| 🟢 | **Brown** | Self-paced reading | reading | word | 5,862 | 0.39 | `probe_brown.py` |
| 🟢 | **NatStor-SPR** | Self-paced reading | reading | word | 8,779 | 0.70 | `probe_natstor_spr.py` |
| 🟢 | **NatStor-Maze** | Maze | reading | word | 9,779 | 0.47 | `probe_maze.py` |
| 🔵 | **UCL-N400** | ERP (N400) | EEG | word | 1,505 | 0.22 | `probe_ucl_n400.py` |
| 🔵 | **Wehbe2014** | BOLD | fMRI (visual) | TR | 1,211 | 0.16 | `probe_wehbe.py` |
| 🔵 | **NatStor-fMRI** | BOLD | fMRI (auditory) | TR | 1,516 | 0.61 | `probe_natstor_fmri.py` |
| 🔵 | **Pereira2018** | BOLD | fMRI (visual) | sentence | 627 | 0.28 | `probe_pereira.py` |
| 🔵 | **Tuckute2024** | BOLD | fMRI (visual) | sentence | 2,000 | 0.22 | `probe_tuckute.py` |

Each dataset's preprocessing (filtering, aggregation, exclusions) is documented in
the docstring of its `probe_<name>.py` driver.

---

## Quick start

```bash
conda env create -f environment.yml
conda activate sfl-emb

# one dataset, one model (fast smoke test)
SFL_MODELS=gpt2_small python probe_pereira.py

# one dataset, all five models
python probe_pereira.py

# everything: 13 datasets × 5 models × 5 conditions
bash cluster/run_all.sh
```

Two environment variables control every entry point:

- **`SFL_MODELS`** — comma-separated subset of
  `gpt,gpt2_small,gpt2_medium,gpt2_large,gpt2_xl` (default: all five).
- **`SFL_BACKEND`** — `himalaya` (GPU ridge, default) or `sklearn` (CPU). The two
  agree to ~1e-4, so you can develop on CPU and scale up on a GPU.

Everything runs on a single machine with a decent GPU — the full sweep (13 datasets ×
5 models × 5 conditions) just takes a while. For faster turnaround it can be
parallelized across GPUs on a cluster (see below).

Results end up in `results/<dataset>_<target>_results.csv`: one row per model, columns
for each condition's mean *r* and SD plus permutation *p*-values.

### System requirements

- **OS:** Linux. Tested on Ubuntu 26.04 LTS (laptop, CPU only) and on the MIT Engaging cluster (Linux, NVIDIA A100 GPUs). Nothing in the code is Linux-specific, but other operating systems have not been tested.
- **Software:** the dependencies in `environment.yml` (Python 3.11). Tested with numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, scikit-learn 1.8.0, matplotlib 3.10.8, torch 2.9.1, transformers 4.57.5, himalaya 0.4.11, pyreadr 0.5.6 and openpyxl 3.1.5.
- **Hardware:** no non-standard hardware is required. A CUDA GPU speeds up embedding extraction and the `himalaya` backend, but everything also runs on CPU with `SFL_BACKEND=sklearn`. About 16 GB of RAM is enough for the demo; the largest models (`gpt2_large`, `gpt2_xl`) and the long word-level datasets benefit from more memory or a GPU.
- **Disk/network:** the pretrained models are downloaded from the Hugging Face Hub on first use (about 0.5 GB for `gpt2`, about 6 GB for all five models).

### Installation

`conda env create -f environment.yml` followed by `conda activate sfl-emb`. Typical install time on a normal desktop computer is roughly 10–20 minutes, most of it spent downloading PyTorch.

### Demo

The repository includes the processed data for every dataset, so the demo needs no extra downloads beyond the model weights. On a CPU-only machine:

```bash
SFL_MODELS=gpt2_small SFL_BACKEND=sklearn python probe_pereira.py
```

Expected output (Pearson *r* averaged over the 10 cross-validation folds; values may differ in the third or fourth decimal across backends and library versions):

```
Pereira2018: 627 sentences, 168 passages
  pereira_gpt2_small             sfl          r = 0.1763  (sd 0.1189)
  pereira_gpt2_small             emb          r = 0.4086  (sd 0.1004)
  pereira_gpt2_small             all          r = 0.4291  (sd 0.1084)
  pereira_gpt2_small             sfl_rff      r = 0.4855  (sd 0.0926)
  pereira_gpt2_small             sfl_rff_emb  r = 0.5191  (sd 0.0876)
```

The script also writes `results/pereira_results.csv`. Expected run time is about 2 minutes on a 16-thread laptop CPU, plus the one-off model download. Note that the demo overwrites the shipped `results/pereira_results.csv` with a single-model table; restore it with `git checkout results/pereira_results.csv`.

### Running on your own data

Each `probe_*.py` driver is a thin wrapper that loads one dataset and passes it to one of four entry points. To analyze a new English dataset, write an analogous driver:

- **Word-level responses without spill-over** (e.g. eye-tracking, Maze, ERP): `src.behavioral.run_word_dataset(dataset, sentences, targets, models, results_dir, cache_dir)`, where `sentences` is a list of whitespace-tokenized sentences and `targets` maps a target name to a response vector aligned to the non-initial words of each sentence (see `probe_ucl_et.py`).
- **Word-level responses with spill-over** (e.g. self-paced reading): `src.behavioral.run_spill_dataset(...)`, which additionally takes the sentence id of each word (see `probe_ucl_spr.py`).
- **Sentence-level responses** (e.g. sentence-wise fMRI): `src.brain.run_sentence_dataset(dataset, sentences, y, models, results_dir, cache_dir, groups=None)`, with one response per sentence; pass `groups` to hold out groups of sentences together (see `probe_pereira.py`).
- **TR-level responses to continuous text** (e.g. story fMRI): `src.brain.run_tr_dataset(dataset, stories, models, results_dir, cache_dir)`, where each story provides its words, the TR index of each word and the response per TR (see `probe_wehbe.py`).

Each entry point fits all five predictor conditions and writes `results/<dataset>_results.csv` in the format described above. Word frequencies come from SUBTLEX-US (`data/subtlex.xlsx`), so new datasets should be in English.

### Reproducing the paper

`bash cluster/run_all.sh` (or the SLURM scripts below) reruns all 13 datasets × 5 models × 5 conditions; `figures/plot_results.py` then regenerates the figures from `results/`. The supplementary analyses are listed under "Beyond the main comparison" below. Some of them (split-half reliability, the layer sweep, and the feature checks for Wehbe2014 and Tuckute2024) need the participant-level data, which are too large to ship and can be downloaded from the original sources (Wehbe2014: https://doi.org/10.5061/dryad.gt413; Tuckute2024: https://osf.io/ru38b/). The main analyses use the participant-averaged responses included in `data/`.

---

## How it works

**Models.** Five GPT variants (`openai-gpt`, `gpt2`, `gpt2-medium`, `gpt2-large`,
`gpt2-xl`); the same model supplies both surprisal and embeddings within an analysis,
recomputed from raw text on every run (cached under `cache/`, git-ignored). No
precomputed activations are stored.

**Predictors.**

- Embeddings are **mean-pooled** over a word's sub-word tokens (and over words for
  sentence/TR units); surprisal is **summed** over sub-words; frequency and length are
  word-level.
- Self-paced-reading datasets add **spillover** (words *i*, *i*−1, *i*−2 concatenated);
  eye-tracking and maze use word *i* only.
- TR-level fMRI pools per-word predictors into 2 s TRs and concatenates four lagged TR
  blocks for the hemodynamic delay.
- `sfl_rff` projects SFL into `D` random Fourier features (RBF-kernel approximation,
  median-heuristic bandwidth), fit on the training fold only — a leakage-safe,
  dimensionality-matched control for EMB.

**Fitting** (`src/fit.py`). Ridge regression, 10-fold cross-validation, α chosen per
fold by inner 5-fold CV. Predictors and responses are standardized within each
training fold. Two cross-validation schemes protect against leakage across the units
that share structure: **Pereira2018** holds out whole passages (`GroupKFold`), and
**NatStor-fMRI** holds out whole stories (leave-one-story-out, 9 folds); everything
else is sequential 10-fold. The `all` and `sfl_rff_emb` concatenations use **banded
ridge** (a separate penalty per feature block) so that combining predictors can never
score below the better block alone.

**Significance** (`src/significance.py`). A paired permutation test on the per-fold *r*
values — all 2ⁿ sign-flips of the paired fold differences.

```mermaid
flowchart LR
    D["raw text +<br/>human data"] --> X["extract.py<br/>embeddings + surprisal"]
    X --> C["conditions.py<br/>5 predictor sets"]
    C --> F["fit.py<br/>ridge · CV · banded ridge"]
    F --> R["results/<br/>mean r + permutation p"]
    R --> P["figures/<br/>result plots"]
```

---

## Repository layout

```
src/                importable pipeline package
  extract.py          transformers embeddings + surprisal (sub-word aligned)
  fit.py              ridge encoding, cross-validation, banded ridge
  rff.py              random-Fourier-feature SFL expansion + word/TR pooling
  conditions.py       the 5-condition runner
  significance.py     paired permutation test on fold-level r
  behavioral.py       word-level & spillover drivers (eye-tracking, SPR, maze, EEG)
  brain.py            sentence-level & TR-level drivers (fMRI)
  decompose.py        surprisal / frequency / length decomposition
  models.py           the 5 models (dims, last layer, tokenizer separators)
  prep/               one-off data prep (passage labels, Whisper timings, TR series)

probe_*.py            one driver per dataset (13); reads data/, writes results/
appendix/            supplementary analyses (layer sweep, reliability, RFF controls)
check-features/      what SFL_RFF vs EMB encode (meaning vs surface form)
figures/             result plots
cluster/             SLURM scripts + environment setup for MIT Engaging
data/                all curated input data
results/             fitted outputs (CSV summaries + per-fold predictions)
docs/                README assets
```

`data/` holds every input: the behavioral corpora, `subtlex.xlsx`, sentence lists, and
the brain datasets under `data/{pereira,tuckute,natstor_fmri,wehbe}/`. Large raw
arrays (fMRI subject `.npy`, audio, the Tuckute response table) are `.gitignore`d —
keep them backed up externally.

---

## Beyond the main comparison

- **Layer analysis** (`appendix/layerwise_logitlens.py`, `layerwise_significance.py`,
  `layerwise_plot.py`) — EMB predictivity at every layer vs SFL, per dataset. Heavy;
  run on the cluster (`cluster/slurm_layerwise.sh`).
- **SFL decomposition** (`src/decompose.py`) — surprisal / frequency / length taken
  individually and in every subset.
- **Reliability** (`appendix/reliability_and_size.py`) — split-half reliability of each
  dataset as a function of sample size.
- **Dimensionality & meaning controls** (`appendix/pereira_length_control.py`,
  `check-features/`) — is EMB's advantage about dimensionality, or content? What do the
  predictors actually encode (surface form vs meaning)?

---

## Figures

- `figures/plot_results.py` — per-dataset bar figures, SFL_RFF variants, SFL
  decomposition, and model-size boxplots, from `results/`.
- `docs/make_hero.py` — the summary figure at the top of this README.

---

## Running on a cluster

The SLURM scripts in `cluster/` are **written and tuned for the MIT Engaging
cluster** (partition names, module loads, GPU types, conda setup). On another cluster
they'll need editing — the `#SBATCH` headers and the environment setup in
`cluster/env.sh` in particular — but the structure carries over.

```bash
sbatch cluster/slurm_engaging.sh            # all datasets, single GPU
sbatch cluster/slurm_engaging_array.sh      # 3 GPUs in parallel (job array)
sbatch cluster/slurm_layerwise.sh           # layer analysis (4-GPU array)
```

The array scripts partition datasets into cost-balanced buckets (one GPU each,
per-task logs, per-dataset outputs, so re-submitting resumes rather than recomputes).
A `flock`-guarded warmup installs dependencies and downloads the five models once.
Details in `cluster/ENGAGING.md`.
