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
  make_legend_rff.py  standalone legend asset for the SFL_RFF SI figure
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

Every dataset runs **5 predictor conditions**. EMB is mean-pooled (over sub-words,
and over words for the pooled datasets):

1. `sfl` · 2. `emb` · 3. `all` (SFL+EMB) · 4. `sfl_rff` · 5. `sfl_rff_emb`

where `sfl_rff` projects SFL through random Fourier features to `D` = the model's
hidden size (dimensionality-matched to EMB), RBF bandwidth by the median heuristic,
everything fit on the training fold only (leakage-safe).

**Fitting** (`src/fit.py`): ridge regression, **10-fold sequential** outer CV
(GroupKFold by passage for Pereira; leave-one-story-out for NatStor-fMRI), alpha
chosen by inner `KFold(5)` over a log grid, per-fold `StandardScaler` on X and y
with `fit_intercept=False`. The **himalaya** GPU backend is the default and agrees
with the **sklearn** backend to ~1e-4. Significance is a paired permutation test on
the fold-level r values.

### Methods notes (for the paper)

- **Banded ridge for SFL+EMB.** A single global ridge α on the concatenation of the
  3 SFL scalars with the 768–1600 EMB dims is a bad compromise: to control the large
  EMB block it over-shrinks the 3 strong SFL coefficients, so `SFL+EMB` can score
  *below* SFL alone (e.g. Provo first-pass: SFL 0.866, single-α `all` 0.852). This is
  a regularization artefact — the concatenation cannot be worse than its best block
  under proper regularization — and per-fold standardization exposes it (raw features
  hide it via an *implicit* scale-based per-feature penalty). Fix: **banded ridge** —
  a separate penalty per feature block (Nunez-Elizalde 2019; Dupré la Tour 2022).
  We select the between-block penalty ratio on a deterministic grid by inner CV
  (`src/fit.py:_banded_fit_predict`), which restores `all ≥ max(SFL, EMB)` (Provo:
  0.879). Applied to `all` only; `sfl_rff_emb` is two same-width D-dim blocks (RFF vs
  EMB) with no size mismatch, so single-α is already monotone.
- **Why `SFL_RFF+EMB` looked "too good".** It never had the heterogeneity problem
  (both blocks are D-dim), so it was fine while `SFL+EMB` was artificially low. With
  banded ridge the `all` bars rise and the comparison is honest.
- **Sub-word / word pooling.** EMB is mean-pooled over sub-words (and over words for
  sentence/TR datasets). Surprisal is always summed over sub-words; length/frequency
  are unaffected.

Extraction is **ex novo**: `src/extract.py` runs the model for both embeddings
(mean over sub-word tokens) and surprisal (sum over sub-word tokens) every run,
caching to `cache/` (git-ignored, regenerable). No precomputed activations are used.

### Where this is written up (Overleaf: `6a56bd55cbb8310ac4c1e6db/`)

All manuscript additions are wrapped in `\new{...}` (defined in the preamble) so
they render **blue** for review. To accept every edit, redefine it as
`\newcommand{\new}[1]{#1}`.

| Change | Manuscript location |
|---|---|
| SFL_RFF analysis (concern → procedure → results, neural only) | SI subsection `sec:appendix-rff` ("Dimensionality control") |
| SFL_RFF figure (fMRI 2×2 + EEG, TikZ, mimics `figure-indep-contrib.tex`) | `figures/figure-rff.tex`, `fig:rff` |
| Qualitative observation: SFL_RFF and EMB predict from *different* features | last paragraph of `sec:appendix-rff` |
| "qualitative results hold under non-linear projections of SFL" | Results, brain subsection |
| Banded ridge for SFL+EMB | Methods → Analyses |
| Passage-level (GroupKFold) CV for Pereira | Methods → Analyses |

Figure assets are copied from `figures/paper_rff/` into the Overleaf tree with an
`-rff` suffix; the legend is generated by `figures/make_legend_rff.py` (colours must
stay in sync with `RFF3` in `figures/plot_results.py`).

**Qualitative observation (GPT-2-XL).** Regenerate with
`python inspect_predicted_extremes.py pereira control n400` (writes
`results_extremes/*_xl_{report.md,predictions.csv}`), then
`python check-features/analysis/shallow_feature_corr.py` → `SHALLOW_FEATURE_CORR.md`.
At comparable predictivity the two predictors' out-of-fold predictions agree only
moderately (Pereira 0.39, UCL-N400 0.46, Tuckute 0.56) and track different things:
SFL_RFF → sentence length (Pereira 0.75 vs EMB 0.25), word length/frequency (Tuckute
0.42 / −0.36 vs EMB 0.19 / −0.12); EMB → imageability (−0.53 vs −0.23) and physical
content (−0.48 vs −0.10).

> ⚠️ **Open: RFF pooling is SUM, not MEAN** (`src/brain.py:60`, `src/rff.py:99,143`) —
> inherited from the original scripts, while EMB and the SFL base are MEAN-pooled.
> Summing `D` bounded random features makes a sentence's SFL_RFF vector scale with its
> word count, so it encodes sentence length by construction. This *is* the Pereira
> exception: with GPT-2-XL, SFL_RFF sum-pooled scores r = 0.480 (length corr 0.753);
> mean-pooled it drops to r = 0.219 (length corr −0.065), i.e. below EMB (0.381).
> Sentence length alone correlates r = 0.381 with the Pereira response. Sum-pooling
> *inflates* the SFL_RFF baseline, so it is conservative w.r.t. the paper's claim — but
> the SI text must say "summed" (it did say "averaged"; corrected), and switching to
> mean-pooling would make EMB > SFL_RFF in 5/5 datasets and require a full RFF re-run.

Reported SI numbers (mean over the 5 models, `results/*_results.csv`):
EMB − SFL_RFF = +0.107 (NatStor-fMRI), +0.085 (Wehbe2014), +0.044 (Tuckute2024),
+0.006 (UCL-N400), **−0.092 (Pereira2018)**; mean over the four = 0.061.
SFL_RFF+EMB − SFL_RFF: mean 0.084; significant in 5/5 models for UCL-N400 (0.080),
Tuckute2024 (0.089), NatStor-fMRI (0.141); 1/5 for Wehbe2014 (0.082); **0/5 for
Pereira2018 (0.030)**.

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
