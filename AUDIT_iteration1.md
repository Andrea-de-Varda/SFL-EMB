# SFL-EMB codebase audit — iteration 1

Audit of the codebase supporting de Varda et al., *"Behavioral and Brain Responses
to Language Reflect Different Levels of Linguistic Representation"* (EMNLP format).

Scope of this iteration: (1) familiarization, (2) missing-data inventory,
(3) `.gitignore` for large files, (4) deletion candidates.
Deferred to a later iteration: (5) bugs/vulnerabilities, incl. the Maze script
run-order issue.

---

## 1. What the project is

A single-pipeline comparison of two predictor families against human language data:

- **SFL** = surprisal + log-frequency + length (from GPT / GPT-2 small→xl)
- **EMB** = final-layer LM embeddings of the same token
- **SFL⌢EMB** = concatenation

Modeled with **ridge regression, 10-fold sequential CV, nested α selection**,
scored by Pearson *r*, significance by paired permutation tests. Core machinery
lives in [utils.py](utils.py) (`test_model_Ridge`, `significance_testing`,
`decompose_sfl`, `probe`, …); every `probe_*.py` is one dataset wired to those helpers.

**Datasets in the current paper (Table 1):** Provo, Meco-en, ZuCo-2, UCL (ET/SPR/N400),
Brown, NatStor (SPR/Maze/fMRI), Wehbe2014 (HP fMRI), Pereira2018, Tuckute2024 (`control_fmri`).

---

## 2. Path problems & missing data

### 2a. Project-root rename (mechanical — data IS present, just renamed)

Every script hardcodes, near the top:

```python
dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)
```

The project now lives at `/home/andrea/Documents/PhD/SFL-EMB`. Two substitutions
fix all *internal* references (do these once we've confirmed the data layout):

| Old prefix | New prefix |
|---|---|
| `/home/dev/Documents/PhD/MIT/project` | `/home/andrea/Documents/PhD/SFL-EMB` |
| `/home/dev/` (home dir) | `/home/andrea/` |

After this rename, all in-repo data resolves (activations JSON, `SFL.csv`,
`pereira_averaged.csv`, `fMRI/fMRI/*.npy`, etc.).

### 2b. MISSING external raw datasets (you said you have these somewhere)

These live under `/home/dev/Documents/Datasets/` (+ a few other external mounts) and
are **not** in the repo. They are the raw corpora the probe scripts read first.
Grouped by paper dataset:

| Paper dataset | Missing file(s) | Old path |
|---|---|---|
| UCL ET + SPR | `eyetracking.RT.txt`, `selfpacedreading.RT.txt`, `stimuli.txt` | `…/Datasets/UCL_ET_Corpus/` |
| Provo | `Provo_Corpus-Eyetracking_Data.csv`, `sentences.txt` | `…/Datasets/Provo/` |
| Meco-en | `Meco_l1.csv` | `/home/dev/Documents/PhD/side-projects/eye-tracking/` |
| ZuCo-2 | `ET_data.csv`, `ET_data_aggregated.csv`, `Matlab/` (raw `.mat`) | `…/Datasets/ZuCo/` |
| Brown | `bnc-brown.csv`, `brown_spr_RAW.csv` | `…/Datasets/Brown/` |
| NatStor SPR | `NaturalStories_RAW.tsv`, `SPR/natural-stories.csv` | `…/Datasets/NaturalStories/` |
| NatStor Maze | `natstor_maze.rds` | `…/Datasets/NaturalStories/` |
| UCL N400 (ERP) | `stimuli_erp.mat` | `…/Datasets/ERP_Frank/` |
| Pereira2018 | `data/` + `M09/data_384sentences.mat` | `…/Datasets/Pereira/data/` **and** `/media/drive2/Datasets/Pereira/data/` (note: a 2nd mount point) |
| NatStor fMRI | `transcribed/{lang}.csv` (the dir `natstor_fmri/transcribed/` is **empty**) | repo-relative |

> Note: some *aggregated* products of these raw datasets are already in the repo
> (`pereira_1/pereira_averaged.csv`, `control_fmri/data/brain-lang-data_*.csv`,
> the `activations_*.json` dumps), so you only need the raw files to **regenerate
> from scratch**, not to reproduce the final figures.

### 2c. Missing data for analyses NOT in the current paper (lower priority)

Used by `extrapolate_semantic_norms.py`, the `SEM/` analyses, and `probing_LMs/`
— all part of an **earlier, broader version** of the project that is not in the
current `emnlp2023.tex` (no SEM / semantic-norms / probing-task section):

- Lexical norms: `subtlex.xlsx`, `concreteness.xlsx`,
  `affective-norms_Warriner_et_al_2013.csv`, `sensorimotor_Lynott-et-al_2020.csv`,
  `socialness.csv`, `semantic_richness_buchanan.csv`, `affective_bestworse.txt`,
  `context_sent_availability.csv`
- `/data/embedding/glove.840B.300d.h5`, `/data/probing/`, `/data/senteval_data/`,
  `/data/models` (SentEval probing pipeline)
- `michaelov_n400/N400_data.csv` (Michaelov N400 — present as derived
  `michaelov_SFL.csv`/`results_michaelov.csv`, but **not** in the current Table 1)

---

## 3. `.gitignore` (created)

[.gitignore](.gitignore) excludes **~114 GB across 140 files** while keeping all
code and small results tracked. Verified: every >10 MB file is matched; small
`results_*.csv` / coefs / predictions stay tracked. **Projected tracked size ≈ 311 MB.**

Excluded, by class:
- **(A) regenerable:** `activations_*.json`, `*.parquet.gzip`, embedding/surprisal
  `*.pkl`, `natstor_fmri/embeddings/`, `probing_LMs/{embeddings,XYdata}/`,
  `cc.en.300.bin*` (fastText).
- **(B) raw data — back up externally, do NOT delete:** `fMRI/fMRI/data_subject_*.npy`,
  `*.wav` + `sound_data/`, `NaturalStories_TimeSeries_20220615.csv`,
  `brain-lang-data_participant_20230728.csv`, `*.rds`.

> 311 MB is committable but on the heavy side for GitHub. If you want it leaner,
> the `results/` pickle coefs (~65 MB) could move to git-lfs or be regenerated.
>
> **Heads-up:** `probing_LMs/SentEval/` is a vendored third-party repo with its own
> nested `.git/` — also `.gitignore`d. See §4.

---

## 4. Deletion candidates

### Safe to delete (superseded / scratch / backups — none referenced by the paper)

| Path | Why |
|---|---|
| `old/` (whole dir) | superseded scripts: `probe_ET_nonlinear*`, `probe_geco`, `probe_DUNDEE`, `probe_PetitePrince`, old results |
| `temp_.py` | scratch file |
| `probe_Brown_NatStor.py` | superseded by `probe_Brown_NatStor_spill.py` |
| `probe_Boyce.py` | superseded by `probe_Boyce_new.py` |
| `other/code_backup_24_8/` | dated code backup (Aug 2024) |
| `results_epnp_UCL (copy).csv` | literal "(copy)" duplicate |
| `ipython.html`, `output.html` | notebook HTML dumps (regenerable; also git-ignored) |
| `__pycache__/` | bytecode cache |

### Flag — NOT in current paper, but confirm before deleting (may be other paper / supplement)

| Path | Note |
|---|---|
| `SEM/` + `plot_sem.py` + `extrapolate_semantic_norms.py` | structural-equation-modeling + semantic-norm analyses, cut from current tex |
| `probing_LMs/` (incl. vendored `SentEval/`, `CompLex/`, `PreTens/`) | layer-probing study, separate from this paper |
| `probe_Michaelov_n400.py` + `michaelov_*` | Michaelov N400 — not in Table 1 |
| `pereira/` vs `pereira_1/` | `pereira_1/` (aggregated) is the version used in the paper; `pereira/` (per-ROI) looks superseded |
| `dendrogram.pdf` | not referenced by current tex |
| `paper/SEM/`, `paper/neurofig/` | figures from a previous revision (main `emnlp2023.tex` doesn't `\input` them) |

### Minor: paper won't compile as-is

`paper/emnlp2023.tex` does `\input{emnlp2023-latex/img}`, `\input{emnlp2023-latex/figures/figure}`,
etc., but those files sit **flat** in `paper/` (no `emnlp2023-latex/` subdir). The
`\input` paths need fixing (or a symlink) before a clean build.

---

## 5. Deferred to next iteration

- **Maze run-order bug** you flagged: queued. Candidates to inspect are
  `probe_maze_natstor.py`, `probe_maze_natstor_nospill.py`, `probe_Boyce_new.py`
  (Maze branch). Typical cause of "had to run lines out of order" = a name used
  before assignment, or a dataframe column overwritten then re-read.
- General bug / vulnerability pass.
