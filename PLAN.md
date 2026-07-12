# SFL-EMB refactor plan (pre-submission)

de Varda et al., *"Behavioral and Brain Responses to Language Reflect Different
Levels of Linguistic Representation."*

This document is the roadmap + record of decisions for cleaning, modernizing, and
extending the codebase ahead of submission. It is the source of truth; the paper
Methods and the top-level `README.md` should be kept consistent with it.
See `paper/PAPER_SUMMARY.md` for the analysis inventory (A1–A6 + RFF).

---

## 1. Goals

1. **Deep cleanup** — remove superseded/scratch code and dead data; organize the
   repo so it is easy to inspect and build on.
2. **Modernize** — drop the outdated `neurox` dependency; do embedding extraction
   and surprisal with vanilla HuggingFace `transformers`. Target the modern
   `analysis` conda env (sklearn 1.8, numpy 2.3, torch 2.9, working GPU).
3. **GPU-accelerated fitting** — add a `himalaya` backend (default) alongside
   `sklearn`, producing identical results up to float precision.
4. **Fold RFF into every dataset** — each dataset runs 5 conditions:
   `SFL, EMB, SFL+EMB (all), SFL_RFF, SFL_RFF+EMB`.
5. **Fix bugs / divergences from the paper.**
6. **Ready to re-run on Engaging (GPU).** The whole pipeline is re-run from scratch.

---

## 2. Confirmed decisions (2026-07-12)

| # | Decision |
|---|---|
| Fitting | **Standardize per fold**: per-fold `StandardScaler` on X and y, `fit_intercept=False`, inner alpha CV = `KFold(5, shuffle=False)`, outer = 10-fold **sequential** (unchanged). Replaces the published raw-features/`fit_intercept=True` behavior; makes himalaya == sklearn. Numbers shift slightly → update Methods. |
| Backend | **himalaya is default** (GPU), sklearn kept compatible. Pick `RidgeCV` vs `KernelRidgeCV` per fold by `n_samples` vs `n_features`. |
| RFF gamma | **Median heuristic**, uniform across **all** datasets incl. Pereira: `gamma = 1/(2·median pairwise sq-dist)` on standardized train features. |
| RFF dim | `n_components = D` (model hidden size). SFL_RFF matches EMB dim; SFL_RFF+EMB = 2D. Leakage-free (imputer/scaler/RBFSampler fit on TRAIN only). |
| Alpha grid | **Widened** to `logspace(-6, 6, 13)` for all conditions (SFL_RFF+EMB is very high-dim). Selected alpha logged per fold to detect boundary hits. |
| Pereira CV | **Keep passage-grouped GroupKFold** (leakage fix); document this single deviation from "sequential" in Methods. |
| check-features | **Keep + modernize** — part of the RFF story ("what does SFL→RFF encode — meaning vs form?"), not scratch. |
| Architecture | **Shared modules + convert** (not a full rewrite): preserve the proven dataset-specific wrangling. |
| Layout | **Moderate reorg**: `src/` shared modules, probes converted in place, `figures/`, `cluster/`, top-level `README.md`, superseded files removed. |
| Runbook | Implement + small-GPU-test locally; Andrea runs the full pipeline on Engaging. |

---

## 3. Target architecture

```
src/
  paths.py          # PROJECT_ROOT resolution (shared, replaces per-script headers)
  extract.py        # transformers: per-word embeddings (mean over sub-words) +
                    #   surprisal (sum over sub-words). Replaces neurox. GPU. Disk cache.
  fit.py            # fit_encoding(X, y, backend): per-fold StandardScaler,
                    #   fit_intercept=False, inner KFold(5), widened alpha grid;
                    #   returns (mean r, fold rs, fold alphas); saves per-fold r.
  rff.py            # median-heuristic gamma, leakage-free RBFSampler(D) SFL expansion.
  conditions.py     # given per-dataset X_sfl / X_emb (+ pooling/lag/group spec),
                    #   build & score the 5 conditions + permutation sig-tests.
  significance.py   # paired permutation test on fold-level r (from utils).
probes/  (repo root, converted in place)  # one thin script per dataset
figures/          # modernized plotting (A1 main, A2 decomp, etc.)
cluster/          # slurm scripts + ENGAGING.md + env spec
main_data/, datasets/, dfs/, corpora/     # curated inputs (kept)
<dataset dirs>/   # pereira_1/, control_fmri/, natstor_fmri/, fMRI/, ZuCo/ (kept)
results/, <dataset>/results/              # outputs
README.md, PLAN.md, paper/
```

`extract.py` is validated against the existing plain-transformers reference at
`natstor_fmri/get_natstor_embeddings.py` (mean-subword vectors, sum-subword
surprisal, tokenizer-separator word alignment: `</w>` for openai-gpt, `Ġ` for gpt2).

Word/last-layer indices per model: gpt=12, gpt2=12, medium=24, large=36, xl=48;
hidden dims 768/768/1024/1280/1600. Consumers drop the sentence-initial word.

---

## 4. Deletion / supersession list

**Delete (superseded or scratch; git-recoverable):**
- `probe_maze_natstor.py` (spill maze — paper uses no-spill; `_nospill` is canonical)
- the NatStor half of `probe_Brown_NatStor_spill.py` is superseded by
  `probe_NatStor_spill_realigned.py` (keep Brown; consolidate)
- `fMRI/fMRI_analysis_HP.py` (old per-subject), `fMRI/sentence-level/fMRI_analysis_HP.py`
  (shuffle-in-fold scratch), `fMRI/plot_fMRI.py` (old plotter)
- `analyze_res_probes.py` (reads stale `results_maze_spill.csv`, old `pereira/` path;
  superseded by `plotting.py`)
- `ZuCo/plot_sfl_decomposed.py` (stale paths; top-level version is canonical)
- `interpret_ERP_probe.py` (unreferenced ERP scratch; neurox/glmnet/researchpy)
- `probe_rffsym_sentence.py`, `probe_rffsym_tr.py` (symmetric-RFF exploratory; the
  median-heuristic RFF becomes the standard folded into every dataset)
- `other/split_half_corr_NEW.py` **or** `other/reliability_and_size.py` — keep one
  A5/A6 owner (decide during conversion; they overlap)
- `DF_spill_maze.csv`, `logs/`, `other/other/`
- `cc.en.300.bin`, `cc.en.300.bin.gz` (~11.7 GB; only used by dead `get_sem`)
- dead `get_sem`/`plot_sem`/`plot_weights_sem` defs in `utils.py` + their imports

**Keep:** `main_data/`, `datasets/`, `dfs/`, `corpora/`, all `control_fmri/`,
`pereira_1/`, `natstor_fmri/`, `ZuCo/`, `fMRI/` data, `check-features/`.
`AUDIT_iteration{1,2}.md`, `ENGAGING.md` → fold into `cluster/` / drop for submission.
`environment_neurox.yml` → replaced by a modern env spec.

---

## 5. Bugs / divergences to fix

1. `plotting.py:427` — `base_plot(res_mich, …)` runs but `res_mich` only defined on
   commented lines → `NameError`. (Michaelov isn't in the paper; remove the panel.)
2. `probe_ET.py:604-616` — N400/P600 significance loops compute `p_emb_base` but
   append the stale `p_base_emb` from the fpass loop. Wrong sig values for those panels.
3. `probe_HP_matched.py` & `probe_natstorfmri_matched.py` both write top-level
   `results/all_results.csv` → collision. Route to per-dataset dirs.
4. Alpha-grid deviations: `fMRI_analysis_HP*` use `0.01..1e5`; sentence-level HP uses
   `0.001..1000` + `np.random.shuffle(X_train)` in-fold. Unify on the widened grid,
   sequential folds.
5. Maze run-order: re-verify `probe_maze_natstor_nospill.py` has no module-level
   use-before-def after conversion (the flagged historical bug).

---

## 6. Per-dataset condition matrix

All 13 datasets × 5 models × 5 conditions (SFL, EMB, all, SFL_RFF, SFL_RFF+EMB),
plus the analysis-specific pooling/spillover:

- **Eye-tracking** (Provo, Meco, ZuCo, UCL_ET): word-level, no spillover.
- **SPR** (UCL_SPR, Brown, NatStor_SPR): word-level, spillover i, i-1, i-2 (block).
- **Maze** (NatStor_Maze): word-level, no spillover, Boyce exclusions.
- **EEG** (UCL_N400): word-level, no spillover, drop punctuation-adjacent.
- **fMRI TR** (Wehbe2014, NatStor_fMRI): pool words→TR, concat previous 4 TRs.
- **fMRI sentence** (Pereira, Tuckute): pool words→sentence; Pereira passage-grouped CV.

RFF is applied to the per-word SFL representation that feeds the SFL condition, then
pooled/lagged identically, so SFL_RFF is a drop-in higher-dim replacement for SFL.

---

## 7. Status

See the session todo list. Order: shared modules → local smoke test → cleanup/reorg
→ behavioral probes → brain probes → appendix + plotting → check-features → README/env/cluster.
