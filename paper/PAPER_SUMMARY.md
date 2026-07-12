# Paper summary (for code refactoring)

Read this instead of the LaTeX source. It lists every analysis in the paper, what output it
produces, and the numbers that were reported, so that the code can be checked against the paper.

Terminology used throughout:

- **SFL** = three scalar predictors per word: surprisal (negative log probability of the word
  given its left context, from a language model), log word frequency, word length in characters.
- **EMB** = the hidden state vector for the word, taken from the final layer of the same model.
  768 to 1600 dimensions depending on model size.
- **SFL+EMB** (called `all` in the code) = the two concatenated.

Main claim of the paper: SFL predicts reading behavior better; EMB predicts brain responses
better. Behavior is a compressed readout; the brain retains richer representational content.

---

## 1. Models

Five models, used both for surprisal and for the word vectors. Same model for both within any
given analysis.

| Label in paper | HuggingFace id | Hidden size |
|---|---|---|
| GPT | `openai-gpt` | 768 |
| GPT-2 124M | `gpt2` | 768 |
| GPT-2 355M | `gpt2-medium` | 1024 |
| GPT-2 774M | `gpt2-large` | 1280 |
| GPT-2 1.5B | `gpt2-xl` | 1600 |

Sub-word handling:
- Word vectors: **average** over the sub-word tokens of the word.
- Word surprisal: **sum** over the sub-word tokens of the word.

Layer: **final layer only** in the main analyses. The one exception is the layer sweep in
Appendix 4 below.

---

## 2. Datasets

13 datasets. Every one is reduced to a single number per data point, so all regressions are
univariate in the target. The `N` column is the number of data points **after** aggregation and
trimming, so it is a direct check that data loading is correct.

### Behavioral (8)

| Name | Measure | Resolution | Edited | N | Participants per point | Split-half reliability |
|---|---|---|---|---|---|---|
| Provo | eye-tracking | word | | 2,659 | 84.00 | 0.93 |
| Meco-en | eye-tracking | word | | 2,096 | 40.13 | 0.87 |
| ZuCo-2 | eye-tracking | word | | 6,461 | 17.44 | 0.76 |
| UCL_ET | eye-tracking | word | | 1,726 | 42.00 | 0.84 |
| UCL_SPR | self-paced reading | word | | 1,316 | 71.11 | 0.22 |
| Brown | self-paced reading | word | | 5,862 | 19.05 | 0.39 |
| NatStor_SPR | self-paced reading | word | yes | 8,779 | 59.63 | 0.70 |
| NatStor_Maze | maze | word | yes | 9,779 | 5.86 | 0.47 |

### Brain (5)

| Name | Measure | Resolution | Edited | N | Participants per point | Split-half reliability |
|---|---|---|---|---|---|---|
| UCL_N400 | EEG, N400 amplitude | word | | 1,505 | 24.00 | 0.22 |
| Wehbe2014 | fMRI BOLD | TR | | 1,211 | 8.00 | 0.16 |
| NatStor_fMRI | fMRI BOLD | TR | yes | 1,516 | 27.18 | 0.61 |
| Pereira2018 | fMRI BOLD | sentence | yes | 627 | 7.84 | 0.28 |
| Tuckute2024 | fMRI BOLD | sentence | | 2,000 | 6.25 | 0.22 |

### Dataset subsetting rules that the code must apply

- **ZuCo-2**: use only the natural reading portion (349 sentences). Do **not** use the
  task-specific portion (390 sentences), where participants searched for semantic relations.
- **UCL_SPR**: use only the 205 sentences shared with UCL_ET, not all 361.
- **Pereira2018**: Experiments 2 and 3 only. 627 sentences total (384 from Exp 2, 243 from Exp 3).
- **NatStor_fMRI**: 9 stories. 8 are the syntactically edited NatStor stories, 1 is an
  expository Wikipedia-derived text called "Tree".
- **NatStor_Maze** and **NatStor_SPR** are collected on the same 10 stories, so their stimulus
  side should be identical.

---

## 3. Preprocessing rules

### Behavioral
- Eye-tracking main measure: **first-pass gaze duration**. Unfixated words get a value of zero.
  Sentence-initial words are dropped. No other filtering.
- Self-paced reading: **spill-over matters**. Include predictors for words `i`, `i-1`, `i-2`.
- Maze and eye-tracking: **no spill-over**. High temporal resolution, negligible spill-over.
- Maze: apply the Boyce et al. (2023) exclusion criteria. Error-correction variant of the task.
- All behavioral responses are averaged across participants to get one value per word.

### EEG (UCL_N400)
- N400 = mean voltage across 12 centro-parietal electrodes, 300 to 500 ms post word onset.
  Electrode sites: 1, 14, 24, 25, 26, 29, 30, 31, 41, 42, 44, 45.
- Use the pre-computed word-by-word amplitudes; apply the exclusion criteria from the original paper.
- **Drop words adjacent to punctuation** (wrap-up effects).
- Average across participants.

### fMRI, story datasets (Wehbe2014, NatStor_fMRI)
- TR = 2 seconds.
- Predictors are downsampled to the TR:
  - Wehbe2014: fixed 0.5 s per word, so average over 4 words per TR.
  - NatStor_fMRI: audio presentation, so word timings come from whisper-timestamped. Average
    over however many words fell within the TR.
- Concatenate the predictors from the **previous 4 TRs** to cover the BOLD delay.

### fMRI, sentence datasets (Pereira2018, Tuckute2024)
- Predictors are averaged over the words in the sentence.

### fMRI response aggregation (all 4 datasets)
- Average over voxels within each region of interest, then across regions, then across
  participants. Result is one number per TR or per sentence.
- Pereira2018, Tuckute2024, NatStor_fMRI: **5 functionally defined language regions**, from a
  sentences > nonwords localizer. Top 10% of voxels by t-value within each parcel.
- Wehbe2014: **no localizer exists** for this dataset. Regions are anatomical, from the AAL
  atlas. Use these 7: Posterior Temporal, Anterior Temporal, Angular Gyrus, Inferior Frontal
  Gyrus, Middle Frontal Gyrus, Inferior Frontal Gyrus (orbital part), Posterior Cingulate.

---

## 4. Fitting procedure

This is the same for every analysis in the paper.

- Ridge regression.
- **10-fold cross-validation with sequential (contiguous) splits.** Not shuffled. Shuffled splits
  let the model exploit temporal autocorrelation in the signal and inflate scores. This is a
  correctness issue, not a style issue.
- Alpha is chosen **inside each training fold** by nested cross-validation. 10 candidates,
  log-spaced from 1e-5 to 1e4.
- Metric: Pearson r between predicted and actual, averaged across the 10 folds.
- Significance: paired permutation test on the fold-level r values. Permute the scores between
  the two conditions being compared, recompute the difference in mean r, enumerate all 2^n
  permutations. p = proportion of permuted differences at least as large as the observed one.

---

## 5. Analysis inventory

Each entry below is one analysis in the paper. The code should have a clear owner for each one.

### A1. Main analysis (Figure: results)
- **Scope**: all 13 datasets x all 5 models x 3 predictor sets (SFL, EMB, SFL+EMB).
- **Output**: a grid of bar plots, one panel per dataset, split into a behavioral panel (A) and
  a brain panel (B). Plus a bottom-left subpanel showing model size vs performance for behavior,
  and a bottom-right subpanel summarizing SFL vs EMB across datasets.
- **Reported values to check against**, as mean differences in r averaged across models:

  Behavioral:
  | Group | SFL minus EMB | (SFL+EMB) minus SFL |
  |---|---|---|
  | Eye-tracking (4 datasets) | +0.1102 | +0.0187 |
  | Self-paced reading (3 datasets) | +0.0360 | +0.0459 |
  | NatStor_Maze | -0.0437 | +0.0958 |

  Brain:
  | Dataset | SFL minus EMB | (SFL+EMB) minus EMB |
  |---|---|---|
  | Pereira2018 | -0.2822 | +0.0234 |
  | Tuckute2024 | -0.1105 | +0.0132 |
  | NatStor_fMRI | -0.0885 | -0.0399 |
  | Wehbe2014 | -0.0879 | -0.0032 |
  | UCL_N400 | -0.0954 | +0.0221 |

- **Expected qualitative pattern**: on behavior, SFL+EMB > SFL > EMB. On brain, EMB and SFL+EMB
  are close and both beat SFL. The EMB vs SFL comparison is **not** significant for Wehbe2014
  (high across-fold variance). It is significant elsewhere.
- **Expected model size pattern**: on behavior, GPT-2 124M is best on average for SFL and EMB
  (355M edges ahead on SFL+EMB). On brain, larger is better, and 124M is always worst.

### A2. SFL decomposition (Appendix, Figure: indep-contrib)
- **Scope**: all 13 datasets, **GPT-2 355M only**.
- **Feature subsets**: all 7 non-empty subsets of {S, F, L}. That is S, F, L, SF, FL, SL, SFL.
- **Spill-over note**: in the datasets that use spill-over (self-paced reading, fMRI), the
  features for `w_i, w_{i-1} ... w_{i-N}` are selected **jointly** as a block. You do not
  enumerate subsets across lags, only across the three feature types.
- **Expected pattern**: length dominates gaze duration. Surprisal dominates the brain datasets,
  except Wehbe2014, where frequency alone is slightly better. Length is the weakest brain
  predictor everywhere.

### A3. Extra fixation measures (Appendix, Figure: fix_measurements)
- **Scope**: the 4 eye-tracking datasets only. Rerun A1 with two additional targets:
  first fixation duration, and total fixation duration.
- **Expected pattern**: same as A1. SFL beats EMB; SFL+EMB adds little over SFL.

### A4. Layer sweep (Appendix, text only, no figure)
- **Scope**: behavioral data only, **GPT-2 355M only**, EMB vs SFL.
- **Procedure**: sweep every layer. Pick the best layer on one held-in dataset per measure type
  (UCL_ET for eye-tracking, UCL_SPR for self-paced reading), then evaluate that layer on the
  held-out datasets of that type.
- **Result found**: best eye-tracking layer = layer 3. Best self-paced reading layer = the last
  layer (which is already the main analysis, so no further evaluation was run for SPR).
- **Reported held-out eye-tracking values, layer 3**:
  | Dataset | r (SFL) | SD | r (EMB) | SD |
  |---|---|---|---|---|
  | Provo | 0.8641 | 0.0193 | 0.8539 | 0.0194 |
  | Meco-en | 0.8251 | 0.0301 | 0.8177 | 0.0271 |
  | ZuCo-2 | 0.7607 | 0.0264 | 0.7863 | 0.0208 |
- Note ZuCo-2 flips (EMB wins). Provo and Meco-en do not. This is reported honestly in the paper,
  so the code should reproduce the flip, not hide it.

### A5. Reliability by sample size (Appendix, Figure: reliab-size)
- **Scope**: all 13 datasets.
- **Procedure**: progressive split-half reliability. For each dataset, treat each item at the
  dataset's own resolution (word, TR, or sentence) as a list of per-participant measurements.
  For n from 2 up to half the maximum available: randomly split n observations per item into two
  halves, average within each half, correlate the two halves across items. Repeat 10 times, take
  mean and SD.
- The single reliability value in the dataset table above is the split-half reliability at the
  full sample size.

### A6. Table 1
- The dataset table above. Mostly descriptive, but `N` and reliability are computed from the data
  and should be reproducible.

---

## 6. Not in the paper yet, must be added to the codebase

The random Fourier features (RFF) work. This expands SFL into a higher-dimensional feature space
so it can capture nonlinear effects. It currently lives in its own separate folder. It should be
folded into the per-dataset scripts instead, so that every dataset runs the same 5 conditions:

1. SFL
2. EMB
3. SFL+EMB (`all`)
4. SFL_RFF
5. SFL_RFF + EMB

Condition 5 is very high dimensional and will likely need a **wider alpha sweep** than the
current 10 log-spaced candidates. Consider widening the sweep for all conditions, not just this one.