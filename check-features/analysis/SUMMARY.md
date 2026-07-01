# What does the SFL→RFF expansion encode — meaning or form?

Three questions, one theme: when we give the 3-dim SFL predictor EMB's dimensionality
via random Fourier features (`SFL+RFF`), does it start encoding **meaning**, or only
re-express **surface form**? And is EMB's content more semantic than `SFL+RFF` even
when the two reach similar brain predictivity?

Predictors (identical ridge pipeline throughout — RidgeCV, 10-fold, per-fold Pearson
*r*, paired permutation tests, 5 LMs GPT…GPT-2-XL):
`SFL` = surprisal/length/frequency (3-dim, linear) · `SFL+RFF` = SFL→RBFSampler(D) ·
`EMB` = LM last layer (D; mean-pooled for the feature probe, **sum**-pooled for the
brain-summary table as requested).

---

## 1 · Predicting 14 sentence features → `feature_predictivity.svg`

Mean *r* over 5 LMs (`results_features.csv`):

| feature group | SFL | SFL+RFF | EMB |
|---|---|---|---|
| **Semantic** (imageability, physical, places, others'-thoughts, valence, arousal, sense) | 0.23 | 0.34 | **0.71** |
| **Form / syntax** (word length/freq, PCFG, grammaticality, conversational, dep length, rated freq) | 0.59 | 0.62 | 0.66 |

- On **semantic** features EMB dominates (0.71 vs 0.34); on **form** features everyone is
  close and EMB has no edge (SFL/SFL+RFF even win on word length & word frequency).
- RFF's gain over raw SFL is robust (5/5 LMs) **only** for the concreteness/perceptual
  cluster — imageability 0.15→0.45, physical 0.17→0.43, places 0.07→0.23 — and **absent**
  for abstract/affective meaning (valence 0.16→0.15, others'-thoughts 0.31→0.31).

## 2 · Brain-encoding summary, all datasets × predictors → `brain_summary.svg`, `brain_summary_*.csv`

`EMB` here = **sum**-pooled (`emb_sum`), as requested. Per-model rows exist for **Pereira**
and **UCL-N400**; **Tuckute / NatStor-fMRI / Wehbe-HP** are shown at the dataset-mean level
(hatched bars, `*`) because their per-model matched CSVs were produced on the cluster but
**not committed** — to complete them: on Engaging `git add results/results_{control,natstorfmri,HP}_matched.csv`,
push, rerun `compile_brain_summary.py`.

| dataset | SFL | SFL+RFF | EMB(sum) | source |
|---|---|---|---|---|
| Pereira | 0.18 | 0.48 | 0.46 | per-model |
| Tuckute | 0.30 | 0.37 | 0.41 | dataset-mean * |
| NatStor-fMRI | 0.29 | 0.26 | 0.31 | dataset-mean * |
| Wehbe-HP | 0.13 | 0.11 | 0.20 | dataset-mean * |
| UCL-N400 | 0.23 | 0.33 | 0.33 | per-model |

(NatStor/HP are TR-level; their `SFL+RFF`/`EMB_sum` are depressed by sum-per-TR pooling —
see `RESULTS_dimensionality_matched.md`. Predictivity of EMB and SFL+RFF is otherwise close.)

## 3 · Error / difference patterns → `ERROR_PATTERNS.md`

I scored every item by disagreement `d = z(EMB_sum pred) − z(SFL+RFF pred)` and inspected
the extremes. The two predictors reach similar accuracy but **diverge on interpretable,
meaning-vs-form lines**:

**Tuckute (has human ratings → quantitative).** EMB-favored vs SFL+RFF-favored sentences
(top vs bottom 15% of `d`, rated subset n=1000):

| rating | EMB-favored | SFL+RFF-favored | Δ |
|---|---|---|---|
| physical | 2.48 | 3.56 | **−1.08** |
| places | 1.84 | 2.63 | **−0.79** |
| imageability | 3.52 | 4.15 | −0.63 |
| others' thoughts (mentalizing) | 4.10 | 3.52 | **+0.59** |
| avg word length | 4.19 | 4.91 | −0.72 |
| word frequency (zipf) | 5.86 | 5.40 | +0.45 |

→ **SFL+RFF inflates the response for sentences with rarer/longer words and concrete
object/place content** (e.g. *"Long chandelier avocado colored earrings glistened."*,
*"Australias Dillon authoritarian shopping department Nab."* — word-salad with long,
low-frequency words). **EMB inflates it for coherent, more abstract/mentalistic
sentences** (higher "others' thoughts", "makes sense", grammaticality).

**Pereira.** `corr(d, sentence length) = −0.22`: SFL+RFF-favored sentences are longer
(13.6 vs 11.7 words) and are concrete-definitional (*"A banana is a long fruit…"*,
*"A raven is a large black bird…"*, *"A desk is a piece of furniture…"*); EMB-favored are
shorter, more abstract/process sentences (*"Every night while we sleep, our minds are as
active…"*, *"Murder is the unlawful killing…"*).

**UCL-N400.** SFL+RFF over-predicts N400 for short, high-frequency function words
(*"an", "a", "off", "single", "first"*) whose **actual** N400 is small (y ≈ −1 to −1.9) —
context-blind lexical errors; EMB (contextual) favors content words. SFL+RFF wins overall
r (0.35 vs 0.30) but makes these frequency-driven mistakes at the extremes.

**One-line pattern:** SFL+RFF's predictions are governed by **lexical surface statistics**
(length, frequency, surprisal accumulation); EMB's by **coherence/meaning**. This is the
same axis the feature probe shows, now visible in the brain over/under-predictions.

---

## Verdict on the theory
- **(a) EMB is more about meaning than SFL+RFF, even at similar predictivity — strongly
  confirmed.** Feature side: EMB ≫ SFL+RFF on every semantic rating (0.71 vs 0.34), ties on
  form. Brain side: at matched predictivity the two rank items differently (§3), splitting
  cleanly along meaning (EMB) vs surface form (SFL+RFF).
- **(b) RFF makes SFL encode some meaning far more than raw SFL — confirmed, but
  selectively.** Only the concreteness/perceptual semantics (imageability/physical/places),
  which are themselves correlates of lexical surface form; not abstract/affective meaning.

**Net:** matching dimensionality equalizes *predictivity* without equalizing *content* —
EMB uniquely carries abstract meaning; SFL+RFF re-expresses form and picks up only the
form-correlated slice of semantics.

## Files
- `make_figures.py` → `plots/feature_predictivity.{svg,png}` (Fig 1, side-by-side bars + group means).
- `compile_brain_summary.py` → `brain_summary_permodel.csv`, `brain_summary_bydataset.csv`,
  `plots/brain_summary.{svg,png}`.
- `error_patterns.py` → `ERROR_PATTERNS.md` (full item lists + the Tuckute rating table).
