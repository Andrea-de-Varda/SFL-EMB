# What does the SFL→RFF expansion encode — meaning or form?

Two probes, one question: when we give the 3-dim SFL predictor EMB's dimensionality
via random Fourier features (`sfl_rff`), does it start encoding **meaning**, or does it
just re-express **surface form**? And is EMB's content more semantic than `sfl_rff`
even when the two reach similar predictivity?

Predictors (identical ridge pipeline throughout — RidgeCV, 10-fold, per-fold Pearson
*r*, paired permutation tests, 5 LMs GPT…GPT-2-XL):
`SFL` = surprisal/length/frequency (3-dim, linear) · `SFL+RFF` = SFL→RBFSampler(D) ·
`EMB` = LM last layer, mean-pooled (D).

---

## Headline numbers

**Feature prediction** (`results_features.csv`; mean *r* over 14 features × 5 LMs):

| feature group | SFL | SFL+RFF | EMB |
|---|---|---|---|
| **Semantic** (imageability, physical, places, others'-thoughts, valence, arousal, sense) | 0.234 | 0.341 | **0.705** |
| **Form / syntax** (word length, word freq, PCFG, grammaticality, conversational, dep length, rated freq) | 0.589 | 0.620 | 0.664 |

- On **semantic** features EMB dominates (0.71 vs 0.34); on **form** features everyone is
  close and EMB has no real edge (SFL/SFL+RFF even win on word length & word frequency).
- So **EMB's advantage is essentially all on meaning**, not form. → `feature_predictivity.svg`

**Brain over/under-prediction** (`results_extremes/`, GPT-2-XL): EMB and SFL+RFF reach
**comparable** brain predictivity (Pereira 0.38/0.48, Tuckute 0.44/0.35, N400 0.29/0.35),
yet their predicted-response rankings agree only ρ = 0.37–0.57 — versus ρ = 0.90–0.98
between the two EMB poolings. Same predictivity, **different representational content**.
And SFL+RFF's predictions are far more tied to surface surprisal/length/freq than EMB's
(e.g. Tuckute surprisal-tracking 0.75 vs 0.54; N400 −0.59 vs −0.31). → `brain_agreement.svg`

---

## Verdict on the theory

### (a) "EMB is more about meaning than SFL+RFF, even at similar predictivity" — **strongly confirmed**
- Direct evidence: across all 7 semantic ratings EMB predicts far better (r 0.48–0.82) than
  SFL+RFF (0.15–0.56); on form features the gap closes or reverses.
- Brain-side corroboration: at matched predictivity the two predictors rank sentences/words
  differently (low ρ), and SFL+RFF leans much harder on surface lexical statistics. EMB's
  brain signal, though only modestly ahead once dimensionality is matched, is carried by
  **qualitatively different, semantic** content.

### (b) "With RFF, SFL+RFF encodes some meaning features far more than raw SFL" — **confirmed, but selectively**
- Large, robust gains (5/5 LMs, p<.05) appear **only** for the *concreteness/perceptual* cluster:
  imageability 0.15→0.45, physical 0.17→0.43, places 0.07→0.23 (+ small robust arousal).
- **No** gain for the *abstract/affective* cluster: valence 0.16→0.15, others'-thoughts
  0.31→0.31, sense ≈ flat. → `rff_gain_dumbbell.svg`
- Reading: the RFF nonlinearity recovers exactly the "semantic" dimensions that are
  **correlates of lexical surface form** (concrete words are shorter, more frequent, more
  predictable). It does **not** recover genuinely conceptual or affective meaning.

### Net interpretation
Both hypotheses hold, and together they sharpen the dimensionality-matched story: EMB
encodes real meaning (all semantic ratings, low surface dependence), whereas SFL+RFF's
apparent brain competitiveness is a **nonlinear re-expression of surface form** that only
picks up the form-correlated slice of semantics (concreteness), not abstract meaning. So
matching dimensionality equalizes *predictivity* without equalizing *content* — which is
the more precise claim than "EMB's advantage is just dimensionality."

**Caveat.** `SFL+RFF` includes surprisal, which itself carries some contextual-semantic
signal; and the semantic ratings it recovers are the ones most confounded with lexical
form. If anything that reinforces (a): strip the form-correlated semantics and little
"meaning" is left in SFL+RFF.

---

## Files
- `make_figures.py` — regenerates the three figures from `results_features.csv` +
  `results_extremes/*_predictions.csv`.
- `plots/feature_predictivity.{svg,png}` — Fig 1 (headline: EMB≫ on semantics, tie on form).
- `plots/rff_gain_dumbbell.{svg,png}` — Fig 2 (RFF gain over SFL; concentrated in concreteness).
- `plots/brain_agreement.{svg,png}` — Fig 3 (comparable predictivity, divergent content).
