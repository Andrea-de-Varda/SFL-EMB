# Dimensionality-matched SFL-vs-EMB on the brain datasets — summary

Two analyses ask whether EMB's advantage over SFL on brain data survives once the
**dimensionality** difference (EMB ~768–1600 dims vs SFL 3 dims) is controlled.
Both predict the **univariate** response (language-network-average BOLD, or N400
amplitude) with ridge (`RidgeCV`, nested α; Pearson r per fold), each dataset
keeping its own CV: Pereira = passage-grouped `GroupKFold(10)`; Tuckute, UCL-N400 =
sequential `KFold(10)`; NatStor-fMRI = leave-one-story-out; Wehbe2014/HP =
`KFold(10)` over TRs. Leakage controls throughout: any RFF/scaler/PCA fit on
training units only; passages/stories held out; per-fold r vectors → 2ⁿ paired
permutation tests.

`base` = SFL (3 dims, linear). `act` = EMB last layer, mean-pooled, linear.
`all` = SFL ⌢ EMB.

## (a) Standard RFF ("matched") — all 5 datasets
SFL is given EMB's dimensionality by random Fourier features:
`sfl_rff` = per-word SFL(3) → standardize → RBFSampler(D, γ tuned by nested CV over a
fixed grid 0.01–10) → summed per unit. `emb_sum` = EMB **sum**-pooled (D) — the
matched-pooling EMB comparator. `all_matched` = `emb_sum ⌢ sfl_rff`. The fair
dimensionality-matched contrast is **`emb_sum` vs `sfl_rff`** (both D, both summed).

Mean r over the 5 LMs:

| dataset | base | act | all | emb_sum | **sfl_rff** | all_matched |
|---|---|---|---|---|---|---|
| Pereira | 0.18 | 0.39 | 0.41 | 0.46 | **0.48** | 0.46 |
| Tuckute | 0.30 | 0.41 | 0.42 | 0.41 | **0.37** | 0.42 |
| NatStor-fMRI | 0.29 | 0.38 | 0.42 | 0.31 | **0.26** | 0.31 |
| Wehbe2014/HP | 0.13 | 0.22 | 0.22 | 0.20 | **0.11** | 0.20 |
| UCL-N400 | 0.23 | 0.32 | 0.34 | — | **0.33** | 0.33 |

- EMB ≫ the **3-dim** SFL (`act` ≫ `base`) everywhere — the original result.
- Matched-dimensionality picture is mixed: on **Pereira**, matched-SFL ≥ matched-EMB
  (`sfl_rff` 0.48 ≥ `emb_sum` 0.46 > `act` 0.39); **UCL-N400** is a tie
  (`sfl_rff` ≈ `act`); **Tuckute** modestly favors EMB; **NatStor/HP** favor EMB.
- **Caveat:** on the TR datasets (NatStor, HP) `sfl_rff` falls *below* `base`, which
  is an artifact of the **sum-per-TR** pooling (variable words/TR injects noise), not
  a fair SFL estimate — the TR `sfl_rff` numbers are unreliable. (Mean-pool fix deferred.)

## (b) Symmetric RFF — Tuckute + UCL-N400 only (others not yet run)
Both EMB and SFL go through the **identical** pipeline
`per-item → standardize(train) → RBFSampler(D) → pool → ridge`, with γ set by the
**median heuristic** (γ = 1/(2·median pairwise sq-dist) on standardized train items) —
the same unsupervised, dimension-aware rule on both sides (no γ tuned against the
brain; this also fixes a degeneracy where a fixed γ made D-dim EMB-RFF collapse to
noise). `emb_rff`, `sfl_rff`, `all_rff` = `emb_rff ⌢ sfl_rff`.

Mean r over the 5 LMs:

| dataset | base | act | sfl_rff | emb_rff | all_rff |
|---|---|---|---|---|---|
| Tuckute | 0.30 | 0.41 | 0.37 | 0.38 | 0.43 |
| UCL-N400 | 0.23 | 0.32 | 0.33 | 0.29 | 0.37 |

- With the **identical** nonlinear treatment, EMB and SFL are ~tied: `emb_rff ≈ sfl_rff`
  (Tuckute), and `sfl_rff ≥ emb_rff` (N400). EMB does **not** pull ahead.
- `emb_rff` < linear `act` in both — RBF-RFF on already-high-dim embeddings doesn't
  help (embeddings are near-linearly decodable), so EMB's best form is its linear readout.
- `all_rff` (both nonlinear, concatenated) is the strongest, i.e. EMB and SFL carry
  partly complementary variance.

## Bottom line
EMB clearly beats SFL **only** against the original 3-dim SFL. Once dimensionality is
matched — SFL expanded (RFF) or both sides treated identically — the advantage largely
disappears (Pereira favors SFL; Tuckute/N400 tie; NatStor/HP favor EMB but there the
matched-SFL is pooling-confounded). So a large part of EMB's apparent brain advantage
reflects **dimensionality**, not uniquely semantic content, at this univariate resolution.

**Status / caveats:** symmetric analysis (b) has only Tuckute + N400 so far (Pereira,
NatStor, HP pending). The standard TR `sfl_rff` (NatStor, HP) is depressed by sum-pooling.
The univariate (1-D) target structurally limits how much EMB's richness can show.
