"""Run the five predictor conditions for one dataset x one model, with sig-tests.

The five conditions (confirmed 2026-07, see PLAN.md):
  1. ``sfl``          - the SFL design matrix (surprisal, log-freq, length; incl.
                        any dataset-specific spillover/lag already baked in).
  2. ``emb``          - the EMB design matrix (final-layer word vectors, pooled/
                        lagged as the dataset requires).
  3. ``all``          - hstack(sfl, emb).
  4. ``sfl_rff``      - RFF(sfl) with n_components = D (model hidden size).
  5. ``sfl_rff_emb``  - hstack(RFF(sfl), emb).

This module covers datasets whose modelling unit IS the row of the SFL/EMB design
matrices (all behavioral datasets + UCL_N400). The four fMRI datasets fit the RFF
at the per-word level and pool afterwards - they call ``fit.score`` with a custom
``build_fn`` built from :mod:`rff` pooling helpers, and reuse :func:`significance_row`.
"""
import numpy as np

from . import fit, rff, significance

CONDITIONS = ["sfl", "emb", "all", "sfl_rff", "sfl_rff_emb"]

# canonical directed contrasts: p = P(mean r of A >= mean r of B) by permutation.
CONTRASTS = {
    "p_sfl_emb":        ("sfl", "emb"),          # behavior: SFL wins; brain: EMB wins
    "p_all_sfl":        ("all", "sfl"),
    "p_all_emb":        ("all", "emb"),
    "p_sflrff_sfl":     ("sfl_rff", "sfl"),      # does the RFF expansion help at all
    "p_sflrff_emb":     ("sfl_rff", "emb"),      # dimensionality-matched SFL vs EMB
    "p_sflrffemb_all":  ("sfl_rff_emb", "all"),
    "p_sflrffemb_emb":  ("sfl_rff_emb", "emb"),
}


def significance_row(prefix, results_dir="results", contrasts=CONTRASTS):
    """Compute the permutation p-values for all contrasts of a saved run."""
    return {name: significance.significance_testing(
                f"{prefix}_{a}", f"{prefix}_{b}", results_dir)
            for name, (a, b) in contrasts.items()}


def build_results_table(rows_by_model, model_order=None):
    """Assemble a per-model results DataFrame from ``{model_key: row_dict}``.

    Columns are ordered: each condition's r then sd, then the p-value contrasts.
    Index is the model key in ``model_order`` (defaults to MODEL_ORDER).
    """
    import pandas as pd
    from .models import MODEL_ORDER

    order = model_order or [m for m in MODEL_ORDER if m in rows_by_model]
    cols = ([c for cond in CONDITIONS for c in (cond, f"{cond}_sd")]
            + list(CONTRASTS.keys()))
    df = pd.DataFrame({m: rows_by_model[m] for m in order}).T
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.index.name = "model"
    return df


def run_conditions(specs, y, folds, prefix, results_dir="results",
                   backend="himalaya", verbose=True):
    """Score an ordered dict of conditions and return a flat result row.

    ``specs`` maps condition name -> kwargs for ``fit.score`` (either ``X=matrix``
    for a fixed design or ``build_fn=fn`` for a per-fold builder, e.g. pooled RFF).
    Returns ``{cond}``/``{cond}_sd`` per condition plus the ``p_*`` contrasts.
    Per-fold r vectors are persisted for the permutation tests / re-analysis.
    """
    row = {}
    for cond, kw in specs.items():
        m, sd, _, _ = fit.score(y=y, folds=folds, saveto=f"{prefix}_{cond}",
                                results_dir=results_dir, backend=backend, **kw)
        row[cond], row[f"{cond}_sd"] = m, sd
        if verbose:
            print(f"  {prefix:<30s} {cond:<12s} r = {m:.4f}  (sd {sd:.4f})")
    row.update(significance_row(prefix, results_dir))
    return row


def run_all_conditions(X_sfl, X_emb, y, folds, prefix, n_components=None,
                       results_dir="results", backend="himalaya", seed=0,
                       verbose=True):
    """Score the 5 conditions for a per-unit (word/sentence-row) dataset.

    RFF is applied directly to the rows of ``X_sfl`` (the modelling unit IS the
    row). Datasets that fit RFF at the word level and pool afterwards (fMRI) use
    :func:`run_conditions` with pooled build_fns instead.
    """
    X_sfl = np.asarray(X_sfl, dtype=np.float64)
    X_emb = np.asarray(X_emb, dtype=np.float64)
    if n_components is None:
        n_components = X_emb.shape[1]
    specs = {
        "sfl":         dict(X=X_sfl),
        "emb":         dict(X=X_emb),
        "all":         dict(X=np.hstack([X_sfl, X_emb])),
        "sfl_rff":     dict(build_fn=rff.rff_build_fn(X_sfl, n_components, seed)),
        "sfl_rff_emb": dict(build_fn=rff.rff_concat_build_fn(X_sfl, X_emb, n_components, seed)),
    }
    return run_conditions(specs, y, folds, prefix, results_dir, backend, verbose)
