"""Drivers for the fMRI datasets (sentence-level and TR-level).

These differ from the behavioral/EEG datasets in that the modelling unit (sentence
or 2 s TR) aggregates several words, so the RFF is fit at the WORD level on the
training fold and then pooled to units — the pooling therefore happens inside each
fold. Everything is recomputed from the sentences with the shared extractor (no
neurox JSON / precomputed SFL.csv needed); the brain response y is loaded by the
dataset script.

Pooling convention (from the original scripts): EMB and the SFL `base` predictor
are MEAN-pooled per unit; the RFF predictor is SUM-pooled (then, for TR datasets,
concatenated with 4 lagged TR blocks for the BOLD delay).
"""
from pathlib import Path

import numpy as np

from . import extract, fit, rff, conditions, behavioral, decompose
from .models import MODELS


def _wsi(sentences):
    """Sentence index for each non-initial word, matching extract_words flattening."""
    return np.array([i for i, s in enumerate(sentences) for _ in list(s)[1:]])


# ============================================================================ #
# sentence-level (Pereira, Tuckute)
# ============================================================================ #
def run_sentence_dataset(dataset, sentences, y, models, results_dir, cache_dir,
                         groups=None, backend="himalaya", seed=0, chunked=False):
    """5 conditions for a sentence-level fMRI dataset.

    ``groups`` (per-sentence labels) -> GroupKFold (Pereira passages); else 10-fold
    sequential (Tuckute). EMB/SFL base mean-pooled per sentence; SFL_RFF sum-pooled.
    """
    y = np.asarray(y, dtype=float)
    n = len(sentences)
    assert len(y) == n, f"{dataset}: {len(y)} responses vs {n} sentences"
    wsi = _wsi(sentences)
    freqs, lens = behavioral.freq_length(sentences)
    units = np.arange(n)
    folds = fit.make_folds(n, groups=groups, n_splits=10)

    rows = {}
    for model in models:
        emb_w, surp_w = extract.extract_words(
            model, sentences, drop_first=True, chunked=chunked,
            cache=Path(cache_dir) / f"{dataset}_{model}.npz")
        word_sfl = np.column_stack([surp_w, freqs, lens])
        assert len(word_sfl) == len(wsi)
        D = MODELS[model]["dim"]

        sfl_base = rff.pool_words_to_units(word_sfl, wsi, units, "mean")   # (n, 3)
        emb_mean = rff.pool_words_to_units(emb_w, wsi, units, "mean")      # (n, D)

        def rff_build(train, test, _ws=word_sfl, _wsi=wsi, _u=units, _D=D):
            mask = np.isin(_wsi, _u[train])
            t = rff.fit_rff(_ws[mask], _D, seed)
            pooled = rff.pool_words_to_units(t(_ws), _wsi, _u, "sum")      # (n, D)
            return pooled[train], pooled[test]

        def rffemb_build(train, test, _e=emb_mean, _rb=rff_build):
            a, b = _rb(train, test)
            return np.hstack([a, _e[train]]), np.hstack([b, _e[test]])

        specs = {
            "sfl":         dict(X=sfl_base),
            "emb":         dict(X=emb_mean),
            "all":         dict(X=np.hstack([sfl_base, emb_mean])),
            "sfl_rff":     dict(build_fn=rff_build),
            "sfl_rff_emb": dict(build_fn=rffemb_build),
        }
        rows[model] = conditions.run_conditions(specs, y, folds, f"{dataset}_{model}",
                                                results_dir, backend)
        decompose.maybe_decompose(model, sfl_base, y, folds, f"{dataset}_{model}",
                                  n_blocks=1, results_dir=results_dir, backend=backend)
    conditions.build_results_table(rows).to_csv(Path(results_dir) / f"{dataset}_results.csv")
    return rows


# ============================================================================ #
# TR-level (Wehbe2014, NatStor_fMRI)
# ============================================================================ #
def run_tr_dataset(dataset, stories, models, results_dir, cache_dir,
                   loso=False, backend="himalaya", seed=0, delays=(1, 2, 3, 4)):
    """5 conditions for a TR-level fMRI dataset.

    ``stories`` is a list of per-story dicts, each with:
        words   : list[str]            (whitespace words, first word kept for context)
        word_tr : np.ndarray[int]      TR index for each NON-initial word
        n_tr    : int                  number of TRs in the story
        y       : np.ndarray           brain response (len == n_tr, or == len(keep))
        keep    : np.ndarray[int]      OPTIONAL rows to retain AFTER lagging (Wehbe's
                                       per-run [20:-15] trim); defaults to all TRs.
    Words are pooled to TRs (mean for EMB/SFL base, sum for RFF) then concatenated
    with ``delays`` lagged TR blocks, WITHIN each story, before stories are
    concatenated. ``loso=True`` -> leave-one-story-out folds (NatStor); else 10-fold
    sequential over the concatenated TRs (Wehbe).
    """
    ntr = [s["n_tr"] for s in stories]
    keeps = [np.asarray(s.get("keep", np.arange(s["n_tr"])), int) for s in stories]
    y = np.concatenate([np.asarray(s["y"], float) for s in stories])
    bounds = np.cumsum([0] + [len(k) for k in keeps])

    def _agg(word_feats, wtr, n_tr, keep, pooling):
        return rff.aggregate_to_tr(word_feats, wtr, n_tr, pooling, delays)[keep]
    if loso:
        folds = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(stories))]
    else:
        folds = fit.make_folds(len(y), n_splits=10)

    rows = {}
    for model in models:
        D = MODELS[model]["dim"]
        # per-story word features (cached), pooled EMB/SFL-base per TR + lags
        emb_tr, sflbase_tr, word_sfl_by, wtr_by = [], [], [], []
        for si, st in enumerate(stories):
            emb_w, surp_w = extract.extract_words(
                model, [st["words"]], drop_first=True, chunked=True,
                cache=Path(cache_dir) / f"{dataset}_{model}_story{si}.npz")
            fr, ln = behavioral.freq_length([st["words"]])
            wsfl = np.column_stack([surp_w, fr, ln])
            wtr = np.asarray(st["word_tr"], int)
            emb_tr.append(_agg(emb_w, wtr, st["n_tr"], keeps[si], "mean"))
            sflbase_tr.append(_agg(wsfl, wtr, st["n_tr"], keeps[si], "mean"))
            word_sfl_by.append(wsfl)
            wtr_by.append(wtr)
        emb_mat = np.vstack(emb_tr)
        sfl_mat = np.vstack(sflbase_tr)

        def rff_build(train, test, _wsfl=word_sfl_by, _wtr=wtr_by, _D=D):
            train_stories = [i for i in range(len(stories))
                             if np.any((train >= bounds[i]) & (train < bounds[i + 1]))]
            tw = np.concatenate([_wsfl[i] for i in train_stories])
            t = rff.fit_rff(tw, _D, seed)
            per_story = [_agg(t(_wsfl[i]), _wtr[i], ntr[i], keeps[i], "sum")
                         for i in range(len(stories))]
            X = np.vstack(per_story)
            return X[train], X[test]

        def rffemb_build(train, test, _e=emb_mat, _rb=rff_build):
            a, b = _rb(train, test)
            return np.hstack([a, _e[train]]), np.hstack([b, _e[test]])

        specs = {
            "sfl":         dict(X=sfl_mat),
            "emb":         dict(X=emb_mat),
            "all":         dict(X=np.hstack([sfl_mat, emb_mat])),
            "sfl_rff":     dict(build_fn=rff_build),
            "sfl_rff_emb": dict(build_fn=rffemb_build),
        }
        rows[model] = conditions.run_conditions(specs, y, folds, f"{dataset}_{model}",
                                                results_dir, backend)
        decompose.maybe_decompose(model, sfl_mat, y, folds, f"{dataset}_{model}",
                                  n_blocks=len(delays), results_dir=results_dir, backend=backend)
    conditions.build_results_table(rows).to_csv(Path(results_dir) / f"{dataset}_results.csv")
    return rows
