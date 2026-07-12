"""Helpers for the word-level behavioral datasets (eye-tracking + self-paced reading).

Two reusable drivers so each dataset script stays thin and the wrangling that is
genuinely dataset-specific (data loading, target construction, exclusions) lives in
the script, while the shared mechanics (frequency/length, extraction, spillover,
the 5-condition fit) live here.

  * :func:`run_word_dataset`  - word is the unit, no spillover (eye-tracking, maze).
  * :func:`run_spill_dataset` - self-paced reading: SFL and EMB both concatenate
    the current word and the two preceding words (i, i-1, i-2).

Convention shared with the paper / Provo template: sentences are whitespace word
lists that INCLUDE the sentence-initial word (for left context); it is dropped when
aligning features to the behavioral rows (``drop_first=True``). SFL columns are
[surprisal, log-frequency, length]; RFF n_components = the model hidden size D.
"""
import re
from pathlib import Path

import numpy as np

from .paths import PROJECT_ROOT
from . import extract, fit, conditions
from .models import MODELS

_PUNCT = re.compile(r'[\.\,\:\-\?\!\)\(\"]')


def load_subtlex(path=None):
    """SUBTLEX log-frequency dict (natural log of FREQcount) + the OOV floor."""
    import pandas as pd
    path = path or PROJECT_ROOT / "data/subtlex.xlsx"
    freq = pd.read_excel(path)
    f = {row.Word: np.log(row.FREQcount) for _, row in freq.iterrows()}
    return f, min(f.values())


def freq_length(sentences, drop_first=True, subtlex=None):
    """Per-word log-frequency and character length, flattened across sentences."""
    f, minf = subtlex or load_subtlex()
    freqs, lens = [], []
    for words in sentences:
        ws = words[1:] if drop_first else words
        for w in ws:
            freqs.append(f.get(_PUNCT.sub("", w), minf))
            lens.append(len(w))
    return np.array(freqs, dtype=float), np.array(lens, dtype=float)


def _cache(cache_dir, name):
    return None if cache_dir is None else Path(cache_dir) / f"{name}.npz"


def run_word_dataset(dataset, sentences, targets, models, results_dir, cache_dir,
                     backend="himalaya", row_mask=None, chunked=False):
    """Run the 5 conditions for a word-level dataset, one results table per target.

    ``targets`` maps target_name -> y aligned to the non-initial words (pre-mask).
    ``row_mask`` (optional bool array, same length) selects rows to keep uniformly
    across features and every target (e.g. ZuCo drops NaN-gaze words).
    """
    freqs, lens = freq_length(sentences)
    rows = {t: {} for t in targets}
    for model in models:
        emb, surp = extract.extract_words(model, sentences, drop_first=True,
                                          chunked=chunked,
                                          cache=_cache(cache_dir, f"{dataset}_{model}"))
        sfl = np.column_stack([surp, freqs, lens])
        assert len(sfl) == len(next(iter(targets.values()))), \
            f"{dataset}/{model}: {len(sfl)} feature rows vs target length"
        emb_m, sfl_m = (emb, sfl) if row_mask is None else (emb[row_mask], sfl[row_mask])
        tgt = targets if row_mask is None else {t: np.asarray(y)[row_mask] for t, y in targets.items()}
        folds = fit.make_folds(len(sfl_m), n_splits=10)
        D = MODELS[model]["dim"]
        for tname, y in tgt.items():
            prefix = f"{dataset}_{tname}_{model}"
            rows[tname][model] = conditions.run_all_conditions(
                sfl_m, emb_m, y, folds, prefix, n_components=D,
                results_dir=results_dir, backend=backend)
    _save_tables(dataset, targets, rows, results_dir)
    return rows


def _spill(mat, keep_idx):
    """Concatenate [row i, row i-1, row i-2] for each i in keep_idx."""
    return np.array([np.concatenate([mat[i], mat[i - 1], mat[i - 2]]) for i in keep_idx])


def run_spill_dataset(dataset, sentences, target, sent_ids, models, results_dir,
                      cache_dir, backend="himalaya"):
    """Self-paced reading with i, i-1, i-2 spillover on BOTH SFL and EMB.

    ``target`` is the per-word reading time aligned to the non-initial words;
    ``sent_ids`` gives each such word's sentence id (so spillover never crosses a
    sentence boundary). Words at within-sentence position < 2 (0-indexed among the
    non-initial words of their sentence) are dropped, as they lack two predecessors.
    """
    freqs, lens = freq_length(sentences)
    target = np.asarray(target, dtype=float)
    sent_ids = np.asarray(sent_ids)
    assert len(target) == len(freqs) == len(sent_ids), \
        f"{dataset}: target {len(target)} / freqs {len(freqs)} / sent_ids {len(sent_ids)} misaligned"
    # keep indices with two same-sentence predecessors
    keep = [i for i in range(len(target))
            if i >= 2 and sent_ids[i] == sent_ids[i - 1] == sent_ids[i - 2]]
    keep = np.array(keep)
    y = target[keep]
    rows = {}
    for model in models:
        emb, surp = extract.extract_words(model, sentences, drop_first=True,
                                          cache=_cache(cache_dir, f"{dataset}_{model}"))
        sfl = np.column_stack([surp, freqs, lens])
        sfl_sp = _spill(sfl, keep)          # (n, 9)
        emb_sp = _spill(emb, keep)          # (n, 3D)
        finite = np.isfinite(y) & np.isfinite(sfl_sp).all(1)   # drop non-finite RT/SFL
        sfl_sp, emb_sp, y_m = sfl_sp[finite], emb_sp[finite], y[finite]
        folds = fit.make_folds(len(y_m), n_splits=10)
        D = MODELS[model]["dim"]
        prefix = f"{dataset}_{model}"
        rows[model] = conditions.run_all_conditions(
            sfl_sp, emb_sp, y_m, folds, prefix, n_components=D,
            results_dir=results_dir, backend=backend)
    conditions.build_results_table(rows).to_csv(Path(results_dir) / f"{dataset}_results.csv")
    return rows


def _save_tables(dataset, targets, rows, results_dir):
    for tname in targets:
        if rows[tname]:
            out = Path(results_dir) / f"{dataset}_{tname}_results.csv"
            conditions.build_results_table(rows[tname]).to_csv(out)
            print(f"wrote {out}")
