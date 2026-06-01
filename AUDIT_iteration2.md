# SFL-EMB codebase audit — iteration 2 (reproducible paths)

Follow-up to [AUDIT_iteration1.md](AUDIT_iteration1.md). This iteration made the
scripts runnable without manually setting a working directory, repointed data
reads at the newly-added `main_data/`, simplified ZuCo, and removed code/data that
will not ship with the repo.

## 1. Reproducible working directory

Every paper-relevant script previously began with a hardcoded
`dirName = "/home/dev/Documents/PhD/MIT/project..."; chdir(dirName)`. These were
replaced by an auto-detecting header that finds the repo root (the directory
containing `utils.py`) from the script's own location, with a fallback for
interactive cell-by-cell execution:

```python
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
chdir(str(PROJECT_ROOT))          # or PROJECT_ROOT / "<subdir>" for nested scripts
```

- Works both as `python script.py` (any machine) and run interactively.
- The original per-script working directory is preserved (root for top-level
  scripts; the subdir for nested ones), so all existing relative paths still work.
- `utils.py` got its own `PROJECT_ROOT = Path(__file__).resolve().parent`, and its
  hardcoded fastText / `SEM/*.joblib` paths in `get_sem()` now use it.
- **37 scripts** migrated; all compile. Verified `PROJECT_ROOT` resolves to the
  repo root from top-level, `fMRI/sentence-level/`, `natstor_fmri/`, and `ZuCo/`.

## 2. Data paths repointed to `main_data/`

External raw-data reads were rewritten as `PROJECT_ROOT / "main_data/..."`. Mapping:

| Old | New |
|---|---|
| `…/Datasets/UCL_ET_Corpus/` | `main_data/UCL_ET_CORPUS/` (note case change) |
| `…/Datasets/Provo/` | `main_data/Provo/` |
| `…/Datasets/Brown/` | `main_data/Brown/` |
| `…/Datasets/NaturalStories/` | `main_data/NaturalStories/` |
| `…/Datasets/ERP_Frank/` | `main_data/ERP_Frank/` |
| `…/Datasets/ZuCo/` and `/media/drive2/Datasets/ZuCo/` | `main_data/ZuCo/` |
| `…/Datasets/subtlex.xlsx` | `main_data/subtlex.xlsx` |
| `…/side-projects/eye-tracking/Meco_l1.csv` | `main_data/Meco/Meco_l1.csv` |
| `…/PhD/MIT/project/<x>` (internal) | `PROJECT_ROOT / "<x>"` |

All referenced `main_data/*` files exist **except** `main_data/ZuCo/ET_data.csv`
(per-participant ZuCo data; only the aggregated table was supplied — see §5).

## 3. ZuCo simplified

`ZuCo/ZuCo.py` previously read raw per-subject `.mat` files and then immediately
**overwrote** the result with the aggregated CSV (dead code). It now loads only
`main_data/ZuCo/ET_data_aggregated.csv`. Removed the now-unused `h5py` and
`data_loading_helpers` imports.

## 4. Removed (will not be in the repo)

- **`probing_LMs/`** — deleted wholesale (~1.6 GB; the layer-probing sub-study,
  not part of this paper; depended on the removed norm files).
- **`probe_Michaelov_n400.py`** — deleted (sole input was the removed Michaelov
  N400 file; not in the paper's Table 1).
- **External semantic-norm reads** (Warriner affective, Lynott sensorimotor) in
  `probe_Pereira_1.py` / `probe_Pereira.py` — these were dead code (overwritten by
  the precomputed `interpret_brain_results_Pereira.csv`, which is kept); replaced
  with a comment. Downstream still reads the precomputed CSV.
- **Michaelov reliability blocks** in `other/reliability_and_size.py`,
  `other/split_half_corr.py`, `other/split_half_corr_NEW.py` — commented out, and
  the `out_mich`/`"michaelov"` entries dropped from their output lists/dicts.
- **`probing_LMs/XYdata/` write tails** ("save X,y for probing exp") at the end of
  ~10 analysis scripts — commented out (marked `# [removed: probing_LMs sub-study]`)
  so each script runs end-to-end now that `probing_LMs/` is gone. These were
  optional dumps of already-computed results; nothing downstream depends on them.

## 5. Flags (not changed — need your call)

- **Pre-existing syntax errors** (NOT introduced here; present before migration):
  `other/HP_split_half.py` (empty `for` body), `other/UCL_punct_only.py`
  (unterminated string), `other/try_LLAMA.py` (invalid syntax). These are
  incomplete interactive scripts. Fixing belongs to the iteration-5 "obvious
  mistakes" pass.
- **`probe_Pereira.py`** (per-ROI) is superseded by `probe_Pereira_1.py` (the
  version used in the paper) and still needs raw Pereira data that wasn't supplied.
  It was NOT migrated; recommend deleting it (and the per-ROI `pereira/` dir).
- **`ZuCo/read_matlab_files.py`** + **`ZuCo/data_loading_helpers.py`** are now
  orphaned (only the raw-`.mat` path used them). Recommend deleting.
- **Still-missing data** for the appendix reliability analyses (not the main
  results): per-participant `main_data/ZuCo/ET_data.csv` and raw Pereira
  (`…/Datasets/Pereira/data/`, referenced in `other/split_half_corr.py`).
- **`.gitignore`**: the `*.rds` rule will ignore `main_data/NaturalStories/natstor_maze.rds`.
  Decide whether `main_data/` (332 MB) should be tracked in git or kept external;
  if tracked, that rule needs an exception.
