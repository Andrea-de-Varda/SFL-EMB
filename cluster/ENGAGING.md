# Running SFL-EMB on MIT Engaging

## Environment

Reuse the existing **`modularity-conda`** env (it already has torch + transformers).
The only extras the pipeline needs are **himalaya** (GPU ridge) and **pyreadr** (reads
`data/NaturalStories/natstor_maze.rds`); the warmup step below installs them on first
run, or install once yourself:

```bash
# after the module load + conda activation in cluster/env.sh:
pip install himalaya pyreadr openpyxl
```

`cluster/env.sh` centralizes the setup — the module loads, the finicky conda
**activate → deactivate → activate** dance `modularity-conda` needs in a batch shell,
and the HF cache/offload env vars. Both SLURM scripts `source` it, so change the env
name or cache paths there in one place. GPT / GPT-2 are public, so no `HF_TOKEN` is
required (uncomment it in `env.sh` only if you add a gated model).

## Code + data

```bash
cd /orcd/data/evelina9/001/USERS/devar_ag
git clone https://github.com/Andrea-de-Varda/SFL-EMB.git   # branch: refactor/modernize-pipeline
cd SFL-EMB
```

Everything the pipeline reads is tracked under `data/` **except** the large arrays
that are `.gitignore`d (fMRI subject `.npy`, the Tuckute response table, audio, the
NatStor timeseries, the Pereira parquet). rsync those from the laptop to the matching
`data/{wehbe,tuckute,natstor_fmri,pereira}/` paths.

## Run

Single GPU (all 13 datasets sequentially):

```bash
sbatch cluster/slurm_engaging.sh
sbatch cluster/slurm_engaging.sh probe_pereira.py probe_wehbe.py   # a subset
```

**3 GPUs in parallel** (recommended — ~3x faster):

```bash
sbatch cluster/slurm_engaging_array.sh
```

A SLURM job array (`--array=0-2`), one A100 per task. The 13 datasets are statically
split into 3 cost-balanced buckets, so each dataset runs on exactly one GPU — no
re-runs, no clobber (`results/` and `cache/` are per-dataset keyed). Per-task logs
land in `logs/sfl_emb_<jobid>_{0,1,2}.out`. A `flock`-guarded warmup installs the two
deps and downloads the 5 models once (the first task does it; the others wait, then
find everything cached). Edit the `BUCKET_0/1/2` arrays to re-balance.

Both scripts extract embeddings/surprisal fresh and cache them under `cache/`
(git-ignored), so re-runs skip extraction. Outputs: `results/<dataset>_..._results.csv`
plus per-fold prediction files.

## Layer-matched EMB vs SFL (logit lens)

```bash
sbatch cluster/slurm_layerwise.sh
squeue -u $USER                       # 4 tasks: sfl_layerwise_<jobid>_{0,1,2,3}
```

`--array=0-3`, one A100 per task, 13 datasets x 25 layers x 2 ridge fits. Too heavy
for a laptop GPU (~30 s per cell => hours). Each dataset writes its **own** csv
(`appendix/layerwise/<dataset>.csv`), so array tasks never share a writer and any
completed `(dataset, layer)` cell is skipped — **re-submitting the same job resumes**
rather than redoing work. Progress: `tail -f logs/sfl_layerwise_<jobid>_0.out`.

Each row stores the **per-fold r vectors** (`emb_folds` / `sfl_folds`), which the
paired permutation test needs; fold means/SDs alone cannot support it. Rows written
before those columns existed are detected and recomputed, not skipped — so the first
run under this schema redoes all 13 datasets.

Then, locally:

```bash
python appendix/layerwise_logitlens.py --merge
python appendix/layerwise_significance.py   # paired permutation test vs standard SFL
python appendix/layerwise_plot.py
```

## Figures / appendix (run LOCALLY after pulling results — not on the login node)

```bash
python figures/plot_results.py                 # main + RFF + decomposition + model-size
python figures/sync_to_overleaf.py             # copy every figure into the Overleaf tree (+ audit)
python figures/verify_paper_numbers.py         # re-derive every reported number, assert vs the .tex
python appendix/layer_sweep.py                 # A4 layer sweep (gpt2_medium)
python appendix/layerwise_logitlens.py --merge # per-dataset csvs -> appendix/layerwise_logitlens.csv
python appendix/layerwise_plot.py              # -> figures/layerwise/{behavior,brain}.png
python appendix/reliability_and_size.py        # A5 reliability (CPU, from raw data; caches to cache/)
python appendix/reliability_and_size.py --replot   # re-plot A5 from cache, no recompute
python check-features/predict_features.py      # SFL_RFF meaning-vs-form probe
```
