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

## Figures / appendix (after the run)

```bash
python figures/plot_results.py                 # main comparison + SFL decomposition
python appendix/layer_sweep.py                 # A4 layer sweep (gpt2_medium)
python appendix/reliability_and_size.py        # A5 reliability
python check-features/predict_features.py      # SFL_RFF meaning-vs-form probe
```
