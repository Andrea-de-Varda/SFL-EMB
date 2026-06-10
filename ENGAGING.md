# Running the matched-brain analyses on Engaging

Login node: `engaging` (`orcd-login.mit.edu`, user `devar_ag`). Scratch/project
dir used here: `/orcd/data/evelina9/001/USERS/devar_ag/`.

## 1. Code (git clone on the cluster)
```bash
ssh engaging
cd /orcd/data/evelina9/001/USERS/devar_ag
git clone https://github.com/Andrea-de-Varda/SFL-EMB.git   # or git@ if keys are set up
cd SFL-EMB
```
This brings the scripts + all tracked small files (`main_data/`, `*/SFL.csv`,
`UCL_reading_all.csv`, `results/`, `environment_neurox.yml`, …).

## 2. Data (rsync the gitignored inputs from the laptop, ~45 GB)
Run **on the laptop**:
```bash
cd ~/Documents/PhD/SFL-EMB
bash sync_to_engaging.sh        # rsync -P, resumable
```
Moves: `control_fmri/activations_*` + `data/`; `natstor_fmri/embeddings,response,transcribed`;
`fMRI/HP_emb/activations_*` + `fMRI/fMRI/*.npy` + `fMRI/SFL` + `sentences.txt`.
UCL-N400 embeddings are **not** moved — they're extracted on the cluster (GPU).

## 3. Environment (`neurox`, python 3.9)
The laptop env: python 3.9, neurox 1.0.9, scikit-learn **0.24.2** (pin this — RidgeCV/
RBFSampler behaviour), numpy 1.19.5, scipy 1.5.4, pandas 1.4.4.
```bash
module load anaconda3/2022.05-x86_64           # per your usual Engaging setup
conda create -y -n neurox python=3.9
conda activate neurox
pip install neurox==1.0.9 scikit-learn==0.24.2 numpy==1.19.5 scipy==1.5.4 pandas==1.4.4 pyarrow
# neurox pulls torch + transformers. If the cluster CUDA rejects the default torch,
# install a cluster-matching build (only the N400 extraction actually needs a GPU).
```
`environment_neurox.yml` / `requirements_neurox.txt` are committed as reference (the
torch pin there is the laptop's `+cu102` build — replace with a cluster build).

## 4. Run
- **CPU jobs** (load precomputed activations + CV): Tuckute, NatStor-fMRI, HP.
- **GPU job**: UCL-N400 (extracts embeddings first). Set HF cache + token as usual:
  `export HF_HOME=/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new`.

```bash
conda activate neurox
bash run_matched_brain.sh natstorfmri      # one dataset
bash run_matched_brain.sh                   # all four (control, n400, natstorfmri, HP)
```
Each overwrites its `results/*.csv` (base/act/all + p_* kept for plotting.py, plus
the matched columns + all-pairs permutation p-values).

Suggested SLURM split: a CPU job (`-c 4 --mem=64G -t 12:00:00`) for control+natstorfmri+HP,
and a separate GPU job for n400.
