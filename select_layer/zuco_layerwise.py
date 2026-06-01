import os
from os import chdir, path
import numpy as np
import h5py
import sys
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import pickle

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT / "ZuCo")
chdir(dirName)

parentdir = path.dirname(dirName)
sys.path.insert(0, parentdir) 

from utils import get_surprisal_tokens, loadmat1, test_model_Ridge, test_model_GAM, base_plot, significance_testing, decompose_sfl

out_mean = pd.read_csv(PROJECT_ROOT / "main_data/ZuCo/ET_data_aggregated.csv") # NOTE: I changed the location

ff = np.array(out_mean[out_mean.word_idx != 0]["ff"])
fp = np.array(out_mean[out_mean.word_idx != 0]["gd"])
tt = np.array(out_mean[out_mean.word_idx != 0]["tt"])

###############################################################################    

import neurox.data.loader as data_loader

act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium.json', 1024, is_brnn=False)

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act_eye(act_list, layer, dim):
    out = []
    for act_sent in act_list:
        out.extend([get_layer(v, layer, dim) for v in act_sent[1:]])
    return np.array(out)

med_filt   = get_act_eye(act_gpt2_medium, 3, 1024)

###############################################################################

df = pd.read_csv("SFL.csv")

filt_na =  ~np.isnan(fp)
ff = ff[filt_na]
fp = fp[filt_na]
tf = tt[filt_na]

################################

# first-pass, activation only
m_med_fpass, sd_med_fpass     = test_model_Ridge(med_filt[filt_na], fp, 10, "zuco_med_layer3")

# first-pass, baseline only
baseline_med   = np.array(list(zip(df["s_med"], df["freq"], df["len"])))[filt_na]
m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(baseline_med, fp, 10, "zuco_med_base_layer3")

# all
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)

m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(gpt2_med_all, fp, 10, "zuco_med_all_layer3")


df_results = pd.DataFrame({
    "base_fpass":[m_med_fpass_b],
    "base_fpass_sd": [sd_med_fpass_b],
    "act_fpass": [m_med_fpass],
    "act_fpass_sd": [sd_med_fpass],
    "all_fpass":[m_med_fpass_all],
    "all_fpass_sd":[sd_med_fpass_all]
    })
df_results.index = ["GPT-2$_{355M}$"]

########################
# Significance testing #
########################

thebase_fpass = ["zuco_med_base_layer3"]
theemb_fpass  = ["zuco_med_layer3"]
all_fpass     = ["zuco_med_all_layer3"]

p_fpass = []
for base, emb, all_ in zip(thebase_fpass, theemb_fpass, all_fpass):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_fpass.append([p_base_emb, p_all_base, p_all_emb])
p_fpass = pd.DataFrame(p_fpass, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_fpass.columns:
    df_results["p_fpass_"+df_column] = list(p_fpass[df_column])

df_results.to_csv("../select_layer/results_zuco.csv")
