import pandas as pd
import numpy as np
import numpy.ma as ma
from os import chdir, path
import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import math
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
from tqdm import tqdm
import warnings
from scipy.stats import zscore

warnings.filterwarnings("ignore", message="Mean of empty slice")

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT / "control_fmri")
chdir(dirName)

parentdir = path.dirname(dirName)
sys.path.insert(0, parentdir) 

from utils import get_surprisal_tokens, loadmat1, test_model_Ridge, test_model_GAM, base_plot, significance_testing, decompose_sfl, probe

############
# GET DATA #
############

control = pd.read_csv("data/brain-lang-data_participant_20230728.csv")
df_control = control[control["roi"] == "lang_LH_netw"].groupby("sentence").agg({"response_target" : "mean", "cond" : "max"})

sentences = df_control.index.tolist()
y = df_control["response_target"].to_numpy()

with open('data/sentences.txt', 'w') as f:
    for line in sentences:
        l = line.strip()
        f.write(f"{l}\n")

##############################
# EXTRACTING REPRESENTATIONS #
##############################

# https://huggingface.co/transformers/v2.2.0/pretrained_models.html

import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "data/sentences.txt",
        dirName+'/activations_gpt.json',
        aggregation="average"
    )

# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "data/sentences.txt",
        dirName+'/activations_gpt2_small.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "data/sentences.txt",
        dirName+'/activations_gpt2_medium.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "data/sentences.txt",
        dirName+'/activations_gpt2_large.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "data/sentences.txt",
        dirName+'/activations_gpt2_xl.json',
        aggregation="average"
    )

act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl.json', 1600, is_brnn=False)

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act(act_list, layer, dim):
    out = []
    for act_sent in act_list:
        act_sent = act_sent[1:].mean(axis=0) # mean-pooling for sentence-level representations
        out.append(get_layer(act_sent, layer, dim))
    return np.array(out)

gpt_filt   = get_act(act_gpt, 12, 768)
small_filt = get_act(act_gpt2_small, 12, 768)
med_filt   = get_act(act_gpt2_medium, 24, 1024)
large_filt = get_act(act_gpt2_large, 36, 1280)
xl_filt    = get_act(act_gpt2_xl, 48, 1600)

sent = [re.sub("\n", "", s).split() for s in open("data/sentences.txt").readlines()]

# GPT
gpt = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
tok_gpt = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "</w>", toker=tok_gpt, model=gpt)]

# GPT2 SMALL (124M parameters)
small = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
tok_small = AutoTokenizer.from_pretrained("gpt2")
s_small = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "Ġ", toker=tok_small, model=small, cased=True)]

# GPT2 medium
med = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
tok_med = AutoTokenizer.from_pretrained("gpt2-medium")
s_med = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "Ġ", toker=tok_med, model=med, cased=True)]

# GPT2 large
large = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
tok_large = AutoTokenizer.from_pretrained("gpt2-large")
s_large = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "Ġ", toker=tok_large, model=large, cased=True)]

# GPT2 xl
xl = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
tok_xl = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "Ġ", toker = tok_xl, model=xl, cased=True)]

# frequency
freq = pd.read_excel(PROJECT_ROOT / "main_data/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}
minfreq = min(f.values())

def get_f(word):
    word = re.sub('[\.\,\:\-\?\!\)\(\"]', "", word)
    try:
        fr = f[word]
    except KeyError:
        fr = minfreq
    return fr

freq = [get_f(word) for s in sent for word in s[1:]]
length = [len(word) for s in sent for word in s[1:]]
sentnum = [idx for idx, s in enumerate(sent) for word in s[1:]]

df = pd.DataFrame({"s_GPT":s_gpt,
                   "s_small":s_small,
                   "s_med":s_med,
                   "s_large":s_large,
                   "s_xl":s_xl,
                   "freq":freq,
                   "len":length,
                   "sentnum":sentnum})
# df.to_csv("SFL.csv")
df = pd.read_csv("SFL.csv")

###############################################################################

def pool_sfl(model, func):
    out = []
    for sentnum in range(2000):
        sent_aggregate = list(func(df[df.sentnum == sentnum][[model, "len", "freq"]]))
        out.append(sent_aggregate)
    pooled = np.array(out)
    return pooled

baseline_gpt   = pool_sfl("s_GPT", np.mean)
baseline_small = pool_sfl("s_small", np.mean)
baseline_med   = pool_sfl("s_med", np.mean)
baseline_large = pool_sfl("s_large", np.mean)
baseline_xl    = pool_sfl("s_xl", np.mean)

# all
gpt_all   = np.concatenate((baseline_gpt, gpt_filt), axis=1)
small_all = np.concatenate((baseline_small, small_filt), axis=1)
med_all   = np.concatenate((baseline_med, med_filt), axis=1)
large_all = np.concatenate((baseline_large, large_filt), axis=1)
xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

###############################################################################

# baseline
base_gpt, base_gpt_sd = test_model_Ridge(baseline_gpt, y, 10, "control_gpt_base")
base_small, base_small_sd = test_model_Ridge(baseline_small, y, 10, "control_small_base")
base_med, base_med_sd = test_model_Ridge(baseline_med, y, 10, "control_med_base")
base_large, base_large_sd = test_model_Ridge(baseline_large, y, 10, "control_large_base")
base_xl, base_xl_sd = test_model_Ridge(baseline_xl, y, 10, "control_xl_base")

# act
gpt, gpt_sd = test_model_Ridge(gpt_filt, y, 10, "control_gpt")
small, small_sd = test_model_Ridge(small_filt, y, 10, "control_small")
med, med_sd = test_model_Ridge(med_filt, y, 10, "control_med")
large, large_sd = test_model_Ridge(large_filt, y, 10, "control_large")
xl, xl_sd = test_model_Ridge(xl_filt, y, 10, "control_xl")

# all
gpt_a, gpt_a_sd = test_model_Ridge(gpt_all, y, 10, "control_gpt_all")
small_a, small_a_sd = test_model_Ridge(small_all, y, 10, "control_small_all")
med_a, med_a_sd = test_model_Ridge(med_all, y, 10, "control_med_all")
large_a, large_a_sd = test_model_Ridge(large_all, y, 10, "control_large_all")
xl_a, xl_a_sd = test_model_Ridge(xl_all, y, 10, "control_xl_all")

df_results = pd.DataFrame({
    "base":[base_gpt, base_small, base_med, base_large, base_xl],
    "base_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
    "act":[gpt, small, med, large, xl],
    "act_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
    "all":[gpt_a, small_a, med_a, large_a, xl_a],
    "all_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

# significance testing
thebase = ["control_gpt_base", "control_small_base", "control_med_base", "control_large_base", "control_xl_base"]
theemb  = ["control_gpt", "control_small", "control_med", "control_large", "control_xl"]
theall  = ["control_gpt_all", "control_small_all", "control_med_all", "control_large_all", "control_xl_all"]

p = []
for base, emb, all_ in zip(thebase, theemb, theall):
    p_base_emb = significance_testing(emb, base)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p.append([p_base_emb, p_all_base, p_all_emb])
p = pd.DataFrame(p, columns=["base_emb", "all_base", "all_emb"])

for df_column in p.columns:
    df_results["p_"+df_column] = list(p[df_column])

df_results.to_csv("results/control_results.csv", index=False)

base_plot(df_results, "act", "base", "all", title="Control", ylim1=0, ylim2=1)

decompose_sfl(baseline_gpt, y, "control_gpt_base", idx_s = 0, idx_f = 2, idx_l = 1)
decompose_sfl(baseline_small, y, "control_small_base", idx_s = 0, idx_f = 2, idx_l = 1)
decompose_sfl(baseline_med, y, "control_med_base", idx_s = 0, idx_f = 2, idx_l = 1)
decompose_sfl(baseline_large, y, "control_large_base", idx_s = 0, idx_f = 2, idx_l = 1)
decompose_sfl(baseline_xl, y, "control_xl_base", idx_s = 0, idx_f = 2, idx_l = 1)

######################
# Semantic variables #
######################

# [removed: SEM follow-up — semantic-norm analysis is not part of the manuscript]
# from utils import plot_sem, get_sem, plot_weights_sem

# sem = get_sem(sent)

# baseline
# gpt_base_sem   = np.concatenate((sem, baseline_gpt), axis=1)
# small_base_sem = np.concatenate((sem, baseline_small), axis=1)
# med_base_sem   = np.concatenate((sem, baseline_med), axis=1)
# large_base_sem = np.concatenate((sem, baseline_large), axis=1)
# xl_base_sem    = np.concatenate((sem, baseline_xl), axis=1)

# embedding
# gpt_sem   = np.concatenate((sem, gpt_filt), axis=1)
# small_sem = np.concatenate((sem, small_filt), axis=1)
# med_sem   = np.concatenate((sem, med_filt), axis=1)
# large_sem = np.concatenate((sem, large_filt), axis=1)
# xl_sem    = np.concatenate((sem, xl_filt), axis=1)

# all
# gpt_all_sem   = np.concatenate((sem, gpt_all), axis=1)
# small_all_sem = np.concatenate((sem, small_all), axis=1)
# med_all_sem   = np.concatenate((sem, med_all), axis=1)
# large_all_sem = np.concatenate((sem, large_all), axis=1)
# xl_all_sem    = np.concatenate((sem, xl_all), axis=1)

# baseline
# base_gpt, base_gpt_sd = test_model_Ridge(gpt_base_sem, y, 10, "control_gpt_base_sem")
# base_small, base_small_sd = test_model_Ridge(small_base_sem, y, 10, "control_small_base_sem")
# base_med, base_med_sd = test_model_Ridge(med_base_sem, y, 10, "control_med_base_sem")
# base_large, base_large_sd = test_model_Ridge(large_base_sem, y, 10, "control_large_base_sem")
# base_xl, base_xl_sd = test_model_Ridge(xl_base_sem, y, 10, "control_xl_base_sem")

# act
# gpt, gpt_sd = test_model_Ridge(gpt_sem, y, 10, "control_gpt_sem")
# small, small_sd = test_model_Ridge(small_sem, y, 10, "control_small_sem")
# med, med_sd = test_model_Ridge(med_sem, y, 10, "control_med_sem")
# large, large_sd = test_model_Ridge(large_sem, y, 10, "control_large_sem")
# xl, xl_sd = test_model_Ridge(xl_sem, y, 10, "control_xl_sem")

# all
# gpt_a, gpt_a_sd = test_model_Ridge(gpt_all_sem, y, 10, "control_gpt_all_sem")
# small_a, small_a_sd = test_model_Ridge(small_all_sem, y, 10, "control_small_all_sem")
# med_a, med_a_sd = test_model_Ridge(med_all_sem, y, 10, "control_med_all_sem")
# large_a, large_a_sd = test_model_Ridge(large_all_sem, y, 10, "control_large_all_sem")
# xl_a, xl_a_sd = test_model_Ridge(xl_all_sem, y, 10, "control_xl_all_sem")

# df_results_sem = pd.DataFrame({
#     "base_sem":[base_gpt, base_small, base_med, base_large, base_xl],
#     "base_sem_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
#     "act_sem":[gpt, small, med, large, xl],
#     "act_sem_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
#     "all_sem":[gpt_a, small_a, med_a, large_a, xl_a],
#     "all_sem_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
#     })
# df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

# significance testing
# thebase = ["control_gpt_base_sem", "control_small_base_sem", "control_med_base_sem", "control_large_base_sem", "control_xl_base_sem"]
# theemb  = ["control_gpt_sem", "control_small_sem", "control_med_sem", "control_large_sem", "control_xl_sem"]
# theall  = ["control_gpt_all_sem", "control_small_all_sem", "control_med_all_sem", "control_large_all_sem", "control_xl_all_sem"]

# p_sem = []
# for base, emb, all_ in zip(thebase, theemb, theall):
#     p_base_emb = significance_testing(emb, base)
#     p_all_base = significance_testing(all_, base)
#     p_all_emb  = significance_testing(all_, emb)
#     p_sem.append([p_base_emb, p_all_base, p_all_emb])
# p_sem = pd.DataFrame(p_sem, columns=["base_emb_sem", "all_base_sem", "all_emb_sem"])

# for df_column in p_sem.columns:
#     df_results_sem["p_"+df_column] = list(p_sem[df_column])

# merged = pd.concat([df_results, df_results_sem], axis=1)
# merged.to_csv("results/all_results_sem.csv", index=False)
# plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", title="Control", subtract=True)

########################
# for last probing exp #
########################

probe(med_filt, y, 10, "control_med_probing")


##########################
# end of weight analysis #
##########################
# save med_filt, y
dataname = "control"
tosave = [med_filt, y]
# with open(f"../probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
