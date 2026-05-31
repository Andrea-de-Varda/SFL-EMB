import neurox
import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import re
from scipy.special import softmax
import torch.nn.functional as F
from glmnet import ElasticNet
import scipy.io
from researchpy import corr_pair
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from scipy.special import softmax
import matplotlib.patches as mpatches
import pickle
import neurox.data.loader as data_loader

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, decompose_sfl

################
# PROVO CORPUS #
################

provo = pd.read_csv('/home/dev/Documents/Datasets/Provo/Provo_Corpus-Eyetracking_Data.csv', sep=",")
provo = provo[['Text_ID','Word_Unique_ID','Word_Number', 'Sentence_Number', 'Word_In_Sentence_Number','Word', 'Word_Cleaned', 'Word_Length', 'IA_FIRST_FIXATION_DURATION','IA_FIRST_RUN_DWELL_TIME', 'IA_DWELL_TIME']]
provo['IA_FIRST_FIXATION_DURATION'] = provo['IA_FIRST_FIXATION_DURATION'].fillna(0)
provo['IA_FIRST_RUN_DWELL_TIME'] = provo['IA_FIRST_RUN_DWELL_TIME'].fillna(0)
provo['IA_DWELL_TIME'] = provo['IA_DWELL_TIME'].fillna(0)
provo = provo.groupby(["Sentence_Number", "Word_In_Sentence_Number", "Word", "Text_ID"], as_index=False, sort=False).agg({'IA_FIRST_FIXATION_DURATION':"mean", 'IA_FIRST_RUN_DWELL_TIME':"mean", 'IA_DWELL_TIME':"mean", 'Word_Number':"mean", 'Sentence_Number':"mean", "Word_Unique_ID":"max", "Word_In_Sentence_Number":"mean", "Word":"max", "Word_Cleaned":"max", "Word_Length":"mean", "Text_ID":"mean"})

provo.columns
provo["Word_Unique_ID"]
set(provo["Sentence_Number"])
set(provo["Text_ID"])

# in DF, the first word is missing! Take it from stimuli data
first_word = [s.split("\t")[2][1:-2].split()[0] for s in open("/home/dev/Documents/Datasets/Provo/sentences.txt").readlines()]
d_f_w = {num+1 : first_word[num] for num in range(55)}

# frequency
freq = pd.read_excel("/home/dev/Documents/Datasets/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}
minfreq = min(f.values())

def get_f(word):
    word = re.sub('[\.\,\:\-\?\!\)\(\"]', "", word)
    try:
        fr = f[word]
    except KeyError:
        fr = minfreq
    return fr

ff = []; fp = []; tf = []; provo_len = []; provo_freq = []; provo_sent = []; provo_word = []
for text in set(provo["Text_ID"]):
    temp = provo[provo["Text_ID"] == text]
    ff.extend(list(temp["IA_FIRST_FIXATION_DURATION"]))
    fp.extend(list(temp["IA_FIRST_RUN_DWELL_TIME"]))
    tf.extend(list(temp["IA_DWELL_TIME"]))
    provo_len.extend(list(temp["Word_Length"]))
    provo_freq.extend([get_f(w) for w in list(temp["Word"])])
    provo_word.extend(list(temp["Word"]))
    sent = " ".join([d_f_w[text]]+list(temp["Word"]))
    provo_sent.append(sent)

ff = np.array(ff); fp = np.array(fp); tf = np.array(tf); provo_len = np.array(provo_len); provo_freq = np.array(provo_freq)

# with open('corpora/provo_sentences.txt', 'w') as f:
#     for line in provo_sent:
#         f.write(f"{line}\n")

###############################################################################

df = pd.read_csv("provo_SFL.csv")
s_gpt_provo = np.array(df["s_GPT"])
s_small_provo = np.array(df["s_small"])
s_med_provo = np.array(df["s_med"])
s_large_provo = np.array(df["s_large"])
s_xl_provo = np.array(df["s_xl"])
provo_freq = np.array(df["freq"])
provo_len = np.array(df["len"])

# loading activations
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium_provo.json', 1024, is_brnn=False)

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
    
# first-pass, activation only
m_med_fpass, sd_med_fpass     = test_model_Ridge(med_filt, fp, 10, "provo_med_layer3")

# first-pass, baseline only
baseline_med   = np.array(list(zip(s_med_provo, provo_freq, provo_len)))
m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(baseline_med, fp, 10, "provo_med_base_layer3")

# all
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(gpt2_med_all, fp, 10, "provo_med_all_layer3")

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

thebase_fpass = ["provo_med_base_layer3"]
theemb_fpass  = ["provo_med_layer3"]
all_fpass     = ["provo_med_all_layer3"]

p_fpass = []
for base, emb, all_ in zip(thebase_fpass, theemb_fpass, all_fpass):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_fpass.append([p_base_emb, p_all_base, p_all_emb])
p_fpass = pd.DataFrame(p_fpass, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_fpass.columns:
    df_results["p_fpass_"+df_column] = list(p_fpass[df_column])

df_results.to_csv("select_layer/results_provo.csv")