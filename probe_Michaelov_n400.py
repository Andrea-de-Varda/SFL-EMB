import neurox
import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from glmnet import ElasticNet
import scipy.io
from researchpy import corr_pair
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from scipy.special import softmax
import re
import matplotlib.patches as mpatches
import pickle

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, get_surprisal_tokens, decompose_sfl

df = pd.read_csv("/home/dev/Documents/Datasets/michaelov_n400/N400_data.csv")
df1 = df.groupby(["ContextCode", "Sentence"], as_index=False, sort=False).agg({'Cloze':"mean", 'N400':"mean", 'PlausibilityJudgement':"mean", 'Sentence':"max", "Condition":"max"})

df1.columns
n400 = np.array(df1["N400"])

# with open('corpora/michaelov_sentences.txt', 'w') as f:
#     for line in df1["Sentence"]:
#         f.write(f"{line}\n")


##############################
# extracting representations #
##############################

import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

#######
# GPT #
#######

# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt_michaelov.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "corpora/michaelov_sentences.txt",
        dirName+'/activations_gpt_michaelov.json',
        aggregation="average"
    )

################
# GPT-2 FAMILY #
################

# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small_michaelov.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "corpora/michaelov_sentences.txt",
        dirName+'/activations_gpt2_small_michaelov.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium_michaelov.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "corpora/michaelov_sentences.txt",
        dirName+'/activations_gpt2_medium_michaelov.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large_michaelov.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "corpora/michaelov_sentences.txt",
        dirName+'/activations_gpt2_large_michaelov.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl_michaelov.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "corpora/michaelov_sentences.txt",
        dirName+'/activations_gpt2_xl_michaelov.json',
        aggregation="average"
    )

# loading activations
act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt_michaelov.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small_michaelov.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium_michaelov.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large_michaelov.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl_michaelov.json', 1600, is_brnn=False)

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act(act_list, layer, dim):
    out = []
    for act_sent in act_list:
        out.append(get_layer(act_sent[-1], layer, dim))
    return np.array(out)

gpt_filt   = get_act(act_gpt, 12, 768)
small_filt = get_act(act_gpt2_small, 12, 768)
med_filt   = get_act(act_gpt2_medium, 24, 1024)
large_filt = get_act(act_gpt2_large, 36, 1280)
xl_filt    = get_act(act_gpt2_xl, 48, 1600)
    
#######
# SFL #
#######

sent = [re.sub("\n", "", s).split() for s in open("corpora/michaelov_sentences.txt").readlines()]

# GPT
gpt = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
tok_gpt = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt = [get_surprisal_tokens(s, sep = "</w>", toker=tok_gpt, model=gpt) for s in tqdm(sent)]
s_gpt1 = [s[-1] for s in s_gpt]

# GPT2 SMALL (124M parameters)
small = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
tok_small = AutoTokenizer.from_pretrained("gpt2")
s_small = [get_surprisal_tokens(s, sep = "Ġ", toker=tok_small, model=small, cased=True) for s in tqdm(sent)]
s_small1 = [s[-1] for s in s_small]

# GPT2 medium
med = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
tok_med = AutoTokenizer.from_pretrained("gpt2-medium")
s_med = [get_surprisal_tokens(s, sep = "Ġ", toker=tok_med, model=med, cased=True) for s in tqdm(sent)]
s_med1 = [s[-1] for s in s_med]

# GPT2 large
large = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
tok_large = AutoTokenizer.from_pretrained("gpt2-large")
s_large = [get_surprisal_tokens(s, sep = "Ġ", toker=tok_large, model=large, cased=True) for s in tqdm(sent)]
s_large1 = [s[-1] for s in s_large]

# GPT2 xl
xl = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
tok_xl = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl = [get_surprisal_tokens(s, sep = "Ġ", toker = tok_xl, model=xl, cased=True)for s in tqdm(sent) ]
s_xl1 = [s[-1] for s in s_xl]

################################

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

freq = [get_f(s[-1]) for s in sent]
length = [len(s[-1]) for s in sent]

df = pd.DataFrame({"s_GPT":s_gpt1,
                   "s_small":s_small1,
                   "s_med":s_med1,
                   "s_large":s_large1,
                   "s_xl":s_xl1,
                   "freq":freq,
                   "len":length})
# df.to_csv("michaelov_SFL.csv")
df = pd.read_csv("michaelov_SFL.csv")

# activation only
m_gpt, sd_gpt     = test_model_Ridge(gpt_filt, n400, 10, "michaelov_gpt")
m_small, sd_small = test_model_Ridge(small_filt, n400, 10, "michaelov_small")
m_med, sd_med     = test_model_Ridge(med_filt, n400, 10, "michaelov_med")
m_larg, sd_larg   = test_model_Ridge(large_filt, n400, 10, "michaelov_large")
m_xl, sd_xl       = test_model_Ridge(xl_filt, n400, 10, "michaelov_xl")
print(m_gpt, m_small, m_med, m_larg, m_xl)

# baseline only
baseline_gpt   = np.array(list(zip(df["s_GPT"], df["freq"], df["len"]))) 
baseline_small = np.array(list(zip(df["s_small"], df["freq"], df["len"]))) 
baseline_med   = np.array(list(zip(df["s_med"], df["freq"], df["len"]))) 
baseline_large = np.array(list(zip(df["s_large"], df["freq"], df["len"]))) 
baseline_xl    = np.array(list(zip(df["s_xl"], df["freq"], df["len"]))) 

m_gpt_b, sd_gpt_b     = test_model_Ridge(baseline_gpt, n400, 10, "michaelov_gpt_base")
m_small_b, sd_small_b = test_model_Ridge(baseline_small, n400, 10, "michaelov_small_base")
m_med_b, sd_med_b     = test_model_Ridge(baseline_med, n400, 10, "michaelov_med_base")
m_larg_b, sd_larg_b   = test_model_Ridge(baseline_large, n400, 10, "michaelov_large_base")
m_xl_b, sd_xl_b       = test_model_Ridge(baseline_xl, n400, 10, "michaelov_xl_base")
print(m_gpt_b, m_small_b, m_med_b, m_larg_b, m_xl_b)

# decompose_sfl
decompose_sfl(baseline_gpt, n400, "michaelov_gpt_base")
decompose_sfl(baseline_small, n400, "michaelov_small_base")
decompose_sfl(baseline_med, n400, "michaelov_med_base")
decompose_sfl(baseline_large, n400, "michaelov_large_base")
decompose_sfl(baseline_xl, n400, "michaelov_xl_base")

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_all, sd_gpt_all     = test_model_Ridge(gpt_all, n400, 10, "michaelov_gpt_all"); print(m_gpt_all)
m_small_all, sd_small_all = test_model_Ridge(gpt2_small_all, n400, 10, "michaelov_small_all"); print(m_small_all)
m_med_all, sd_med_all     = test_model_Ridge(gpt2_med_all, n400, 10, "michaelov_med_all")
m_larg_all, sd_larg_all   = test_model_Ridge(gpt2_large_all, n400, 10, "michaelov_large_all")
m_xl_all, sd_xl_all       = test_model_Ridge(gpt2_xl_all, n400, 10, "michaelov_xl_all")
print(m_gpt_all, m_small_all, m_med_all, m_larg_all, m_xl_all)

df_results = pd.DataFrame({
    "base":[m_gpt_b, m_small_b, m_med_b, m_larg_b, m_xl_b],
    "base_sd": [sd_gpt_b, sd_small_b, sd_med_b, sd_larg_b, sd_xl_b],
    "act":[m_gpt, m_small, m_med, m_larg, m_xl],
    "act_sd": [sd_gpt, sd_small, sd_med, sd_larg, sd_xl],
    "all":[m_gpt_all, m_small_all, m_med_all, m_larg_all, m_xl_all],
    "all_sd":[sd_gpt_all, sd_small_all, sd_med_all, sd_larg_all, sd_xl_all]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_results.to_csv("results_michaelov.csv")
df_results = pd.read_csv("results_michaelov.csv")

base_plot(df_results, "act", "base", "all", title="", ylim1=-0.1, ylim2=0.4)

############
# stat sig #
############

thebase = ["michaelov_gpt_base", "michaelov_small_base", "michaelov_med_base", "michaelov_large_base", "michaelov_xl_base"]
theemb  = ["michaelov_gpt", "michaelov_small", "michaelov_med", "michaelov_large", "michaelov_xl"]
theall     = ["michaelov_gpt_all", "michaelov_small_all", "michaelov_med_all", "michaelov_large_all", "michaelov_xl_all"]

p = []
for base, emb, all_ in zip(thebase, theemb, theall):
    p_base_emb = significance_testing(emb, base)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p.append([p_base_emb, p_all_base, p_all_emb])
p = pd.DataFrame(p, columns=["base_emb", "all_base", "all_emb"])

for df_column in p.columns:
    df_results["p_"+df_column] = list(p[df_column])
  
#######
# SEM #
#######

from utils import plot_sem, get_sem, plot_weights_sem

wordlist = [s[-1] for s in sent]
sem = get_sem(wordlist)
# Missing affective = 22.8
# Missing perceptual = 14.4

# baseline
gpt_base_sem   = np.concatenate((sem, baseline_gpt), axis=1)
small_base_sem = np.concatenate((sem, baseline_small), axis=1)
med_base_sem   = np.concatenate((sem, baseline_med), axis=1)
large_base_sem = np.concatenate((sem, baseline_large), axis=1)
xl_base_sem    = np.concatenate((sem, baseline_xl), axis=1)

# embedding
gpt_sem   = np.concatenate((sem, gpt_filt), axis=1)
small_sem = np.concatenate((sem, small_filt), axis=1)
med_sem   = np.concatenate((sem, med_filt), axis=1)
large_sem = np.concatenate((sem, large_filt), axis=1)
xl_sem    = np.concatenate((sem, xl_filt), axis=1)

# all
gpt_all_sem   = np.concatenate((sem, gpt_all), axis=1)
small_all_sem = np.concatenate((sem, gpt2_small_all), axis=1)
med_all_sem   = np.concatenate((sem, gpt2_med_all), axis=1)
large_all_sem = np.concatenate((sem, gpt2_large_all), axis=1)
xl_all_sem    = np.concatenate((sem, gpt2_xl_all), axis=1)

# baseline
m_gpt_b, sd_gpt_b     = test_model_Ridge(gpt_base_sem, n400, 10, "michaelov_gpt_base_sem")
m_small_b, sd_small_b = test_model_Ridge(small_base_sem, n400, 10, "michaelov_small_base_sem")
m_med_b, sd_med_b     = test_model_Ridge(med_base_sem, n400, 10, "michaelov_med_base_sem")
m_larg_b, sd_larg_b   = test_model_Ridge(large_base_sem, n400, 10, "michaelov_large_base_sem")
m_xl_b, sd_xl_b       = test_model_Ridge(xl_base_sem, n400, 10, "michaelov_xl_base_sem")
print(m_gpt_b, m_small_b, m_med_b, m_larg_b, m_xl_b)

# activations
m_gpt, sd_gpt     = test_model_Ridge(gpt_sem, n400, 10, "michaelov_gpt_sem")
m_small, sd_small = test_model_Ridge(small_sem, n400, 10, "michaelov_small_sem")
m_med, sd_med     = test_model_Ridge(med_sem, n400, 10, "michaelov_med_sem")
m_larg, sd_larg   = test_model_Ridge(large_sem, n400, 10, "michaelov_large_sem")
m_xl, sd_xl       = test_model_Ridge(xl_sem, n400, 10, "michaelov_xl_sem")
print(m_gpt, m_small, m_med, m_larg, m_xl)

# all
m_gpt_all, sd_gpt_all     = test_model_Ridge(gpt_all_sem, n400, 10, "michaelov_gpt_all_sem")
m_small_all, sd_small_all = test_model_Ridge(small_all_sem, n400, 10, "michaelov_small_all_sem")
m_med_all, sd_med_all     = test_model_Ridge(med_all_sem, n400, 10, "michaelov_med_all_sem")
m_larg_all, sd_larg_all   = test_model_Ridge(large_all_sem, n400, 10, "michaelov_large_all_sem")
m_xl_all, sd_xl_all       = test_model_Ridge(xl_all_sem, n400, 10, "michaelov_xl_all_sem")
print(m_gpt_all, m_small_all, m_med_all, m_larg_all, m_xl_all)

df_results_sem = pd.DataFrame({
    "base_sem":[m_gpt_b, m_small_b, m_med_b, m_larg_b, m_xl_b],
    "base_sem_sd": [sd_gpt_b, sd_small_b, sd_med_b, sd_larg_b, sd_xl_b],
    "act_sem":[m_gpt, m_small, m_med, m_larg, m_xl],
    "act_sem_sd": [sd_gpt, sd_small, sd_med, sd_larg, sd_xl],
    "all_sem":[m_gpt_all, m_small_all, m_med_all, m_larg_all, m_xl_all],
    "all_sem_sd":[sd_gpt_all, sd_small_all, sd_med_all, sd_larg_all, sd_xl_all]
    })
df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

merged = pd.concat([df_results, df_results_sem], axis=1)
#merged.to_csv("SEM/merged_michaelov.csv")
merged = pd.read_csv("SEM/merged_michaelov.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", title="N400", subtract=True)

########################
# for last probing exp #
########################

from scipy.stats import zscore

m_med, sd_med     = test_model_Ridge(zscore(med_filt), zscore(n400), 10, "michaelov_med_probing")
m_small, sd_small     = test_model_Ridge(zscore(    small_filt), zscore(n400), 10, "michaelov_small_probing")

from joblib import load
pca = load('probing_LMs/pca.pkl')
med_filt_pca = pca.transform(med_filt)

# first-pass, activation only
m_med, sd_med = test_model_Ridge(med_filt_pca, n400, 10, "michaelov_med_probing_pca")

##########################
# end of weight analysis #
##########################
# save med_filt, y
dataname = "michaelov"
tosave = [med_filt, n400]
with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:
    pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
