import os
from os import chdir, path
import numpy as np
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

parentdir = str(PROJECT_ROOT)
sys.path.insert(0, parentdir)

from utils import get_surprisal_tokens, loadmat1, test_model_Ridge, test_model_GAM, base_plot, significance_testing, decompose_sfl, probe
    
# Load the aggregated (participant-averaged) ZuCo eye-tracking data directly.
# (The raw per-subject .mat reading that produced this file is no longer run;
#  only the aggregated table is shipped with the repo. It is already
#  participant-averaged, NaN-filled, and sorted by [idx, word_idx].)
out_mean = pd.read_csv(PROJECT_ROOT / "main_data/ZuCo/ET_data_aggregated.csv")

ff = np.array(out_mean[out_mean.word_idx != 0]["ff"])
fp = np.array(out_mean[out_mean.word_idx != 0]["gd"])
tt = np.array(out_mean[out_mean.word_idx != 0]["tt"])

sentences = []
for sent_idx in set(out_mean["idx"]):
    temp = out_mean[out_mean["idx"] == sent_idx]
    words = " ".join(temp.word)
    sentences.append(words)
    
# with open("sentences.txt", 'w') as fp:
#     for item in list(sentences):
#         fp.write("%s\n" % item)
#     print('Done')

###############################################################################    
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

#######
# GPT #
#######

# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "sentences.txt",
        dirName+'/activations_gpt.json',
        aggregation="average"
    )

################
# GPT-2 FAMILY #
################

# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "sentences.txt",
        dirName+'/activations_gpt2_small.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "sentences.txt",
        dirName+'/activations_gpt2_medium.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "sentences.txt",
        dirName+'/activations_gpt2_large.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "sentences.txt",
        dirName+'/activations_gpt2_xl.json',
        aggregation="average"
    )
    
# loading activations
act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl.json', 1600, is_brnn=False)

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act_eye(act_list, layer, dim):
    out = []
    for act_sent in act_list:
        out.extend([get_layer(v, layer, dim) for v in act_sent[1:]])
    return np.array(out)

gpt_filt   = get_act_eye(act_gpt, 12, 768)
small_filt = get_act_eye(act_gpt2_small, 12, 768)
med_filt   = get_act_eye(act_gpt2_medium, 24, 1024)
large_filt = get_act_eye(act_gpt2_large, 36, 1280)
xl_filt    = get_act_eye(act_gpt2_xl, 48, 1600)

###############################################################################

sent = [re.sub("\n", "", s).split() for s in open("sentences.txt").readlines()]
# NOTE: I manually changed some out-of-voc tokens (país --> pais, – --> -, Lévesque --> Levesque, Gérard --> Gerard, Selâmet --> Selamet, Bôcher --> Bocher, Erdoğan --> Erdogan)

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

################################

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

df = pd.DataFrame({"s_GPT":s_gpt,
                   "s_small":s_small,
                   "s_med":s_med,
                   "s_large":s_large,
                   "s_xl":s_xl,
                   "freq":freq,
                   "len":length})
# df.to_csv("SFL.csv")
df = pd.read_csv("SFL.csv")

filt_na =  ~np.isnan(fp)
ff = ff[filt_na]
fp = fp[filt_na]
tf = tt[filt_na]

################################

# first-pass, activation only
m_gpt_fpass, sd_gpt_fpass     = test_model_Ridge(gpt_filt[filt_na], fp, 10, "zuco_gpt")
m_small_fpass, sd_small_fpass = test_model_Ridge(small_filt[filt_na], fp, 10, "zuco_small")
m_med_fpass, sd_med_fpass     = test_model_Ridge(med_filt[filt_na], fp, 10, "zuco_med")
m_larg_fpass, sd_larg_fpass   = test_model_Ridge(large_filt[filt_na], fp, 10, "zuco_large")
m_xl_fpass, sd_xl_fpass       = test_model_Ridge(xl_filt[filt_na], fp, 10, "zuco_xl")
print(m_gpt_fpass, m_small_fpass, m_med_fpass, m_larg_fpass)

# first-pass, baseline only
baseline_gpt   = np.array(list(zip(df["s_GPT"], df["freq"], df["len"])))[filt_na]
baseline_small = np.array(list(zip(df["s_small"], df["freq"], df["len"])))[filt_na]
baseline_med   = np.array(list(zip(df["s_med"], df["freq"], df["len"])))[filt_na]
baseline_large = np.array(list(zip(df["s_large"], df["freq"], df["len"])))[filt_na]
baseline_xl    = np.array(list(zip(df["s_xl"], df["freq"], df["len"])))[filt_na]

m_gpt_fpass_b, sd_gpt_fpass_b     = test_model_Ridge(baseline_gpt, fp, 10, "zuco_gpt_base")
m_small_fpass_b, sd_small_fpass_b = test_model_Ridge(baseline_small, fp, 10, "zuco_small_base")
m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(baseline_med, fp, 10, "zuco_med_base")
m_larg_fpass_b, sd_larg_fpass_b   = test_model_Ridge(baseline_large, fp, 10, "zuco_large_base")
m_xl_fpass_b, sd_xl_fpass_b       = test_model_Ridge(baseline_xl, fp, 10, "zuco_xl_base")
print(m_gpt_fpass_b, m_small_fpass_b, m_med_fpass_b, m_larg_fpass_b, m_xl_fpass_b)

# decompose_sfl
decompose_sfl(baseline_gpt, fp, "zuco_gpt_base")
decompose_sfl(baseline_small, fp, "zuco_small_base")
decompose_sfl(baseline_med, fp, "zuco_med_base")
decompose_sfl(baseline_large, fp, "zuco_large_base")
decompose_sfl(baseline_xl, fp, "zuco_xl_base")

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_fpass_all, sd_gpt_fpass_all     = test_model_Ridge(gpt_all, fp, 10, "zuco_gpt_all"); print(m_gpt_fpass_all)
m_small_fpass_all, sd_small_fpass_all = test_model_Ridge(gpt2_small_all, fp, 10, "zuco_small_all"); print(m_small_fpass_all)
m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(gpt2_med_all, fp, 10, "zuco_med_all")
m_larg_fpass_all, sd_larg_fpass_all   = test_model_Ridge(gpt2_large_all, fp, 10, "zuco_large_all")
m_xl_fpass_all, sd_xl_fpass_all       = test_model_Ridge(gpt2_xl_all, fp, 10, "zuco_xl_all")
print(m_gpt_fpass_all, m_small_fpass_all, m_med_fpass_all, m_larg_fpass_all, m_xl_fpass_all)

df_results = pd.DataFrame({
    "base_fpass":[m_gpt_fpass_b, m_small_fpass_b, m_med_fpass_b, m_larg_fpass_b, m_xl_fpass_b],
    "base_fpass_sd": [sd_gpt_fpass_b, sd_small_fpass_b, sd_med_fpass_b, sd_larg_fpass_b, sd_xl_fpass_b],
    "act_fpass":[m_gpt_fpass, m_small_fpass, m_med_fpass, m_larg_fpass, m_xl_fpass],
    "act_fpass_sd": [sd_gpt_fpass, sd_small_fpass, sd_med_fpass, sd_larg_fpass, sd_xl_fpass],
    "all_fpass":[m_gpt_fpass_all, m_small_fpass_all, m_med_fpass_all, m_larg_fpass_all, m_xl_fpass_all],
    "all_fpass_sd":[sd_gpt_fpass_all, sd_small_fpass_all, sd_med_fpass_all, sd_larg_fpass_all, sd_xl_fpass_all]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_results.to_csv("results_zuco.csv")

base_plot(df_results, "act_fpass", "base_fpass", "all_fpass", title="")

############
# stat sig #
############

thebase_fpass = ["zuco_gpt_base", "zuco_small_base", "zuco_med_base", "zuco_large_base", "zuco_xl_base"]
theemb_fpass  = ["zuco_gpt", "zuco_small", "zuco_med", "zuco_large", "zuco_xl"]
all_fpass     = ["zuco_gpt_all", "zuco_small_all", "zuco_med_all", "zuco_large_all", "zuco_xl_all"]

p_fpass = []
for base, emb, all_ in zip(thebase_fpass, theemb_fpass, all_fpass):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_fpass.append([p_base_emb, p_all_base, p_all_emb])
p_fpass = pd.DataFrame(p_fpass, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_fpass.columns:
    df_results["p_fpass_"+df_column] = list(p_fpass[df_column])


#################################
# [removed: SEM follow-up — semantic-norm analysis is not part of the manuscript]
# FOLLOW-UP: SEMANTIC VARIABLES #
#################################

# from utils import plot_sem, get_sem, plot_weights_sem

# wordlist = [word for s in sent for word in s[1:]]
# sem = get_sem(wordlist)
# Missing affective = 69.99
# Missing perceptual = 17.68

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
# small_all_sem = np.concatenate((sem, gpt2_small_all), axis=1)
# med_all_sem   = np.concatenate((sem, gpt2_med_all), axis=1)
# large_all_sem = np.concatenate((sem, gpt2_large_all), axis=1)
# xl_all_sem    = np.concatenate((sem, gpt2_xl_all), axis=1)

# model testing

###############################
# first-pass, activation only #
###############################

# with sem
# m_gpt_fpass_sem, sd_gpt_fpass_sem     = test_model_Ridge(gpt_sem, fp, 10, "zuco_gpt_sem")
# m_small_fpass_sem, sd_small_fpass_sem = test_model_Ridge(small_sem, fp, 10, "zuco_small_sem")
# m_med_fpass_sem, sd_med_fpass_sem     = test_model_Ridge(med_sem, fp, 10, "zuco_med_sem")
# m_larg_fpass_sem, sd_larg_fpass_sem   = test_model_Ridge(large_sem, fp, 10, "zuco_large_sem")
# m_xl_fpass_sem, sd_xl_fpass_sem       = test_model_Ridge(xl_sem, fp, 10, "zuco_xl_sem")
# print(m_gpt_fpass_sem, m_small_fpass_sem, m_med_fpass_sem, m_larg_fpass_sem)

########################
# first-pass, baseline #
########################

# with sem
# m_gpt_fpass_b_sem, sd_gpt_fpass_b_sem     = test_model_Ridge(gpt_base_sem, fp, 10, "zuco_gpt_base_sem")
# m_small_fpass_b_sem, sd_small_fpass_b_sem = test_model_Ridge(small_base_sem, fp, 10, "zuco_small_base_sem")
# m_med_fpass_b_sem, sd_med_fpass_b_sem     = test_model_Ridge(med_base_sem, fp, 10, "zuco_med_base_sem")
# m_larg_fpass_b_sem, sd_larg_fpass_b_sem   = test_model_Ridge(large_base_sem, fp, 10, "zuco_large_base_sem")
# m_xl_fpass_b_sem, sd_xl_fpass_b_sem       = test_model_Ridge(xl_base_sem, fp, 10, "zuco_xl_base_sem")
# print(m_gpt_fpass_b_sem, m_small_fpass_b_sem, m_med_fpass_b_sem, m_larg_fpass_b_sem, m_xl_fpass_b_sem)

###################
# first-pass, all #
###################

# with sem
# m_gpt_fpass_all_sem, sd_gpt_fpass_all_sem     = test_model_Ridge(gpt_all_sem, fp, 10, "zuco_gpt_all_sem")
# m_small_fpass_all_sem, sd_small_fpass_all_sem = test_model_Ridge(small_all_sem, fp, 10, "zuco_small_all_sem")
# m_med_fpass_all_sem, sd_med_fpass_all_sem     = test_model_Ridge(med_all_sem, fp, 10, "zuco_med_all_sem")
# m_larg_fpass_all_sem, sd_larg_fpass_all_sem   = test_model_Ridge(large_all_sem, fp, 10, "zuco_large_all_sem")
# m_xl_fpass_all_sem, sd_xl_fpass_all_sem       = test_model_Ridge(xl_all_sem, fp, 10, "zuco_xl_all_sem")
# print(m_gpt_fpass_all_sem, m_small_fpass_all_sem, m_med_fpass_all_sem, m_larg_fpass_all_sem, m_xl_fpass_all_sem)

# df_results_sem = pd.DataFrame({
#     "base_fpass_sem":[m_gpt_fpass_b_sem, m_small_fpass_b_sem, m_med_fpass_b_sem, m_larg_fpass_b_sem, m_xl_fpass_b_sem],
#     "base_fpass_sem_sd": [sd_gpt_fpass_b_sem, sd_small_fpass_b_sem, sd_med_fpass_b_sem, sd_larg_fpass_b_sem, sd_xl_fpass_b_sem],
#     "act_fpass_sem":[m_gpt_fpass_sem, m_small_fpass_sem, m_med_fpass_sem, m_larg_fpass_sem, m_xl_fpass_sem],
#     "act_fpass_sem_sd": [sd_gpt_fpass_sem, sd_small_fpass_sem, sd_med_fpass_sem, sd_larg_fpass_sem, sd_xl_fpass_sem],
#     "all_fpass_sem":[m_gpt_fpass_all_sem, m_small_fpass_all_sem, m_med_fpass_all_sem, m_larg_fpass_all_sem, m_xl_fpass_all_sem],
#     "all_fpass_sem_sd":[sd_gpt_fpass_all_sem, sd_small_fpass_all_sem, sd_med_fpass_all_sem, sd_larg_fpass_all_sem, sd_xl_fpass_all_sem]
#     })
# df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

# df_results = pd.read_csv("results_zuco.csv")
# df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

# merged = pd.concat([df_results, df_results_sem], axis=1)
#merged.to_csv("SEM/merged_zuco.csv")

# plot_sem(merged, "act_fpass", "base_fpass", "all_fpass", "act_fpass_sem", "base_fpass_sem", "all_fpass_sem", title="zuco", subtract=True)


########################
# for last probing exp #
########################

probe(med_filt[filt_na], fp, 10, "zuco_med_probing")
# print(m_med_fpass)


# from joblib import load
# pca = load('/home/dev/Documents/PhD/MIT/project/probing_LMs/pca.pkl')
# med_filt_pca = pca.transform(med_filt)

# # first-pass, activation only
# m_med_fpass, sd_med_fpass = test_model_Ridge(med_filt_pca[filt_na], fp, 10, "zuco_med_probing_pca")

##########################
# end of weight analysis #
##########################
# save med_filt, y
dataname = "zuco"
tosave = [med_filt[filt_na], fp]
# with open(f"../probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################################
###########################
# Other fixation measures #
###########################

# first fixation, activation only
m_gpt_ffix, sd_gpt_ffix     = test_model_Ridge(gpt_filt[filt_na], ff, 10, "zuco_ffix_gpt")
m_small_ffix, sd_small_ffix = test_model_Ridge(small_filt[filt_na], ff, 10, "zuco_ffix_small")
m_med_ffix, sd_med_ffix     = test_model_Ridge(med_filt[filt_na], ff, 10, "zuco_ffix_med")
m_larg_ffix, sd_larg_ffix   = test_model_Ridge(large_filt[filt_na], ff, 10, "zuco_ffix_large")
m_xl_ffix, sd_xl_ffix       = test_model_Ridge(xl_filt[filt_na], ff, 10, "zuco_ffix_xl")
print(m_gpt_ffix, m_small_ffix, m_med_ffix, m_larg_ffix)

m_gpt_totfix, sd_gpt_totfix     = test_model_Ridge(gpt_filt[filt_na], tf, 10, "zuco_totfix_gpt")
m_small_totfix, sd_small_totfix = test_model_Ridge(small_filt[filt_na], tf, 10, "zuco_totfix_small")
m_med_totfix, sd_med_totfix     = test_model_Ridge(med_filt[filt_na], tf, 10, "zuco_totfix_med")
m_larg_totfix, sd_larg_totfix   = test_model_Ridge(large_filt[filt_na], tf, 10, "zuco_totfix_large")
m_xl_totfix, sd_xl_totfix       = test_model_Ridge(xl_filt[filt_na], tf, 10, "zuco_totfix_xl")
print(m_gpt_totfix, m_small_totfix, m_med_totfix, m_larg_totfix)

# first fixation, baseline only
baseline_gpt   = np.array(list(zip(df["s_GPT"], df["freq"], df["len"])))[filt_na]
baseline_small = np.array(list(zip(df["s_small"], df["freq"], df["len"])))[filt_na]
baseline_med   = np.array(list(zip(df["s_med"], df["freq"], df["len"])))[filt_na]
baseline_large = np.array(list(zip(df["s_large"], df["freq"], df["len"])))[filt_na]
baseline_xl    = np.array(list(zip(df["s_xl"], df["freq"], df["len"])))[filt_na]

m_gpt_ffix_b, sd_gpt_ffix_b     = test_model_Ridge(baseline_gpt, ff, 10, "zuco_ffix_gpt_base")
m_small_ffix_b, sd_small_ffix_b = test_model_Ridge(baseline_small, ff, 10, "zuco_ffix_small_base")
m_med_ffix_b, sd_med_ffix_b     = test_model_Ridge(baseline_med, ff, 10, "zuco_ffix_med_base")
m_larg_ffix_b, sd_larg_ffix_b   = test_model_Ridge(baseline_large, ff, 10, "zuco_ffix_large_base")
m_xl_ffix_b, sd_xl_ffix_b       = test_model_Ridge(baseline_xl, ff, 10, "zuco_ffix_xl_base")
print(m_gpt_ffix_b, m_small_ffix_b, m_med_ffix_b, m_larg_ffix_b, m_xl_ffix_b)

m_gpt_totfix_b, sd_gpt_totfix_b     = test_model_Ridge(baseline_gpt, tf, 10, "zuco_totfix_gpt_base")
m_small_totfix_b, sd_small_totfix_b = test_model_Ridge(baseline_small, tf, 10, "zuco_totfix_small_base")
m_med_totfix_b, sd_med_totfix_b     = test_model_Ridge(baseline_med, tf, 10, "zuco_totfix_med_base")
m_larg_totfix_b, sd_larg_totfix_b   = test_model_Ridge(baseline_large, tf, 10, "zuco_totfix_large_base")
m_xl_totfix_b, sd_xl_totfix_b       = test_model_Ridge(baseline_xl, tf, 10, "zuco_totfix_xl_base")
print(m_gpt_totfix_b, m_small_totfix_b, m_med_totfix_b, m_larg_totfix_b, m_xl_totfix_b)

# decompose_sfl
decompose_sfl(baseline_gpt, ff, "zuco_ffix_gpt_base")
decompose_sfl(baseline_small, ff, "zuco_ffix_small_base")
decompose_sfl(baseline_med, ff, "zuco_ffix_med_base")
decompose_sfl(baseline_large, ff, "zuco_ffix_large_base")
decompose_sfl(baseline_xl, ff, "zuco_ffix_xl_base")

decompose_sfl(baseline_gpt, tf, "zuco_totfix_gpt_base")
decompose_sfl(baseline_small, tf, "zuco_totfix_small_base")
decompose_sfl(baseline_med, tf, "zuco_totfix_med_base")
decompose_sfl(baseline_large, tf, "zuco_totfix_large_base")
decompose_sfl(baseline_xl, tf, "zuco_totfix_xl_base")

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_ffix_all, sd_gpt_ffix_all     = test_model_Ridge(gpt_all, ff, 10, "zuco_ffix_gpt_all"); print(m_gpt_ffix_all)
m_small_ffix_all, sd_small_ffix_all = test_model_Ridge(gpt2_small_all, ff, 10, "zuco_ffix_small_all"); print(m_small_ffix_all)
m_med_ffix_all, sd_med_ffix_all     = test_model_Ridge(gpt2_med_all, ff, 10, "zuco_ffix_med_all")
m_larg_ffix_all, sd_larg_ffix_all   = test_model_Ridge(gpt2_large_all, ff, 10, "zuco_ffix_large_all")
m_xl_ffix_all, sd_xl_ffix_all       = test_model_Ridge(gpt2_xl_all, ff, 10, "zuco_ffix_xl_all")
print(m_gpt_ffix_all, m_small_ffix_all, m_med_ffix_all, m_larg_ffix_all, m_xl_ffix_all)

m_gpt_totfix_all, sd_gpt_totfix_all     = test_model_Ridge(gpt_all, tf, 10, "zuco_totfix_gpt_all"); print(m_gpt_totfix_all)
m_small_totfix_all, sd_small_totfix_all = test_model_Ridge(gpt2_small_all, tf, 10, "zuco_totfix_small_all"); print(m_small_totfix_all)
m_med_totfix_all, sd_med_totfix_all     = test_model_Ridge(gpt2_med_all, tf, 10, "zuco_totfix_med_all")
m_larg_totfix_all, sd_larg_totfix_all   = test_model_Ridge(gpt2_large_all, tf, 10, "zuco_totfix_large_all")
m_xl_totfix_all, sd_xl_totfix_all       = test_model_Ridge(gpt2_xl_all, tf, 10, "zuco_totfix_xl_all")
print(m_gpt_totfix_all, m_small_totfix_all, m_med_totfix_all, m_larg_totfix_all, m_xl_totfix_all)

df_results = pd.DataFrame({
    "base_ffix":[m_gpt_ffix_b, m_small_ffix_b, m_med_ffix_b, m_larg_ffix_b, m_xl_ffix_b],
    "base_ffix_sd": [sd_gpt_ffix_b, sd_small_ffix_b, sd_med_ffix_b, sd_larg_ffix_b, sd_xl_ffix_b],
    "act_ffix":[m_gpt_ffix, m_small_ffix, m_med_ffix, m_larg_ffix, m_xl_ffix],
    "act_ffix_sd": [sd_gpt_ffix, sd_small_ffix, sd_med_ffix, sd_larg_ffix, sd_xl_ffix],
    "all_ffix":[m_gpt_ffix_all, m_small_ffix_all, m_med_ffix_all, m_larg_ffix_all, m_xl_ffix_all],
    "all_ffix_sd":[sd_gpt_ffix_all, sd_small_ffix_all, sd_med_ffix_all, sd_larg_ffix_all, sd_xl_ffix_all],
    "base_totfix":[m_gpt_totfix_b, m_small_totfix_b, m_med_totfix_b, m_larg_totfix_b, m_xl_totfix_b],
    "base_totfix_sd": [sd_gpt_totfix_b, sd_small_totfix_b, sd_med_totfix_b, sd_larg_totfix_b, sd_xl_totfix_b],
    "act_totfix":[m_gpt_totfix, m_small_totfix, m_med_totfix, m_larg_totfix, m_xl_totfix],
    "act_totfix_sd": [sd_gpt_totfix, sd_small_totfix, sd_med_totfix, sd_larg_totfix, sd_xl_totfix],
    "all_totfix":[m_gpt_totfix_all, m_small_totfix_all, m_med_totfix_all, m_larg_totfix_all, m_xl_totfix_all],
    "all_totfix_sd":[sd_gpt_totfix_all, sd_small_totfix_all, sd_med_totfix_all, sd_larg_totfix_all, sd_xl_totfix_all]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

thebase_ffix = ["zuco_gpt_base", "zuco_small_base", "zuco_med_base", "zuco_large_base", "zuco_xl_base"]
theemb_ffix  = ["zuco_gpt", "zuco_small", "zuco_med", "zuco_large", "zuco_xl"]
all_ffix     = ["zuco_gpt_all", "zuco_small_all", "zuco_med_all", "zuco_large_all", "zuco_xl_all"]

p_ffix = []
for base, emb, all_ in zip(thebase_ffix, theemb_ffix, all_ffix):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_ffix.append([p_base_emb, p_all_base, p_all_emb])
p_ffix = pd.DataFrame(p_ffix, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_ffix.columns:
    df_results["p_ffix_"+df_column] = list(p_ffix[df_column])
    
thebase_totfix = ["zuco_gpt_base", "zuco_small_base", "zuco_med_base", "zuco_large_base", "zuco_xl_base"]
theemb_totfix  = ["zuco_gpt", "zuco_small", "zuco_med", "zuco_large", "zuco_xl"]
all_totfix     = ["zuco_gpt_all", "zuco_small_all", "zuco_med_all", "zuco_large_all", "zuco_xl_all"]

p_totfix = []
for base, emb, all_ in zip(thebase_totfix, theemb_totfix, all_totfix):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_totfix.append([p_base_emb, p_all_base, p_all_emb])
p_totfix = pd.DataFrame(p_totfix, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_totfix.columns:
    df_results["p_totfix_"+df_column] = list(p_totfix[df_column])
    
df_results.to_csv("results_zuco_ff.csv")