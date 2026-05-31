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

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, plot_sem, get_sem, plot_weights_sem, decompose_sfl


df_all = pd.read_csv("UCL_reading_all.csv")

# get surprisal from LMs

def get_p(sent, w):
    inpts = toker(sent, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inpts).logits[:, -1, :]
    target_id = toker(" "+w)["input_ids"][0]
    p = softmax(logits[0])[target_id].item()
    return p

def get_p2(sent, w):
    inpts = toker(sent, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inpts).logits[:, -1, :]
    target = toker(" "+w)["input_ids"]
    if len(target) == 1:
        target_id = target[0]
        p = softmax(logits[0])[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = softmax(logits[0])[target_id].item()
        out_p.append(p)
        sent = sent+toker.decode(target_id)
        for token in target[1:]:
            t = toker.decode(token)
            p = get_p(sent, t)
            out_p.append(p)
            #print(sent, "--"+t, p)
            sent = sent+t
        p_multi = np.prod(out_p)
        return p_multi, 1

def surp_df(colname):
    out = []
    for index, row in tqdm(df_all.iterrows(), total=len(df_all)):
        if row["item"] == row["item"]: # check for NaN
            p, _ = get_p2(row["item"], row["word"])
            out.append(-np.log(p))
        else:
            out.append(np.nan)
    df_all[colname] = out

# GPTNeo
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
surp_df("s_GPTNeo")

# GPTNeo med
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
surp_df("s_GPTNeo_med")

# GPTNeo large
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7b", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7b")
surp_df("s_GPTNeo_large")

# Pythia
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-v0", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-v0")
surp_df("s_Pythia")

# Pythia 160M
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m-v0", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-v0")
surp_df("s_Pythia_160M")

# Pythia 410M
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m-v0", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-v0")
surp_df("s_Pythia_410M")

# Pythia 1B
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-v0", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-v0")
surp_df("s_Pythia_1B")

# Pythia 1.4B
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-v0", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-v0")
surp_df("s_Pythia_1.4B")

# Pythia 2.8B
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b-v0", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-v0")
surp_df("s_Pythia_2.8B")

# Transformer-XL
model = AutoModelForCausalLM.from_pretrained("transfo-xl-wt103", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("transfo-xl-wt103")
surp_df("s_transfo_xl")


#df_all.to_csv("UCL_reading_additional_models.csv", index=False)

################################################################################################

df_all = pd.read_csv("UCL_reading_additional_models.csv")

########################################################################################
# (changed later) S and F should have the same base of the log (nat) for comparability #
########################################################################################

df_noinit = df_all.dropna(subset=["s_GPT2_small"]) # remove sentence-starting
corr_pair(df_noinit[["logfreq", "s_GPT2_small", "s_GPTNeo", "s_Pythia", "s_transfo_xl"]])

filtr = []
for index, row in df_noinit.iterrows():
    if "." in row.word:
        filtr.append(False)
    elif "," in row.word:
        filtr.append(False)
    else:
        filtr.append(True)
filtr = np.array(filtr)

##############################
# EXTRACTING REPRESENTATIONS #
##############################

# https://huggingface.co/transformers/v2.2.0/pretrained_models.html

import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

#######
# GPT #
#######
    
# OPT, small (125M 12 12 768) (Model #L #H dmodel)
if path.isfile(dirName+'/activations_opt.json') == False:
    transformers_extractor.extract_representations('facebook/opt-125m',
        "corpora/sentences.txt",
        dirName+'/activations_opt.json',
        aggregation="average"
    )
    
# GPT-Neo
if path.isfile(dirName+'/activations_gptNeo.json') == False:
    transformers_extractor.extract_representations("EleutherAI/gpt-neo-125m",
        "corpora/sentences.txt",
        dirName+'/activations_gptNeo.json',
        aggregation="average"
    )
    
if path.isfile(dirName+'/activations_gptNeo_med.json') == False:
    transformers_extractor.extract_representations("EleutherAI/gpt-neo-1.3B",
        "corpora/sentences.txt",
        dirName+'/activations_gptNeo_med.json',
        aggregation="average"
    )

if path.isfile(dirName+'/activations_gptNeo_large.json') == False:
    transformers_extractor.extract_representations("EleutherAI/gpt-neo-2.7b",
        "corpora/sentences.txt",
        dirName+'/activations_gptNeo_large.json',
        aggregation="average"
    )

# HAVE NOT RUN YET
# # Falcon
# if path.isfile(dirName+'/activations_falcon.json') == False:
#     transformers_extractor.extract_representations('tiiuae/falcon-7b',
#         "corpora/sentences.txt",
#         dirName+'/activations_falcon.json',
#         aggregation="average"
#     )

# EleutherAI/pythia-70m-v0
# 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B
if path.isfile(dirName+'/activations_pythia.json') == False:
    transformers_extractor.extract_representations("EleutherAI/pythia-70m-v0",
        "corpora/sentences.txt",
        dirName+'/activations_pythia.json',
        aggregation="average"
    )

if path.isfile(dirName+'/activations_pythia_160M.json') == False:
    transformers_extractor.extract_representations("EleutherAI/pythia-160m-v0",
        "corpora/sentences.txt",
        dirName+'/activations_pythia_160M.json',
        aggregation="average"
    )
    
if path.isfile(dirName+'/activations_pythia_410M.json') == False:
    transformers_extractor.extract_representations("EleutherAI/pythia-410m-v0",
        "corpora/sentences.txt",
        dirName+'/activations_pythia_410M.json',
        aggregation="average"
    )
    
if path.isfile(dirName+'/activations_pythia_1B.json') == False:
    transformers_extractor.extract_representations("EleutherAI/pythia-1b-v0",
        "corpora/sentences.txt",
        dirName+'/activations_pythia_1B.json',
        aggregation="average"
    )
    
if path.isfile(dirName+'/activations_pythia_1.4B.json') == False:
    transformers_extractor.extract_representations("EleutherAI/pythia-1.4b-v0",
        "corpora/sentences.txt",
        dirName+'/activations_pythia_1.4B.json',
        aggregation="average"
    )
    
if path.isfile(dirName+'/activations_pythia_2.8B.json') == False:
    transformers_extractor.extract_representations("EleutherAI/pythia-2.8b-v0",
        "corpora/sentences.txt",
        dirName+'/activations_pythia_2.8B.json',
        aggregation="average"
    )


# OPT
# 13 layers (w/ emb), 768
act_opt, num_layers_opt = data_loader.load_activations(dirName+'/activations_opt.json', 768, is_brnn=False) 

# GPT-NEO
# 13 layers (w/ emb), 768
act_neo, num_layers_neo = data_loader.load_activations(dirName+'/activations_gptNeo.json', 768, is_brnn=False) 

# 25 layers (w/ emb), 2048
act_neo_med, num_layers_neo_med = data_loader.load_activations(dirName+'/activations_gptNeo_med.json', 2048, is_brnn=False) 

# 33 layers (w/ emb), 2560
act_neo_large, num_layers_neo_large = data_loader.load_activations(dirName+'/activations_gptNeo_large.json', 2560, is_brnn=False) 

# PYTHIA
# 7 layers, 512
act_pyt, num_layers_pyt = data_loader.load_activations(dirName+'/activations_pythia.json', 512, is_brnn=False) 

# 13 layers, 768
act_pyt_160m, num_layers_pyt_160m = data_loader.load_activations(dirName+'/activations_pythia_160M.json', 768, is_brnn=False) 

# 25 layers, 1024
act_pyt_410m, num_layers_pyt_410m = data_loader.load_activations(dirName+'/activations_pythia_410M.json', 1024, is_brnn=False) 

# 17 layers, 2048
act_pyt_1b, num_layers_pyt_1b = data_loader.load_activations(dirName+'/activations_pythia_1B.json', 2048, is_brnn=False) 

# 25 layers, 2048
act_pyt_1dot4b, num_layers_pyt_1dot4b = data_loader.load_activations(dirName+'/activations_pythia_1.4B.json', 2048, is_brnn=False) 

# 33 layers, 2560
act_pyt_2dot8b, num_layers_pyt_2dot8b = data_loader.load_activations(dirName+'/activations_pythia_2.8B.json', 2560, is_brnn=False) 

# TRANSFORMER-XL
# 19 layers, 1024
act_transfo_xl, num_layers_transfo_xl = data_loader.load_activations(dirName+'/activations_transf_xl.json', 1024, is_brnn=False) 


available = set(df_all.sent_nr-1) 

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act_eye(act_list, layer, dim):
    out = []
    for i in available:
        act_sent = act_list[i][1:]
        out.extend([get_layer(v, layer, dim) for v in act_sent])
    return np.array(out)

neo        = get_act_eye(act_neo, 12, 768)
neo_med    = get_act_eye(act_neo_med, 24, 2048)
neo_large  = get_act_eye(act_neo_large, 32, 2560)
pyt        = get_act_eye(act_pyt, 6, 512)
pyt_160m   = get_act_eye(act_pyt_160m, 12, 768)
pyt_410m   = get_act_eye(act_pyt_410m, 24, 1024)
pyt_1b     = get_act_eye(act_pyt_1b, 16, 2048)
pyt_1dot4b = get_act_eye(act_pyt_1dot4b, 24, 2048)
pyt_2dot8b = get_act_eye(act_pyt_2dot8b, 32, 2560)
transfo_xl = get_act_eye(act_transfo_xl, 18, 1024)

neo_f        = neo[filtr]
neo_med_f    = neo_med[filtr]
neo_large_f  = neo_large[filtr]
pyt_f        = pyt[filtr]
pyt_160m_f   = pyt_160m[filtr]
pyt_410m_f   = pyt_410m[filtr]
pyt_1b_f     = pyt_1b[filtr]
pyt_1dot4b_f = pyt_1dot4b[filtr]
pyt_2dot8b_f = pyt_2dot8b[filtr]
transfo_xl_f = transfo_xl[filtr]

fp   = np.array(df_noinit["RTfirstpass"])
n400 = np.array(df_noinit["N400"])[filtr]
p600 = np.array(df_noinit["P600"])[filtr]
spr  = np.array(df_noinit["RT"])

#################################################
# need to account for spillover effects in SPR! #
#################################################

out = []
masks = []
for sent in set(df_noinit.sent_nr):
    temp = df_noinit[df_noinit.sent_nr == sent]
    idxmax = max(temp.word_pos)+1
    for idx in range(4, idxmax):
        twobef = temp[temp.word_pos == idx-2]
        onebef = temp[temp.word_pos == idx-1]
        actual = temp[temp.word_pos == idx]
        a = twobef.values.tolist()[0]+onebef.values.tolist()[0]+actual.values.tolist()[0]
        out.append(a)
        masks.append(True)
cols = [colname + "-2" for colname in df_noinit.columns.tolist()] + [colname + "-1" for colname in df_noinit.columns.tolist()] + df_noinit.columns.tolist()
df_spill = pd.DataFrame(out, columns = cols)

spr_spill = np.array(df_spill["RT"])

# filtering embeddings
count = 0
to_filt = []
for sent in set(df_noinit.sent_nr):
    temp = df_noinit[df_noinit.sent_nr == sent]
    idxmax = max(temp.word_pos)+1
    for idx in range(2, idxmax):
        if idx > 3:
            to_filt.append([count, count-1, count-2])
        count += 1

def spill_embeddings(emb, filtlist):
    filtered = []
    for idx1, idx2, idx3 in filtlist:
        filtered.append(np.concatenate([emb[idx1], emb[idx2], emb[idx3]]))
    return np.array(filtered)
            

neo_spill        = spill_embeddings(neo, to_filt)
neo_med_spill    = spill_embeddings(neo_med, to_filt)
neo_large_spill  = spill_embeddings(neo_large, to_filt)
pyt_spill        = spill_embeddings(pyt, to_filt)
pyt_160m_spill   = spill_embeddings(pyt_160m, to_filt)
pyt_410m_spill   = spill_embeddings(pyt_410m, to_filt)
pyt_1b_spill     = spill_embeddings(pyt_1b, to_filt)
pyt_1dot4b_spill = spill_embeddings(pyt_1dot4b, to_filt)
pyt_2dot8b_spill = spill_embeddings(pyt_2dot8b, to_filt)
transfo_xl_spill = spill_embeddings(transfo_xl, to_filt)

####################
# Ridge regression #
####################

####################
# ACTIVATIONS ONLY #
####################

# first-pass, activation only
m_neo_fpass, sd_neo_fpass               = test_model_Ridge(neo, fp, 10, "UCL_fpass_neo")
m_neo_med_fpass, sd_neo_med_fpass       = test_model_Ridge(neo_med, fp, 10, "UCL_fpass_neo_med")
m_neo_large_fpass, sd_neo_large_fpass   = test_model_Ridge(neo_large, fp, 10, "UCL_fpass_neo_large")
m_pyt_fpass, sd_pyt_fpass               = test_model_Ridge(pyt, fp, 10, "UCL_fpass_pyth")
m_pyt_160m_fpass, sd_pyt_160m_fpass     = test_model_Ridge(pyt_160m, fp, 10, "UCL_fpass_pyth_160M")
m_pyt_410m_fpass, sd_pyt_410m_fpass     = test_model_Ridge(pyt_410m, fp, 10, "UCL_fpass_pyth_410M")
m_pyt_1b_fpass, sd_pyt_1b_fpass         = test_model_Ridge(pyt_1b, fp, 10, "UCL_fpass_pyth_1B")
m_pyt_1dot4b_fpass, sd_pyt_1dot4b_fpass = test_model_Ridge(pyt_1dot4b, fp, 10, "UCL_fpass_pyth_1.4B")
m_pyt_2dot8b_fpass, sd_pyt_2dot8b_fpass = test_model_Ridge(pyt_2dot8b, fp, 10, "UCL_fpass_pyth_2.8B")
m_transfo_xl_fpass, sd_transfo_xl_fpass = test_model_Ridge(transfo_xl, fp, 10, "UCL_fpass_transfo_xl")
print(m_neo_fpass,m_neo_med_fpass,m_neo_large_fpass,m_pyt_fpass,m_pyt_160m_fpass,m_pyt_410m_fpass,m_pyt_1b_fpass,m_pyt_1dot4b_fpass,m_pyt_2dot8b_fpass,m_transfo_xl_fpass)
print(sd_neo_fpass,sd_neo_med_fpass,sd_neo_large_fpass,sd_pyt_fpass,sd_pyt_160m_fpass,sd_pyt_410m_fpass,sd_pyt_1b_fpass,sd_pyt_1dot4b_fpass,sd_pyt_2dot8b_fpass,sd_transfo_xl_fpass)

# n400, activation only
m_neo_n400, sd_neo_n400               = test_model_Ridge(neo_f, n400, 10, "UCL_n400_neo")
m_neo_med_n400, sd_neo_med_n400       = test_model_Ridge(neo_med_f, n400, 10, "UCL_n400_neo_med")
m_neo_large_n400, sd_neo_large_n400   = test_model_Ridge(neo_large_f, n400, 10, "UCL_n400_neo_large")
m_pyt_n400, sd_pyt_n400               = test_model_Ridge(pyt_f, n400, 10, "UCL_n400_pyth")
m_pyt_160m_n400, sd_pyt_160m_n400     = test_model_Ridge(pyt_160m_f, n400, 10, "UCL_n400_pyth_160M")
m_pyt_410m_n400, sd_pyt_410m_n400     = test_model_Ridge(pyt_410m_f, n400, 10, "UCL_n400_pyth_410M")
m_pyt_1b_n400, sd_pyt_1b_n400         = test_model_Ridge(pyt_1b_f, n400, 10, "UCL_n400_pyth_1B")
m_pyt_1dot4b_n400, sd_pyt_1dot4b_n400 = test_model_Ridge(pyt_1dot4b_f, n400, 10, "UCL_n400_pyth_1.4B")
m_pyt_2dot8b_n400, sd_pyt_2dot8b_n400 = test_model_Ridge(pyt_2dot8b_f, n400, 10, "UCL_n400_pyth_2.8B")
m_transfo_xl_n400, sd_transfo_xl_n400 = test_model_Ridge(transfo_xl_f, n400, 10, "UCL_n400_transfo_xl")
print(m_neo_n400,m_neo_med_n400,m_neo_large_n400,m_pyt_n400,m_pyt_160m_n400,m_pyt_410m_n400,m_pyt_1b_n400,m_pyt_1dot4b_n400,m_pyt_2dot8b_n400,m_transfo_xl_n400)
print(sd_neo_n400,sd_neo_med_n400,sd_neo_large_n400,sd_pyt_n400,sd_pyt_160m_n400,sd_pyt_410m_n400,sd_pyt_1b_n400,sd_pyt_1dot4b_n400,sd_pyt_2dot8b_n400,sd_transfo_xl_n400)

# p600, activations only
m_neo_p600, sd_neo_p600               = test_model_Ridge(neo_f, p600, 10, "UCL_p600_neo")
m_neo_med_p600, sd_neo_med_p600       = test_model_Ridge(neo_med_f, p600, 10, "UCL_p600_neo_med")
m_neo_large_p600, sd_neo_large_p600   = test_model_Ridge(neo_large_f, p600, 10, "UCL_p600_neo_large")
m_pyt_p600, sd_pyt_p600               = test_model_Ridge(pyt_f, p600, 10, "UCL_p600_pyth")
m_pyt_160m_p600, sd_pyt_160m_p600     = test_model_Ridge(pyt_160m_f, p600, 10, "UCL_p600_pyth_160M")
m_pyt_410m_p600, sd_pyt_410m_p600     = test_model_Ridge(pyt_410m_f, p600, 10, "UCL_p600_pyth_410M")
m_pyt_1b_p600, sd_pyt_1b_p600         = test_model_Ridge(pyt_1b_f, p600, 10, "UCL_p600_pyth_1B")
m_pyt_1dot4b_p600, sd_pyt_1dot4b_p600 = test_model_Ridge(pyt_1dot4b_f, p600, 10, "UCL_p600_pyth_1.4B")
m_pyt_2dot8b_p600, sd_pyt_2dot8b_p600 = test_model_Ridge(pyt_2dot8b_f, p600, 10, "UCL_p600_pyth_2.8B")
m_transfo_xl_p600, sd_transfo_xl_p600 = test_model_Ridge(transfo_xl_f, p600, 10, "UCL_p600_transfo_xl")
print(m_neo_p600,m_neo_med_p600,m_neo_large_p600,m_pyt_p600,m_pyt_160m_p600,m_pyt_410m_p600,m_pyt_1b_p600,m_pyt_1dot4b_p600,m_pyt_2dot8b_p600,m_transfo_xl_p600)
print(sd_neo_p600,sd_neo_med_p600,sd_neo_large_p600,sd_pyt_p600,sd_pyt_160m_p600,sd_pyt_410m_p600,sd_pyt_1b_p600,sd_pyt_1dot4b_p600,sd_pyt_2dot8b_p600,sd_transfo_xl_p600)

# SPR, spill
m_neo_spr, sd_neo_spr               = test_model_Ridge(neo_spill, spr_spill, 10, "UCL_spr_neo")
m_neo_med_spr, sd_neo_med_spr       = test_model_Ridge(neo_med_spill, spr_spill, 10, "UCL_spr_neo_med")
m_neo_large_spr, sd_neo_large_spr   = test_model_Ridge(neo_large_spill, spr_spill, 10, "UCL_spr_neo_large")
m_pyt_spr, sd_pyt_spr               = test_model_Ridge(pyt_spill, spr_spill, 10, "UCL_spr_pyth")
m_pyt_160m_spr, sd_pyt_160m_spr     = test_model_Ridge(pyt_160m_spill, spr_spill, 10, "UCL_spr_pyth_160M")
m_pyt_410m_spr, sd_pyt_410m_spr     = test_model_Ridge(pyt_410m_spill, spr_spill, 10, "UCL_spr_pyth_410M")
m_pyt_1b_spr, sd_pyt_1b_spr         = test_model_Ridge(pyt_1b_spill, spr_spill, 10, "UCL_spr_pyth_1B")
m_pyt_1dot4b_spr, sd_pyt_1dot4b_spr = test_model_Ridge(pyt_1dot4b_spill, spr_spill, 10, "UCL_spr_pyth_1.4B")
m_pyt_2dot8b_spr, sd_pyt_2dot8b_spr = test_model_Ridge(pyt_2dot8b_spill, spr_spill, 10, "UCL_spr_pyth_2.8B")
m_transfo_xl_spr, sd_transfo_xl_spr = test_model_Ridge(transfo_xl_spill, spr_spill, 10, "UCL_spr_transfo_xl")
print(m_neo_spr,m_neo_med_spr,m_neo_large_spr,m_pyt_spr,m_pyt_160m_spr,m_pyt_410m_spr,m_pyt_1b_spr,m_pyt_1dot4b_spr,m_pyt_2dot8b_spr,m_transfo_xl_spr)
print(sd_neo_spr,sd_neo_med_spr,sd_neo_large_spr,sd_pyt_spr,sd_pyt_160m_spr,sd_pyt_410m_spr,sd_pyt_1b_spr,sd_pyt_1dot4b_spr,sd_pyt_2dot8b_spr,sd_transfo_xl_spr)

#################
# BASELINE ONLY #
#################

# first-pass, baseline only
m_neo_fpass_b, sd_neo_fpass_b     = test_model_Ridge(np.array(df_noinit[["s_GPTNeo", "logfreq", "length"]]), fp, 10, "UCL_fpass_neo_base")
m_neo_med_fpass_b, sd_neo_med_fpass_b = test_model_Ridge(np.array(df_noinit[["s_GPTNeo_med", "logfreq", "length"]]), fp, 10, "UCL_fpass_neo_med_base")
m_neo_large_fpass_b, sd_neo_large_fpass_b     = test_model_Ridge(np.array(df_noinit[["s_GPTNeo_large", "logfreq", "length"]]), fp, 10, "UCL_fpass_neo_large_base")
m_pyt_fpass_b, sd_pyt_fpass_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia", "logfreq", "length"]]), fp, 10, "UCL_fpass_pyt_base")
m_pyt_160m_fpass_b, sd_pyt_160m_fpass_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_160M", "logfreq", "length"]]), fp, 10, "UCL_fpass_pyt_160M_base")
m_pyt_410m_fpass_b, sd_pyt_410m_fpass_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_410M", "logfreq", "length"]]), fp, 10, "UCL_fpass_pyt_410M_base")
m_pyt_1b_fpass_b, sd_pyt_1b_fpass_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_1B", "logfreq", "length"]]), fp, 10, "UCL_fpass_pyt_1B_base")
m_pyt_1dot4b_fpass_b, sd_pyt_1dot4b_fpass_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_1.4B", "logfreq", "length"]]), fp, 10, "UCL_fpass_pyt_1.4B_base")
m_pyt_2dot8b_fpass_b, sd_pyt_2dot8b_fpass_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_2.8B", "logfreq", "length"]]), fp, 10, "UCL_fpass_pyt_2.8B_base")
m_transf_xl_fpass_b, sd_transf_xl_fpass_b   = test_model_Ridge(np.array(df_noinit[["s_transfo_xl", "logfreq", "length"]]), fp, 10, "UCL_fpass_transfo_xl_base")
print(m_neo_fpass_b,m_neo_med_fpass_b,m_neo_large_fpass_b,m_pyt_fpass_b,m_pyt_160m_fpass_b,m_pyt_410m_fpass_b,m_pyt_1b_fpass_b,m_pyt_1dot4b_fpass_b,m_pyt_2dot8b_fpass_b,m_transf_xl_fpass_b)
print(sd_neo_fpass_b,sd_neo_med_fpass_b,sd_neo_large_fpass_b,sd_pyt_fpass_b,sd_pyt_160m_fpass_b,sd_pyt_410m_fpass_b,sd_pyt_1b_fpass_b,sd_pyt_1dot4b_fpass_b,sd_pyt_2dot8b_fpass_b,sd_transf_xl_fpass_b)


# N400, baseline only
m_neo_n400_b, sd_neo_n400_b     = test_model_Ridge(np.array(df_noinit[["s_GPTNeo", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_neo_base")
m_neo_med_n400_b, sd_neo_med_n400_b = test_model_Ridge(np.array(df_noinit[["s_GPTNeo_med", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_neo_med_base")
m_neo_large_n400_b, sd_neo_large_n400_b     = test_model_Ridge(np.array(df_noinit[["s_GPTNeo_large", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_neo_large_base")
m_pyt_n400_b, sd_pyt_n400_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_pyt_base")
m_pyt_160m_n400_b, sd_pyt_160m_n400_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_160M", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_pyt_160M_base")
m_pyt_410m_n400_b, sd_pyt_410m_n400_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_410M", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_pyt_410M_base")
m_pyt_1b_n400_b, sd_pyt_1b_n400_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_1B", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_pyt_1B_base")
m_pyt_1dot4b_n400_b, sd_pyt_1dot4b_n400_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_1.4B", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_pyt_1.4B_base")
m_pyt_2dot8b_n400_b, sd_pyt_2dot8b_n400_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_2.8B", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_pyt_2.8B_base")
m_transf_xl_n400_b, sd_transf_xl_n400_b   = test_model_Ridge(np.array(df_noinit[["s_transfo_xl", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_transfo_xl_base")
print(m_neo_n400_b,m_neo_med_n400_b,m_neo_large_n400_b,m_pyt_n400_b,m_pyt_160m_n400_b,m_pyt_410m_n400_b,m_pyt_1b_n400_b,m_pyt_1dot4b_n400_b,m_pyt_2dot8b_n400_b,m_transf_xl_n400_b)
print(sd_neo_n400_b,sd_neo_med_n400_b,sd_neo_large_n400_b,sd_pyt_n400_b,sd_pyt_160m_n400_b,sd_pyt_410m_n400_b,sd_pyt_1b_n400_b,sd_pyt_1dot4b_n400_b,sd_pyt_2dot8b_n400_b,sd_transf_xl_n400_b)


# p600, baseline only
m_neo_p600_b, sd_neo_p600_b     = test_model_Ridge(np.array(df_noinit[["s_GPTNeo", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_neo_base")
m_neo_med_p600_b, sd_neo_med_p600_b = test_model_Ridge(np.array(df_noinit[["s_GPTNeo_med", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_neo_med_base")
m_neo_large_p600_b, sd_neo_large_p600_b     = test_model_Ridge(np.array(df_noinit[["s_GPTNeo_large", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_neo_large_base")
m_pyt_p600_b, sd_pyt_p600_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_pyt_base")
m_pyt_160m_p600_b, sd_pyt_160m_p600_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_160M", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_pyt_160M_base")
m_pyt_410m_p600_b, sd_pyt_410m_p600_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_410M", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_pyt_410M_base")
m_pyt_1b_p600_b, sd_pyt_1b_p600_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_1B", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_pyt_1B_base")
m_pyt_1dot4b_p600_b, sd_pyt_1dot4b_p600_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_1.4B", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_pyt_1.4B_base")
m_pyt_2dot8b_p600_b, sd_pyt_2dot8b_p600_b   = test_model_Ridge(np.array(df_noinit[["s_Pythia_2.8B", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_pyt_2.8B_base")
m_transf_xl_p600_b, sd_transf_xl_p600_b   = test_model_Ridge(np.array(df_noinit[["s_transfo_xl", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_transfo_xl_base")
print(m_neo_n400_b,m_neo_med_n400_b,m_neo_large_n400_b,m_pyt_n400_b,m_pyt_160m_n400_b,m_pyt_410m_n400_b,m_pyt_1b_n400_b,m_pyt_1dot4b_n400_b,m_pyt_2dot8b_n400_b,m_transf_xl_n400_b)
print(sd_neo_n400_b,sd_neo_med_n400_b,sd_neo_large_n400_b,sd_pyt_n400_b,sd_pyt_160m_n400_b,sd_pyt_410m_n400_b,sd_pyt_1b_n400_b,sd_pyt_1dot4b_n400_b,sd_pyt_2dot8b_n400_b,sd_transf_xl_n400_b)


# spr_spill, baseline only
m_neo_spr_b, sd_neo_spr_b              = test_model_Ridge(np.array(df_spill[["s_GPTNeo", "logfreq", "length",
                                                                              "s_GPTNeo-1", "logfreq-1", "length-1",
                                                                              "s_GPTNeo-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_neo_base")
m_neo_med_spr_b, sd_neo_med_spr_b      = test_model_Ridge(np.array(df_spill[["s_GPTNeo_med", "logfreq", "length",
                                                                             "s_GPTNeo_med-1", "logfreq-1", "length-1",
                                                                             "s_GPTNeo_med-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_neo_med_base")
m_neo_large_spr_b, sd_neo_large_spr_b  = test_model_Ridge(np.array(df_spill[["s_GPTNeo_large", "logfreq", "length",
                                                                             "s_GPTNeo_large-1", "logfreq-1", "length-1",
                                                                             "s_GPTNeo_large-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_neo_large_base")
m_pyt_spr_b, sd_pyt_spr_b              = test_model_Ridge(np.array(df_spill[["s_Pythia", "logfreq", "length",
                                                                             "s_Pythia-1", "logfreq-1", "length-1",
                                                                             "s_Pythia-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_pyt_base")
m_pyt_160m_spr_b, sd_pyt_160m_spr_b    = test_model_Ridge(np.array(df_spill[["s_Pythia_160M", "logfreq", "length",
                                                                             "s_Pythia_160M-1", "logfreq-1", "length-1",
                                                                             "s_Pythia_160M-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_pyt_160M_base")
m_pyt_410m_spr_b, sd_pyt_410m_spr_b    = test_model_Ridge(np.array(df_spill[["s_Pythia_410M", "logfreq", "length",
                                                                             "s_Pythia_410M-1", "logfreq-1", "length-1",
                                                                             "s_Pythia_410M-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_pyt_410M_base")
m_pyt_1b_spr_b, sd_pyt_1b_spr_b        = test_model_Ridge(np.array(df_spill[["s_Pythia_1B", "logfreq", "length",
                                                                             "s_Pythia_1B-1", "logfreq-1", "length-1",
                                                                             "s_Pythia_1B-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_pyt_1B_base")
m_pyt_1dot4b_spr_b, sd_pyt_1dot4b_spr_b= test_model_Ridge(np.array(df_spill[["s_Pythia_1.4B", "logfreq", "length",
                                                                             "s_Pythia_1.4B-1", "logfreq-1", "length-1",
                                                                             "s_Pythia_1.4B-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_pyt_1.4B_base")
m_pyt_2dot8b_spr_b, sd_pyt_2dot8b_spr_b= test_model_Ridge(np.array(df_spill[["s_Pythia_2.8B", "logfreq", "length",
                                                                             "s_Pythia_2.8B-1", "logfreq-1", "length-1",
                                                                             "s_Pythia_2.8B-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_pyt_2.8B_base")
m_transf_xl_spr_b, sd_transf_xl_spr_b  = test_model_Ridge(np.array(df_spill[["s_transfo_xl", "logfreq", "length",
                                                                             "s_transfo_xl-1", "logfreq-1", "length-1",
                                                                             "s_transfo_xl-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_transfo_xl_base")
print(m_neo_spr_b,m_neo_med_spr_b,m_neo_large_spr_b,m_pyt_spr_b,m_pyt_160m_spr_b,m_pyt_410m_spr_b,m_pyt_1b_spr_b,m_pyt_1dot4b_spr_b,m_pyt_2dot8b_spr_b,m_transf_xl_spr_b)
print(sd_neo_spr_b,sd_neo_med_spr_b,sd_neo_large_spr_b,sd_pyt_spr_b,sd_pyt_160m_spr_b,sd_pyt_410m_spr_b,sd_pyt_1b_spr_b,sd_pyt_1dot4b_spr_b,sd_pyt_2dot8b_spr_b,sd_transf_xl_spr_b)

##########################
# BASELINE + ACTIVATIONS #
##########################

neo_all = np.concatenate((np.array(df_noinit[["s_GPTNeo", "logfreq", "length"]]), neo), axis=1)
neo_med_all = np.concatenate((np.array(df_noinit[["s_GPTNeo_med", "logfreq", "length"]]), neo_med), axis=1)
neo_large_all = np.concatenate((np.array(df_noinit[["s_GPTNeo_large", "logfreq", "length"]]), neo_large), axis=1)
pyt_all = np.concatenate((np.array(df_noinit[["s_Pythia", "logfreq", "length"]]), pyt), axis=1)
pyt_160m_all = np.concatenate((np.array(df_noinit[["s_Pythia_160M", "logfreq", "length"]]), pyt_160m), axis=1)
pyt_410m_all = np.concatenate((np.array(df_noinit[["s_Pythia_410M", "logfreq", "length"]]), pyt_410m), axis=1)
pyt_1b_all = np.concatenate((np.array(df_noinit[["s_Pythia_1B", "logfreq", "length"]]), pyt_1b), axis=1)
pyt_1dot4b_all = np.concatenate((np.array(df_noinit[["s_Pythia_1.4B", "logfreq", "length"]]), pyt_1dot4b), axis=1)
pyt_2dot8b_all = np.concatenate((np.array(df_noinit[["s_Pythia_2.8B", "logfreq", "length"]]), pyt_2dot8b), axis=1)
transfo_xl_all = np.concatenate((np.array(df_noinit[["s_transfo_xl", "logfreq", "length"]]), transfo_xl), axis=1)

# fpass, all
m_neo_fpass_all, sd_neo_fpass_all               = test_model_Ridge(neo_all, fp, 10, "UCL_fpass_neo_all")
m_neo_med_fpass_all, sd_neo_med_fpass_all       = test_model_Ridge(neo_med_all, fp, 10, "UCL_fpass_neo_med_all")
m_neo_large_fpass_all, sd_neo_large_fpass_all   = test_model_Ridge(neo_large_all, fp, 10, "UCL_fpass_neo_large_all")
m_pyt_fpass_all, sd_pyt_fpass_all               = test_model_Ridge(pyt_all, fp, 10, "UCL_fpass_pyth_all")
m_pyt_160m_fpass_all, sd_pyt_160m_fpass_all     = test_model_Ridge(pyt_160m_all, fp, 10, "UCL_fpass_pyth_160M_all")
m_pyt_410m_fpass_all, sd_pyt_410m_fpass_all     = test_model_Ridge(pyt_410m_all, fp, 10, "UCL_fpass_pyth_410M_all")
m_pyt_1b_fpass_all, sd_pyt_1b_fpass_all         = test_model_Ridge(pyt_1b_all, fp, 10, "UCL_fpass_pyth_1B_all")
m_pyt_1dot4b_fpass_all, sd_pyt_1dot4b_fpass_all = test_model_Ridge(pyt_1dot4b_all, fp, 10, "UCL_fpass_pyth_1.4B_all")
m_pyt_2dot8b_fpass_all, sd_pyt_2dot8b_fpass_all = test_model_Ridge(pyt_2dot8b_all, fp, 10, "UCL_fpass_pyth_2.8B_all")
m_transfo_xl_fpass_all, sd_transfo_xl_fpass_all = test_model_Ridge(transfo_xl_all, fp, 10, "UCL_fpass_transfo_xl_all")
print(m_neo_fpass_all,m_neo_med_fpass_all,m_neo_large_fpass_all,m_pyt_fpass_all,m_pyt_160m_fpass_all,m_pyt_410m_fpass_all,m_pyt_1b_fpass_all,m_pyt_1dot4b_fpass_all,m_pyt_2dot8b_fpass_all,m_transfo_xl_fpass_all)
print(sd_neo_fpass_all,sd_neo_med_fpass_all,sd_neo_large_fpass_all,sd_pyt_fpass_all,sd_pyt_160m_fpass_all,sd_pyt_410m_fpass_all,sd_pyt_1b_fpass_all,sd_pyt_1dot4b_fpass_all,sd_pyt_2dot8b_fpass_all,sd_transfo_xl_fpass_all)

# n400
m_neo_n400_all, sd_neo_n400_all               = test_model_Ridge(neo_all[filtr], n400, 10, "UCL_n400_neo_all")
m_neo_med_n400_all, sd_neo_med_n400_all       = test_model_Ridge(neo_med_all[filtr], n400, 10, "UCL_n400_neo_med_all")
m_neo_large_n400_all, sd_neo_large_n400_all   = test_model_Ridge(neo_large_all[filtr], n400, 10, "UCL_n400_neo_large_all")
m_pyt_n400_all, sd_pyt_n400_all               = test_model_Ridge(pyt_all[filtr], n400, 10, "UCL_n400_pyth_all")
m_pyt_160m_n400_all, sd_pyt_160m_n400_all     = test_model_Ridge(pyt_160m_all[filtr], n400, 10, "UCL_n400_pyth_160M_all")
m_pyt_410m_n400_all, sd_pyt_410m_n400_all     = test_model_Ridge(pyt_410m_all[filtr], n400, 10, "UCL_n400_pyth_410M_all")
m_pyt_1b_n400_all, sd_pyt_1b_n400_all         = test_model_Ridge(pyt_1b_all[filtr], n400, 10, "UCL_n400_pyth_1B_all")
m_pyt_1dot4b_n400_all, sd_pyt_1dot4b_n400_all = test_model_Ridge(pyt_1dot4b_all[filtr], n400, 10, "UCL_n400_pyth_1.4B_all")
m_pyt_2dot8b_n400_all, sd_pyt_2dot8b_n400_all = test_model_Ridge(pyt_2dot8b_all[filtr], n400, 10, "UCL_n400_pyth_2.8B_all")
m_transf_xl_n400_all, sd_transf_xl_n400_all = test_model_Ridge(transfo_xl_all[filtr], n400, 10, "UCL_n400_transfo_xl_all")
print(m_neo_n400_all,m_neo_med_n400_all,m_neo_large_n400_all,m_pyt_n400_all,m_pyt_160m_n400_all,m_pyt_410m_n400_all,m_pyt_1b_n400_all,m_pyt_1dot4b_n400_all,m_pyt_2dot8b_n400_all,m_transf_xl_n400_all)
print(sd_neo_n400_all,sd_neo_med_n400_all,sd_neo_large_n400_all,sd_pyt_n400_all,sd_pyt_160m_n400_all,sd_pyt_410m_n400_all,sd_pyt_1b_n400_all,sd_pyt_1dot4b_n400_all,sd_pyt_2dot8b_n400_all,sd_transf_xl_n400_all)

# P600
m_neo_p600_all, sd_neo_p600_all               = test_model_Ridge(neo_all[filtr], p600, 10, "UCL_p600_neo_all")
m_neo_med_p600_all, sd_neo_med_p600_all       = test_model_Ridge(neo_med_all[filtr], p600, 10, "UCL_p600_neo_med_all")
m_neo_large_p600_all, sd_neo_large_p600_all   = test_model_Ridge(neo_large_all[filtr], p600, 10, "UCL_p600_neo_large_all")
m_pyt_p600_all, sd_pyt_p600_all               = test_model_Ridge(pyt_all[filtr], p600, 10, "UCL_p600_pyth_all")
m_pyt_160m_p600_all, sd_pyt_160m_p600_all     = test_model_Ridge(pyt_160m_all[filtr], p600, 10, "UCL_p600_pyth_160M_all")
m_pyt_410m_p600_all, sd_pyt_410m_p600_all     = test_model_Ridge(pyt_410m_all[filtr], p600, 10, "UCL_p600_pyth_410M_all")
m_pyt_1b_p600_all, sd_pyt_1b_p600_all         = test_model_Ridge(pyt_1b_all[filtr], p600, 10, "UCL_p600_pyth_1B_all")
m_pyt_1dot4b_p600_all, sd_pyt_1dot4b_p600_all = test_model_Ridge(pyt_1dot4b_all[filtr], p600, 10, "UCL_p600_pyth_1.4B_all")
m_pyt_2dot8b_p600_all, sd_pyt_2dot8b_p600_all = test_model_Ridge(pyt_2dot8b_all[filtr], p600, 10, "UCL_p600_pyth_2.8B_all")
m_transf_xl_p600_all, sd_transf_xl_p600_all = test_model_Ridge(transfo_xl_all[filtr], p600, 10, "UCL_p600_transfo_xl_all")
print(m_neo_p600_all,m_neo_med_p600_all,m_neo_large_p600_all,m_pyt_p600_all,m_pyt_160m_p600_all,m_pyt_410m_p600_all,m_pyt_1b_p600_all,m_pyt_1dot4b_p600_all,m_pyt_2dot8b_p600_all,m_transf_xl_p600_all)
print(sd_neo_p600_all,sd_neo_med_p600_all,sd_neo_large_p600_all,sd_pyt_p600_all,sd_pyt_160m_p600_all,sd_pyt_410m_p600_all,sd_pyt_1b_p600_all,sd_pyt_1dot4b_p600_all,sd_pyt_2dot8b_p600_all,sd_transf_xl_p600_all)

# SPR, spill
neo_all_spill        = np.concatenate((np.array(df_spill[["s_GPTNeo", "logfreq", "length",
                                                          "s_GPTNeo-1", "logfreq-1", "length-1",
                                                          "s_GPTNeo-2", "logfreq-2", "length-2"]]), neo_spill), axis=1)
neo_med_all_spill    = np.concatenate((np.array(df_spill[["s_GPTNeo_med", "logfreq", "length",
                                                          "s_GPTNeo_med-1", "logfreq-1", "length-1",
                                                          "s_GPTNeo_med-2", "logfreq-2", "length-2"]]), neo_med_spill), axis=1)
neo_large_all_spill  = np.concatenate((np.array(df_spill[["s_GPTNeo_large", "logfreq", "length",
                                                          "s_GPTNeo_large-1", "logfreq-1", "length-1",
                                                          "s_GPTNeo_large-2", "logfreq-2", "length-2"]]), neo_large_spill), axis=1)
pyt_all_spill        = np.concatenate((np.array(df_spill[["s_Pythia", "logfreq", "length",
                                                          "s_Pythia-1", "logfreq-1", "length-1",
                                                          "s_Pythia-2", "logfreq-2", "length-2"]]), pyt_spill), axis=1)
pyt_160m_all_spill   = np.concatenate((np.array(df_spill[["s_Pythia_160M", "logfreq", "length",
                                                          "s_Pythia_160M-1", "logfreq-1", "length-1",
                                                          "s_Pythia_160M-2", "logfreq-2", "length-2"]]), pyt_160m_spill), axis=1)
pyt_410m_all_spill   = np.concatenate((np.array(df_spill[["s_Pythia_410M", "logfreq", "length",
                                                          "s_Pythia_410M-1", "logfreq-1", "length-1",
                                                          "s_Pythia_410M-2", "logfreq-2", "length-2"]]), pyt_410m_spill), axis=1)
pyt_1b_all_spill     = np.concatenate((np.array(df_spill[["s_Pythia_1B", "logfreq", "length",
                                                          "s_Pythia_1B-1", "logfreq-1", "length-1",
                                                          "s_Pythia_1B-2", "logfreq-2", "length-2"]]), pyt_1b_spill), axis=1)
pyt_1dot4b_all_spill = np.concatenate((np.array(df_spill[["s_Pythia_1.4B", "logfreq", "length",
                                                          "s_Pythia_1.4B-1", "logfreq-1", "length-1",
                                                          "s_Pythia_1.4B-2", "logfreq-2", "length-2"]]), pyt_1dot4b_spill), axis=1)
pyt_2dot8b_all_spill = np.concatenate((np.array(df_spill[["s_Pythia_2.8B", "logfreq", "length",
                                                          "s_Pythia_2.8B-1", "logfreq-1", "length-1",
                                                          "s_Pythia_2.8B-2", "logfreq-2", "length-2"]]), pyt_2dot8b_spill), axis=1)
transf_xl_all_spill  = np.concatenate((np.array(df_spill[["s_transfo_xl", "logfreq", "length",
                                                          "s_transfo_xl-1", "logfreq-1", "length-1",
                                                          "s_transfo_xl-2", "logfreq-2", "length-2"]]), transfo_xl_spill), axis=1)

# spr, all
m_neo_spr_all, sd_neo_spr_all               = test_model_Ridge(neo_all_spill, spr_spill, 10, "UCL_spr_neo_all")
m_neo_med_spr_all, sd_neo_med_spr_all       = test_model_Ridge(neo_med_all_spill, spr_spill, 10, "UCL_spr_neo_med_all")
m_neo_large_spr_all, sd_neo_large_spr_all   = test_model_Ridge(neo_large_all_spill, spr_spill, 10, "UCL_spr_neo_large_all")
m_pyt_spr_all, sd_pyt_spr_all               = test_model_Ridge(pyt_all_spill, spr_spill, 10, "UCL_spr_pyth_all")
m_pyt_160m_spr_all, sd_pyt_160m_spr_all     = test_model_Ridge(pyt_160m_all_spill, spr_spill, 10, "UCL_spr_pyth_160M_all")
m_pyt_410m_spr_all, sd_pyt_410m_spr_all     = test_model_Ridge(pyt_410m_all_spill, spr_spill, 10, "UCL_spr_pyth_410M_all")
m_pyt_1b_spr_all, sd_pyt_1b_spr_all         = test_model_Ridge(pyt_1b_all_spill, spr_spill, 10, "UCL_spr_pyth_1B_all")
m_pyt_1dot4b_spr_all, sd_pyt_1dot4b_spr_all = test_model_Ridge(pyt_1dot4b_all_spill, spr_spill, 10, "UCL_spr_pyth_1.4B_all")
m_pyt_2dot8b_spr_all, sd_pyt_2dot8b_spr_all = test_model_Ridge(pyt_2dot8b_all_spill, spr_spill, 10, "UCL_spr_pyth_2.8B_all")
m_transf_xl_spr_all, sd_transf_xl_spr_all = test_model_Ridge(transf_xl_all_spill, spr_spill, 10, "UCL_spr_transfo_xl_all")
print(m_neo_spr_all,m_neo_med_spr_all,m_neo_large_spr_all,m_pyt_spr_all,m_pyt_160m_spr_all,m_pyt_410m_spr_all,m_pyt_1b_spr_all,m_pyt_1dot4b_spr_all,m_pyt_2dot8b_spr_all,m_transf_xl_spr_all)
print(sd_neo_spr_all,sd_neo_med_spr_all,sd_neo_large_spr_all,sd_pyt_spr_all,sd_pyt_160m_spr_all,sd_pyt_410m_spr_all,sd_pyt_1b_spr_all,sd_pyt_1dot4b_spr_all,sd_pyt_2dot8b_spr_all,sd_transf_xl_spr_all)


df_results = pd.DataFrame({
    "base_fpass":[m_neo_fpass_b,m_neo_med_fpass_b,m_neo_large_fpass_b,m_pyt_fpass_b,m_pyt_160m_fpass_b,m_pyt_410m_fpass_b,m_pyt_1b_fpass_b,m_pyt_1dot4b_fpass_b,m_pyt_2dot8b_fpass_b,m_transf_xl_fpass_b],
    "base_fpass_sd": [sd_neo_fpass_b,sd_neo_med_fpass_b,sd_neo_large_fpass_b,sd_pyt_fpass_b,sd_pyt_160m_fpass_b,sd_pyt_410m_fpass_b,sd_pyt_1b_fpass_b,sd_pyt_1dot4b_fpass_b,sd_pyt_2dot8b_fpass_b,sd_transf_xl_fpass_b],
    "base_n400":  [m_neo_n400_b,m_neo_med_n400_b,m_neo_large_n400_b,m_pyt_n400_b,m_pyt_160m_n400_b,m_pyt_410m_n400_b,m_pyt_1b_n400_b,m_pyt_1dot4b_n400_b,m_pyt_2dot8b_n400_b,m_transf_xl_n400_b],
    "base_n400_sd":  [sd_neo_n400_b,sd_neo_med_n400_b,sd_neo_large_n400_b,sd_pyt_n400_b,sd_pyt_160m_n400_b,sd_pyt_410m_n400_b,sd_pyt_1b_n400_b,sd_pyt_1dot4b_n400_b,sd_pyt_2dot8b_n400_b,sd_transf_xl_n400_b],
    "base_p600":  [m_neo_p600_b,m_neo_med_p600_b,m_neo_large_p600_b,m_pyt_p600_b,m_pyt_160m_p600_b,m_pyt_410m_p600_b,m_pyt_1b_p600_b,m_pyt_1dot4b_p600_b,m_pyt_2dot8b_p600_b,m_transf_xl_p600_b],
    "base_p600_sd":  [sd_neo_p600_b,sd_neo_med_p600_b,sd_neo_large_p600_b,sd_pyt_p600_b,sd_pyt_160m_p600_b,sd_pyt_410m_p600_b,sd_pyt_1b_p600_b,sd_pyt_1dot4b_p600_b,sd_pyt_2dot8b_p600_b,sd_transf_xl_p600_b],
    "base_spr": [m_neo_spr_b,m_neo_med_spr_b,m_neo_large_spr_b,m_pyt_spr_b,m_pyt_160m_spr_b,m_pyt_410m_spr_b,m_pyt_1b_spr_b,m_pyt_1dot4b_spr_b,m_pyt_2dot8b_spr_b,m_transf_xl_spr_b],
    "base_spr_sd":[sd_neo_spr_b,sd_neo_med_spr_b,sd_neo_large_spr_b,sd_pyt_spr_b,sd_pyt_160m_spr_b,sd_pyt_410m_spr_b,sd_pyt_1b_spr_b,sd_pyt_1dot4b_spr_b,sd_pyt_2dot8b_spr_b,sd_transf_xl_spr_b],
    "act_fpass":[m_neo_fpass,m_neo_med_fpass,m_neo_large_fpass,m_pyt_fpass,m_pyt_160m_fpass,m_pyt_410m_fpass,m_pyt_1b_fpass,m_pyt_1dot4b_fpass,m_pyt_2dot8b_fpass,m_transfo_xl_fpass],
    "act_fpass_sd": [sd_neo_fpass,sd_neo_med_fpass,sd_neo_large_fpass,sd_pyt_fpass,sd_pyt_160m_fpass,sd_pyt_410m_fpass,sd_pyt_1b_fpass,sd_pyt_1dot4b_fpass,sd_pyt_2dot8b_fpass,sd_transfo_xl_fpass],
    "act_n400":  [m_neo_n400,m_neo_med_n400,m_neo_large_n400,m_pyt_n400,m_pyt_160m_n400,m_pyt_410m_n400,m_pyt_1b_n400,m_pyt_1dot4b_n400,m_pyt_2dot8b_n400,m_transfo_xl_n400],
    "act_n400_sd":  [sd_neo_n400,sd_neo_med_n400,sd_neo_large_n400,sd_pyt_n400,sd_pyt_160m_n400,sd_pyt_410m_n400,sd_pyt_1b_n400,sd_pyt_1dot4b_n400,sd_pyt_2dot8b_n400,sd_transfo_xl_n400],
    "act_p600":  [m_neo_p600,m_neo_med_p600,m_neo_large_p600,m_pyt_p600,m_pyt_160m_p600,m_pyt_410m_p600,m_pyt_1b_p600,m_pyt_1dot4b_p600,m_pyt_2dot8b_p600,m_transfo_xl_p600],
    "act_p600_sd":  [sd_neo_p600,sd_neo_med_p600,sd_neo_large_p600,sd_pyt_p600,sd_pyt_160m_p600,sd_pyt_410m_p600,sd_pyt_1b_p600,sd_pyt_1dot4b_p600,sd_pyt_2dot8b_p600,sd_transfo_xl_p600],
    "act_spr": [m_neo_spr,m_neo_med_spr,m_neo_large_spr,m_pyt_spr,m_pyt_160m_spr,m_pyt_410m_spr,m_pyt_1b_spr,m_pyt_1dot4b_spr,m_pyt_2dot8b_spr,m_transfo_xl_spr],
    "act_spr_sd":[sd_neo_spr,sd_neo_med_spr,sd_neo_large_spr,sd_pyt_spr,sd_pyt_160m_spr,sd_pyt_410m_spr,sd_pyt_1b_spr,sd_pyt_1dot4b_spr,sd_pyt_2dot8b_spr,sd_transfo_xl_spr],
    "all_fpass":[m_neo_fpass_all,m_neo_med_fpass_all,m_neo_large_fpass_all,m_pyt_fpass_all,m_pyt_160m_fpass_all,m_pyt_410m_fpass_all,m_pyt_1b_fpass_all,m_pyt_1dot4b_fpass_all,m_pyt_2dot8b_fpass_all,m_transfo_xl_fpass_all],
    "all_fpass_sd":[sd_neo_fpass_all,sd_neo_med_fpass_all,sd_neo_large_fpass_all,sd_pyt_fpass_all,sd_pyt_160m_fpass_all,sd_pyt_410m_fpass_all,sd_pyt_1b_fpass_all,sd_pyt_1dot4b_fpass_all,sd_pyt_2dot8b_fpass_all,sd_transfo_xl_fpass_all],
    "all_n400":[m_neo_n400_all,m_neo_med_n400_all,m_neo_large_n400_all,m_pyt_n400_all,m_pyt_160m_n400_all,m_pyt_410m_n400_all,m_pyt_1b_n400_all,m_pyt_1dot4b_n400_all,m_pyt_2dot8b_n400_all,m_transf_xl_n400_all],
    "all_n400_sd":[sd_neo_n400_all,sd_neo_med_n400_all,sd_neo_large_n400_all,sd_pyt_n400_all,sd_pyt_160m_n400_all,sd_pyt_410m_n400_all,sd_pyt_1b_n400_all,sd_pyt_1dot4b_n400_all,sd_pyt_2dot8b_n400_all,sd_transf_xl_n400_all],
    "all_p600":[m_neo_p600_all,m_neo_med_p600_all,m_neo_large_p600_all,m_pyt_p600_all,m_pyt_160m_p600_all,m_pyt_410m_p600_all,m_pyt_1b_p600_all,m_pyt_1dot4b_p600_all,m_pyt_2dot8b_p600_all,m_transf_xl_p600_all],
    "all_p600_sd":[sd_neo_p600_all,sd_neo_med_p600_all,sd_neo_large_p600_all,sd_pyt_p600_all,sd_pyt_160m_p600_all,sd_pyt_410m_p600_all,sd_pyt_1b_p600_all,sd_pyt_1dot4b_p600_all,sd_pyt_2dot8b_p600_all,sd_transf_xl_p600_all],
    "all_spr":[m_neo_spr_all,m_neo_med_spr_all,m_neo_large_spr_all,m_pyt_spr_all,m_pyt_160m_spr_all,m_pyt_410m_spr_all,m_pyt_1b_spr_all,m_pyt_1dot4b_spr_all,m_pyt_2dot8b_spr_all,m_transf_xl_spr_all],
    "all_spr_sd":[sd_neo_spr_all,sd_neo_med_spr_all,sd_neo_large_spr_all,sd_pyt_spr_all,sd_pyt_160m_spr_all,sd_pyt_410m_spr_all,sd_pyt_1b_spr_all,sd_pyt_1dot4b_spr_all,sd_pyt_2dot8b_spr_all,sd_transf_xl_spr_all]
    })

# df_results.to_csv("results/results_UCL_newmodels.csv", index=True)

df_results = pd.read_csv("results/results_UCL_newmodels.csv")

########################
# significance_testing #
########################


thebase_fpass = ["UCL_fpass_neo_base", "UCL_fpass_neo_med_base", "UCL_fpass_neo_large_base", "UCL_fpass_pyt_base", "UCL_fpass_pyt_160M_base", "UCL_fpass_pyt_410M_base", "UCL_fpass_pyt_1B_base", "UCL_fpass_pyt_1.4B_base", "UCL_fpass_pyt_2.8B_base","UCL_fpass_transfo_xl_base"]
theemb_fpass  = ["UCL_fpass_neo", "UCL_fpass_neo_med", "UCL_fpass_neo_large", "UCL_fpass_pyth", "UCL_fpass_pyth_160M", "UCL_fpass_pyth_410M", "UCL_fpass_pyth_1B", "UCL_fpass_pyth_1.4B", "UCL_fpass_pyth_2.8B","UCL_fpass_transfo_xl"]
all_fpass     = ["UCL_fpass_neo_all", "UCL_fpass_neo_med_all", "UCL_fpass_neo_large_all", "UCL_fpass_pyth_all", "UCL_fpass_pyth_160M_all", "UCL_fpass_pyth_410M_all", "UCL_fpass_pyth_1B_all", "UCL_fpass_pyth_1.4B_all", "UCL_fpass_pyth_2.8B_all","UCL_fpass_transfo_xl_all"]

thebase_spr = ["UCL_spr_neo_base", "UCL_spr_neo_med_base", "UCL_spr_neo_large_base", "UCL_spr_pyt_base", "UCL_spr_pyt_160M_base", "UCL_spr_pyt_410M_base", "UCL_spr_pyt_1B_base", "UCL_spr_pyt_1.4B_base", "UCL_spr_pyt_2.8B_base","UCL_spr_transfo_xl_base"]
theemb_spr  = ["UCL_spr_neo", "UCL_spr_neo_med", "UCL_spr_neo_large", "UCL_spr_pyth", "UCL_spr_pyth_160M", "UCL_spr_pyth_410M", "UCL_spr_pyth_1B", "UCL_spr_pyth_1.4B", "UCL_spr_pyth_2.8B","UCL_spr_transfo_xl"]
all_spr     = ["UCL_spr_neo_all", "UCL_spr_neo_med_all", "UCL_spr_neo_large_all", "UCL_spr_pyth_all", "UCL_spr_pyth_160M_all", "UCL_spr_pyth_410M_all", "UCL_spr_pyth_1B_all", "UCL_spr_pyth_1.4B_all", "UCL_spr_pyth_2.8B_all","UCL_spr_transfo_xl_all"]

thebase_n400 = ["UCL_n400_neo_base", "UCL_n400_neo_med_base", "UCL_n400_neo_large_base", "UCL_n400_pyt_base", "UCL_n400_pyt_160M_base", "UCL_n400_pyt_410M_base", "UCL_n400_pyt_1B_base", "UCL_n400_pyt_1.4B_base", "UCL_n400_pyt_2.8B_base","UCL_n400_transfo_xl_base"]
theemb_n400  = ["UCL_n400_neo", "UCL_n400_neo_med", "UCL_n400_neo_large", "UCL_n400_pyth", "UCL_n400_pyth_160M", "UCL_n400_pyth_410M", "UCL_n400_pyth_1B", "UCL_n400_pyth_1.4B", "UCL_n400_pyth_2.8B","UCL_n400_transfo_xl"]
all_n400     = ["UCL_n400_neo_all", "UCL_n400_neo_med_all", "UCL_n400_neo_large_all", "UCL_n400_pyth_all", "UCL_n400_pyth_160M_all", "UCL_n400_pyth_410M_all", "UCL_n400_pyth_1B_all", "UCL_n400_pyth_1.4B_all", "UCL_n400_pyth_2.8B_all","UCL_n400_transfo_xl_all"]

thebase_p600 = ["UCL_p600_neo_base", "UCL_p600_neo_med_base", "UCL_p600_neo_large_base", "UCL_p600_pyt_base", "UCL_p600_pyt_160M_base", "UCL_p600_pyt_410M_base", "UCL_p600_pyt_1B_base", "UCL_p600_pyt_1.4B_base", "UCL_p600_pyt_2.8B_base","UCL_p600_transfo_xl_base"]
theemb_p600  = ["UCL_p600_neo", "UCL_p600_neo_med", "UCL_p600_neo_large", "UCL_p600_pyth", "UCL_p600_pyth_160M", "UCL_p600_pyth_410M", "UCL_p600_pyth_1B", "UCL_p600_pyth_1.4B", "UCL_p600_pyth_2.8B","UCL_p600_transfo_xl"]
all_p600     = ["UCL_p600_neo_all", "UCL_p600_neo_med_all", "UCL_p600_neo_large_all", "UCL_p600_pyth_all", "UCL_p600_pyth_160M_all", "UCL_p600_pyth_410M_all", "UCL_p600_pyth_1B_all", "UCL_p600_pyth_1.4B_all", "UCL_p600_pyth_2.8B_all","UCL_p600_transfo_xl_all"]

p_fpass = []
for base, emb, all_ in zip(thebase_fpass, theemb_fpass, all_fpass):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_fpass.append([p_base_emb, p_all_base, p_all_emb])
p_fpass = pd.DataFrame(p_fpass, columns=["base_emb", "all_base", "all_emb"])

p_spr = []
for base, emb, all_ in zip(thebase_spr, theemb_spr, all_spr):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_spr.append([p_base_emb, p_all_base, p_all_emb])
p_spr = pd.DataFrame(p_spr, columns=["base_emb", "all_base", "all_emb"])

p_n400 = []
for base, emb, all_ in zip(thebase_n400, theemb_n400, all_n400):
    p_emb_base = significance_testing(emb, base)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_n400.append([p_base_emb, p_all_base, p_all_emb])
p_n400 = pd.DataFrame(p_n400, columns=["base_emb", "all_base", "all_emb"])

p_p600 = []
for base, emb, all_ in zip(thebase_p600, theemb_p600, all_p600):
    p_emb_base = significance_testing(emb, base)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_p600.append([p_base_emb, p_all_base, p_all_emb])
p_p600 = pd.DataFrame(p_p600, columns=["base_emb", "all_base", "all_emb"])

for df, dfname in zip([p_fpass, p_spr, p_n400, p_p600], ["fpass", "spr", "n400", "p600"]):
    for df_column in df.columns:
        df_results["p_"+dfname+"_"+df_column] = list(df[df_column])

# df_results.to_csv("results/results_UCL_newmodels.csv", index=True)
df_results = pd.read_csv("results/results_UCL_newmodels.csv")
df_results_old = pd.read_csv("results/results_UCL.csv")

############
# plotting #
############
from math import sqrt, e

col1 = "act_p600"
col2 = "base_p600"
col3 = "all_p600"

ylim1 = 0
ylim2= .85
title = "UCL$_{P600}$"

heights = [item for triplet in [[df_results.loc[idx, col1], df_results.loc[idx, col2], df_results.loc[idx, col3]] for idx in range(10)] for item in triplet]
heights_old = [item for triplet in [[df_results_old.loc[idx, col1], df_results_old.loc[idx, col2], df_results_old.loc[idx, col3]] for idx in range(5)] for item in triplet]
heights = heights + heights_old
yerr=[item for triplet in [[df_results.loc[idx, col1+"_sd"], df_results.loc[idx, col2+"_sd"], df_results.loc[idx, col3+"_sd"]] for idx in range(10)] for item in triplet]
yerr_old = [item for triplet in [[df_results_old.loc[idx, col1+"_sd"], df_results_old.loc[idx, col2+"_sd"], df_results_old.loc[idx, col3+"_sd"]] for idx in range(5)] for item in triplet]
yerr = yerr + yerr_old
yerr = [y/sqrt(10) for y in yerr]

bars = [1,2,3,5,6,7,9,10,11,14,15,16,18,19,20,22,23,24,26,27,28,30,31,32,34,35,36,39,40,41, 44,45,46,48,49,50,52,53,54,56,57,58,60,61,62]

plt.rcParams.update({
    "text.usetex": False})
fig = plt.figure(figsize = (15/1.7, 3.5/1.7), dpi=300)
plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
plt.minorticks_on()
plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
plt.bar(bars, 
        heights,capsize=2, color = ['lightblue', 'lightgreen', "indianred"]*5,width=0.9,
        tick_label=["", "GPT-Neo$_{125M}$", "", "", "GPT-Neo$_{1.3B}$", "", "", "GPT-Neo$_{2.7B}$","", "",  "Pythia$_{70M}$", "", "", "Pythia$_{160M}$", "",  "", "Pythia$_{410M}$", "",  "", "Pythia$_{1B}$", "",  "", "Pythia$_{1.4B}$", "",  "", "Pythia$_{2.8B}$", "",  "", "TransformerXL", "", "", "GPT", "", "", "GPT-2$_{124M}$", "", "", "GPT-2$_{355M}$","", "",  "GPT-2$_{774M}$", "", "", "GPT-2$_{1.5B}$", ""], zorder=5, yerr=yerr
        )
plt.axvline(x=12.5, color='black', linestyle='dashed')
plt.axvline(x=37.5, color='black', linestyle='dashed')
plt.axvline(x=42.5, color='black', linestyle='dashed')

plt.ylabel("r")
plt.xticks(fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
plt.ylim(ylim1, ylim2)
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
plt.title(title, fontsize=16)
#plt.tight_layout()
plt.show()



m_neo_fpass_b,m_neo_med_fpass_b,m_neo_large_fpass_b,m_pyt_fpass_b,m_pyt_160m_fpass_b,m_pyt_410m_fpass_b,m_pyt_1b_fpass_b,m_pyt_1dot4b_fpass_b,m_pyt_2dot8b_fpass_b,m_transf_xl_fpass_b


col = "all_p600"
df = pd.DataFrame({
"params" : [125, 1300, 2700, 70, 160, 410, 1000, 1400, 2800, 257, 117, 124, 355, 774, 1500],
"typ" :  ["neo", "neo", "neo", "pt", "pt", "pt", "pt", "pt", "pt", "xl", "gpt", "gpt", "gpt", "gpt", "gpt"],
"res" : df_results[col].tolist() + df_results_old[col].tolist()})

fig = plt.figure(figsize = (5, 3.5), dpi=300)
colors = ['blue', 'green', "red", "orange"]
markers = ['o', 's', '^', 'x']
labels = ["GPT-Neo", "Pythia", "TransformerXL", "GPT"]
plt.plot(df[df["typ"] == "neo"]["params"], df[df["typ"] == "neo"]["res"], color=colors[0], marker=markers[0], label=labels[0])
plt.plot(df[df["typ"] == "pt"]["params"], df[df["typ"] == "pt"]["res"], color=colors[1], marker=markers[1], label=labels[1])
plt.plot(df[df["typ"] == "xl"]["params"], df[df["typ"] == "xl"]["res"], color=colors[2], marker=markers[2], label=labels[2])
plt.plot(df[df["typ"] == "gpt"]["params"], df[df["typ"] == "gpt"]["res"], color=colors[3], marker=markers[3], label=labels[3])
#plt.ylim(.7885, .792)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=4)
plt.grid(True)
plt.xlabel('N° parameters (M)')
plt.ylabel('R')
plt.title('UCL$_{P600}$')
plt.show()

