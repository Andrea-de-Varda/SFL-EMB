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

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT)
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, plot_sem, get_sem, plot_weights_sem, decompose_sfl, probe

######################
# UCL READING CORPUS #
######################

eye = pd.read_csv(PROJECT_ROOT / "main_data/UCL_ET_CORPUS/eyetracking.RT.txt", sep="\t")
sentences = pd.read_csv(PROJECT_ROOT / "main_data/UCL_ET_CORPUS/stimuli.txt", sep="\t", encoding='cp1252')
eye = eye[eye["RTfirstpass"]>0].groupby(["sent_nr", "word_pos"], as_index=False, sort=False).agg({'RTfirstfix':"mean", 'RTfirstpass':"mean", 'RTrightbound':"mean", 'RTgopast':"mean", "word":"max"})
sent_map = {row.sent_nr:row.sentence.strip() for index, row in sentences.iterrows()}

with open("sentences.txt", 'w') as fp:
    for item in list(sentences.sentence):
        fp.write("%s\n" % item)
    print('Done')
    
spr =  pd.read_csv(PROJECT_ROOT / "main_data/UCL_ET_CORPUS/selfpacedreading.RT.txt", sep="\t")
spr = spr.groupby(["sent_nr", "word_pos"], as_index=False, sort=False).agg({'RT':"mean", "word":"max"})

df_eye_spr = pd.merge(eye, spr, on=["sent_nr", "word_pos", "word"], how="left")
df_eye_spr["sentence"] = df_eye_spr["sent_nr"].map(sent_map)

############
# ERP DATA #
############

mat = scipy.io.loadmat(PROJECT_ROOT / "main_data/ERP_Frank/stimuli_erp.mat")
mat.keys()

# Frank has probabilities from 10 increasingly large corpora in RNN
# N-gram size: 2, 3, 4

out = []
for idx, sentence in enumerate(mat["sentences"]):
    tokens = sentence[0][0]
    sentence = [t[0] for t in tokens]
    ERP = mat["ERP"][idx][0]
    #dH = mat["dH_pos"][idx][0] # 10 rnn models, entropy reduction over k (to 4) upcoming words
    #dH_POS = mat["dH"][idx][0]
    s_rnn = mat["surp_rnn"][idx][0] # 10 models
    s_rnn_pos = mat["surp_pos_rnn"][idx][0]
    s_psg = mat["surp_psg"][idx][0] # 9 models
    s_psg_pos = mat["surp_pos_psg"][idx][0]
    s_ngram = mat["surp_ngram"][idx][0] # 3 models (*n*-grams)
    s_ngram_pos = mat["surp_pos_ngram"][idx][0]
    artefact = mat["artefact"][idx][0]
    freq = mat["logwordfreq"][idx][0]
    length = mat["wordlength"][idx][0][0]
    ERP[np.where(artefact == 1)] = np.nan # filter artefacts
    print(ERP.shape[0] == len(sentence))
    # for x in [dH, dH_POS, s_rnn, s_rnn_pos, s_psg, s_psg_pos, s_ngram, s_ngram_pos]:
    #     print(x.shape)
    for index in range(len(sentence)):
        word = sentence[index]
        erp = ERP[index]
        ELAN, LAN, N400, EPNP, P600, PNP = np.nanmean(erp, axis=0)
        #rnn1, rnn2, rnn3, rnn4, rnn5, rnn6, rnn7, rnn8, rnn9, rnn10 = s_rnn[index]
        #rnn1_pos, rnn2_pos, rnn3_pos, rnn4_pos, rnn5_pos, rnn6_pos, rnn7_pos, rnn8_pos, rnn9_pos, rnn10_pos = s_rnn_pos[index]
        rnn = s_rnn[index][9]
        rnn_pos = s_rnn_pos[index][9]
        psg = s_psg[index][8]
        psg_pos = s_psg_pos[index][8]
        bigram, trigram, tetragram = s_ngram[index]
        item = sentence[:index]
        frequency = freq[index][0]
        thelength = length[index]
        out.append([" ".join(sentence).strip(), index+1, " ".join(item), word, ELAN, LAN, N400, EPNP, P600, PNP, rnn, psg, bigram, trigram, tetragram, frequency, thelength])
        
frank = pd.DataFrame(out, columns = ["sentence", "word_pos", "item", "word", "ELAN", "LAN", "N400", "EPNP", "P600", "PNP", "rnn", "psg", "bigram", "trigram", "tetragram", "logfreq", "length"])
corr_pair(frank[["N400", "rnn", "psg", "bigram", "trigram", "tetragram", "logfreq", "length"]])

# merge dataframes
df_all = pd.merge(df_eye_spr, frank, on=["sentence", "word", "word_pos"], how="left")

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

# GPT
model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("openai-gpt")
surp_df("s_GPT")

# GPT2 SMALL (124M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2")
surp_df("s_GPT2_small")

# GPT2 medium
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-medium")
surp_df("s_GPT2_med")

# GPT2 large
model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-large")
surp_df("s_GPT2_large")

# GPT2 xl
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-xl")
surp_df("s_GPT2_xl")

# Transformer-XL
model = AutoModelForCausalLM.from_pretrained("transfo-xl-wt103", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("transfo-xl-wt103")
surp_df("s_transfo_xl")


df_all = df_all[['sent_nr', 'word_pos', 'word', 'sentence', 'item', 'RTfirstfix', 'RTfirstpass', 'RTrightbound','RTgopast', 'RT', 'ELAN', 'LAN', 'N400', 'EPNP', 'P600', 'PNP', 'rnn', 'psg', 'bigram', 'trigram', 'tetragram','logfreq', 'length', 's_GPT', 's_GPT2_small', 's_GPT2_med','s_GPT2_large', 's_GPT2_xl', 's_transfo_xl']]

# df_all.to_csv("UCL_reading_all.csv", index=False)

################################################################################################

df_all = pd.read_csv("UCL_reading_all.csv")

########################################################################################
# (changed later) S and F should have the same base of the log (nat) for comparability #
########################################################################################

# frequency
freq = pd.read_excel(PROJECT_ROOT / "main_data/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}
minfreq = min(freq["Lg10WF"])

def get_f(word):
    word = re.sub('[\.\,\:\-\?\!\)\(\"]', "", word)
    try:
        fr = f[word]
    except KeyError:
        fr = minfreq
    return fr

df_all["logfreq"] = [get_f(row.word) for index, row in df_all.iterrows()]

df_noinit = df_all.dropna(subset=["s_GPT2_small"]) # remove sentence-starting

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

# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "corpora/sentences.txt",
        dirName+'/activations_gpt.json',
        aggregation="average"
    )

################
# GPT-2 FAMILY #
################

# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "corpora/sentences.txt",
        dirName+'/activations_gpt2_small.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "corpora/sentences.txt",
        dirName+'/activations_gpt2_medium.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "corpora/sentences.txt",
        dirName+'/activations_gpt2_large.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "corpora/sentences.txt",
        dirName+'/activations_gpt2_xl.json',
        aggregation="average"
    )

act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl.json', 1600, is_brnn=False)

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

gpt_filt   = get_act_eye(act_gpt, 12, 768)
small_filt = get_act_eye(act_gpt2_small, 12, 768)
med_filt   = get_act_eye(act_gpt2_medium, 24, 1024)
large_filt = get_act_eye(act_gpt2_large, 36, 1280)
xl_filt    = get_act_eye(act_gpt2_xl, 48, 1600)

gpt_filt_f   = gpt_filt[filtr]
small_filt_f = small_filt[filtr]
med_filt_f   = med_filt[filtr]
large_filt_f = large_filt[filtr]
xl_filt_f    = xl_filt[filtr]

fp   = np.array(df_noinit["RTfirstpass"])
ff   = np.array(df_noinit["RTfirstfix"])
tf   = np.array(df_noinit["RTrightbound"])
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
            
gpt_filt_spill   = spill_embeddings(gpt_filt, to_filt)
small_filt_spill = spill_embeddings(small_filt, to_filt)
med_filt_spill   = spill_embeddings(med_filt, to_filt)
large_filt_spill = spill_embeddings(large_filt, to_filt)
xl_filt_spill    = spill_embeddings(xl_filt, to_filt)

####################
# Ridge regression #
####################

from sklearn.model_selection import KFold

####################
# ACTIVATIONS ONLY #
####################

# first-pass, activation only
m_gpt_fpass, sd_gpt_fpass     = test_model_Ridge(gpt_filt, fp, 10, "UCL_fpass_gpt")
m_small_fpass, sd_small_fpass = test_model_Ridge(small_filt, fp, 10, "UCL_fpass_small")
m_med_fpass, sd_med_fpass     = test_model_Ridge(med_filt, fp, 10, "UCL_fpass_med")
m_larg_fpass, sd_larg_fpass   = test_model_Ridge(large_filt, fp, 10, "UCL_fpass_large")
m_xl_fpass, sd_xl_fpass       = test_model_Ridge(xl_filt, fp, 10, "UCL_fpass_xl")
print(m_gpt_fpass, m_small_fpass, m_med_fpass, m_larg_fpass, m_xl_fpass)

# n400, activation only
m_gpt_n400, sd_gpt_n400     = test_model_Ridge(gpt_filt_f, n400, 10, "UCL_n400_gpt")
m_small_n400, sd_small_n400 = test_model_Ridge(small_filt_f, n400, 10, "UCL_n400_small")
m_med_n400, sd_med_n400     = test_model_Ridge(med_filt_f, n400, 10, "UCL_n400_med")
m_larg_n400, sd_larg_n400   = test_model_Ridge(large_filt_f, n400, 10, "UCL_n400_large")
m_xl_n400, sd_xl_n400       = test_model_Ridge(xl_filt_f, n400, 10, "UCL_n400_xl")
print(m_gpt_n400, m_small_n400, m_med_n400, m_larg_n400, m_xl_n400)

# p600, activations only
m_gpt_p600, sd_gpt_p600     = test_model_Ridge(gpt_filt_f, p600, 10, "UCL_p600_gpt")
m_small_p600, sd_small_p600 = test_model_Ridge(small_filt_f, p600, 10, "UCL_p600_small")
m_med_p600, sd_med_p600     = test_model_Ridge(med_filt_f, p600, 10, "UCL_p600_med")
m_larg_p600, sd_larg_p600   = test_model_Ridge(large_filt_f, p600, 10, "UCL_p600_large")
m_xl_p600, sd_xl_p600       = test_model_Ridge(xl_filt_f, p600, 10, "UCL_p600_xl")
print(m_gpt_p600, m_small_p600, m_med_p600, m_larg_p600, m_xl_p600)

# SPR, spill
m_gpt_spr, sd_gpt_spr     = test_model_Ridge(gpt_filt_spill, spr_spill, 10, "UCL_spr_gpt")
m_small_spr, sd_small_spr = test_model_Ridge(small_filt_spill, spr_spill, 10, "UCL_spr_small")
m_med_spr, sd_med_spr     = test_model_Ridge(med_filt_spill, spr_spill, 10, "UCL_spr_med")
m_larg_spr, sd_larg_spr   = test_model_Ridge(large_filt_spill, spr_spill, 10, "UCL_spr_large")
m_xl_spr, sd_xl_spr       = test_model_Ridge(xl_filt_spill, spr_spill, 10, "UCL_spr_xl")
print(m_gpt_spr, m_small_spr, m_med_spr, m_larg_spr, m_xl_spr)

#################
# BASELINE ONLY #
#################

# first-pass, baseline only
m_gpt_fpass_b, sd_gpt_fpass_b     = test_model_Ridge(np.array(df_noinit[["s_GPT", "logfreq", "length"]]), fp, 10, "UCL_fpass_gpt_base")
m_small_fpass_b, sd_small_fpass_b = test_model_Ridge(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]]), fp, 10, "UCL_fpass_small_base")
m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), fp, 10, "UCL_fpass_med_base")
m_larg_fpass_b, sd_larg_fpass_b   = test_model_Ridge(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]]), fp, 10, "UCL_fpass_large_base")
m_xl_fpass_b, sd_xl_fpass_b       = test_model_Ridge(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]]), fp, 10, "UCL_fpass_xl_base")
print(m_gpt_fpass_b, m_small_fpass_b, m_med_fpass_b, m_larg_fpass_b, m_xl_fpass_b)

# N400, baseline only
m_gpt_n400_b, sd_gpt_n400_b     = test_model_Ridge(np.array(df_noinit[["s_GPT", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_gpt_base")
m_small_n400_b, sd_small_n400_b = test_model_Ridge(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_small_base")
m_med_n400_b, sd_med_n400_b     = test_model_Ridge(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_med_base")
m_larg_n400_b, sd_larg_n400_b   = test_model_Ridge(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_large_base")
m_xl_n400_b, sd_xl_n400_b       = test_model_Ridge(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]])[filtr], n400, 10, "UCL_n400_xl_base")
print(m_gpt_n400_b, m_small_n400_b, m_med_n400_b, m_larg_n400_b, m_xl_n400_b)

# p600, baseline only
m_gpt_p600_b, sd_gpt_p600_b     = test_model_Ridge(np.array(df_noinit[["s_GPT", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_gpt_base")
m_small_p600_b, sd_small_p600_b = test_model_Ridge(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_small_base")
m_med_p600_b, sd_med_p600_b     = test_model_Ridge(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_med_base")
m_larg_p600_b, sd_larg_p600_b   = test_model_Ridge(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_large_base")
m_xl_p600_b, sd_xl_p600_b       = test_model_Ridge(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]])[filtr], p600, 10, "UCL_p600_xl_base")
print(m_gpt_p600_b, m_small_p600_b, m_med_p600_b, m_larg_p600_b, m_xl_p600_b)

# spr_spill, baseline only
m_gpt_spr_b, sd_gpt_spr_b     = test_model_Ridge(np.array(df_spill[["s_GPT", "logfreq", "length",
                                                              "s_GPT-1", "logfreq-1", "length-1",
                                                              "s_GPT-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_gpt_base")
m_small_spr_b, sd_small_spr_b = test_model_Ridge(np.array(df_spill[["s_GPT2_small", "logfreq", "length",
                                                              "s_GPT2_small-1", "logfreq-1", "length-1",
                                                              "s_GPT2_small-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_small_base")
m_med_spr_b, sd_med_spr_b     = test_model_Ridge(np.array(df_spill[["s_GPT2_med", "logfreq", "length",
                                                              "s_GPT2_med-1", "logfreq-1", "length-1",
                                                              "s_GPT2_med-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_med_base")
m_larg_spr_b, sd_larg_spr_b   = test_model_Ridge(np.array(df_spill[["s_GPT2_large", "logfreq", "length",
                                                              "s_GPT2_large-1", "logfreq-1", "length-1",
                                                              "s_GPT2_large-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_large_base")
m_xl_spr_b, sd_xl_spr_b       = test_model_Ridge(np.array(df_spill[["s_GPT2_xl", "logfreq", "length",
                                                              "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                              "s_GPT2_xl-2", "logfreq-2", "length-2"]]), spr_spill, 10, "UCL_spr_xl_base")
print(m_gpt_spr_b, m_small_spr_b, m_med_spr_b, m_larg_spr_b, m_xl_spr_b)


# decompose SFL
decompose_sfl(np.array(df_noinit[["s_GPT", "logfreq", "length"]]), fp, "UCL_fpass_gpt_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]]), fp, "UCL_fpass_small_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), fp, "UCL_fpass_med_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]]), fp, "UCL_fpass_large_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]]), fp, "UCL_fpass_xl_base")

decompose_sfl(np.array(df_noinit[["s_GPT", "logfreq", "length"]])[filtr], n400, "UCL_n400_gpt_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]])[filtr], n400, "UCL_n400_small_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]])[filtr], n400, "UCL_n400_med_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]])[filtr], n400, "UCL_n400_large_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]])[filtr], n400, "UCL_n400_xl_base")

decompose_sfl(np.array(df_noinit[["s_GPT", "logfreq", "length"]])[filtr], p600, "UCL_p600_gpt_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]])[filtr], p600, "UCL_p600_small_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]])[filtr], p600, "UCL_p600_med_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]])[filtr], p600, "UCL_p600_large_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]])[filtr], p600, "UCL_p600_xl_base")

decompose_sfl(np.array(df_spill[["s_GPT", "logfreq", "length",
                                  "s_GPT-1", "logfreq-1", "length-1",
                                  "s_GPT-2", "logfreq-2", "length-2"]]), spr_spill, "UCL_spr_gpt_base")
decompose_sfl(np.array(df_spill[["s_GPT2_small", "logfreq", "length",
                                  "s_GPT2_small-1", "logfreq-1", "length-1",
                                  "s_GPT2_small-2", "logfreq-2", "length-2"]]), spr_spill, "UCL_spr_small_base")
decompose_sfl(np.array(df_spill[["s_GPT2_med", "logfreq", "length",
                                  "s_GPT2_med-1", "logfreq-1", "length-1",
                                  "s_GPT2_med-2", "logfreq-2", "length-2"]]), spr_spill, "UCL_spr_med_base")
decompose_sfl(np.array(df_spill[["s_GPT2_large", "logfreq", "length",
                                  "s_GPT2_large-1", "logfreq-1", "length-1",
                                  "s_GPT2_large-2", "logfreq-2", "length-2"]]), spr_spill, "UCL_spr_large_base")
decompose_sfl(np.array(df_spill[["s_GPT2_xl", "logfreq", "length",
                                  "s_GPT2_xl-1", "logfreq-1", "length-1",
                                  "s_GPT2_xl-2", "logfreq-2", "length-2"]]), spr_spill, "UCL_spr_xl_base")

##########################
# BASELINE + ACTIVATIONS #
##########################

gpt_all        = np.concatenate((np.array(df_noinit[["s_GPT", "logfreq", "length"]]), gpt_filt), axis=1)
gpt2_small_all = np.concatenate((np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]]), small_filt), axis=1)
gpt2_med_all   = np.concatenate((np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), med_filt), axis=1)
gpt2_large_all = np.concatenate((np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]]), large_filt), axis=1)
gpt2_xl_all    = np.concatenate((np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]]), xl_filt), axis=1)

# first-pass
m_gpt_fpass_all, sd_gpt_fpass_all     = test_model_Ridge(gpt_all, fp, 10, "UCL_fpass_gpt_all")
m_small_fpass_all, sd_small_fpass_all = test_model_Ridge(gpt2_small_all, fp, 10, "UCL_fpass_small_all")
m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(gpt2_med_all, fp, 10, "UCL_fpass_med_all")
m_larg_fpass_all, sd_larg_fpass_all   = test_model_Ridge(gpt2_large_all, fp, 10, "UCL_fpass_large_all")
m_xl_fpass_all, sd_xl_fpass_all       = test_model_Ridge(gpt2_xl_all, fp, 10, "UCL_fpass_xl_all")
print(m_gpt_fpass_all, m_small_fpass_all, m_med_fpass_all, m_larg_fpass_all, m_xl_fpass_all)

# n400
m_gpt_n400_all, sd_gpt_n400_all     = test_model_Ridge(gpt_all[filtr], n400, 10, "UCL_n400_gpt_all")
m_small_n400_all, sd_small_n400_all = test_model_Ridge(gpt2_small_all[filtr], n400, 10, "UCL_n400_small_all")
m_med_n400_all, sd_med_n400_all     = test_model_Ridge(gpt2_med_all[filtr], n400, 10, "UCL_n400_med_all")
m_larg_n400_all, sd_larg_n400_all   = test_model_Ridge(gpt2_large_all[filtr], n400, 10, "UCL_n400_large_all")
m_xl_n400_all, sd_xl_n400_all       = test_model_Ridge(gpt2_xl_all[filtr], n400, 10, "UCL_n400_xl_all")
print(m_gpt_n400_all, m_small_n400_all, m_med_n400_all, m_larg_n400_all, m_xl_n400_all)

# P600
m_gpt_p600_all, sd_gpt_p600_all     = test_model_Ridge(gpt_all[filtr], p600, 10, "UCL_p600_gpt_all")
m_small_p600_all, sd_small_p600_all = test_model_Ridge(gpt2_small_all[filtr], p600, 10, "UCL_p600_small_all")
m_med_p600_all, sd_med_p600_all     = test_model_Ridge(gpt2_med_all[filtr], p600, 10, "UCL_p600_med_all")
m_larg_p600_all, sd_larg_p600_all   = test_model_Ridge(gpt2_large_all[filtr], p600, 10, "UCL_p600_large_all")
m_xl_p600_all, sd_xl_p600_all       = test_model_Ridge(gpt2_xl_all[filtr], p600, 10, "UCL_p600_xl_all")
print(m_gpt_p600_all, m_small_p600_all, m_med_p600_all, m_larg_p600_all, m_xl_p600_all)

# SPR, spill
gpt_all_spill        = np.concatenate((np.array(df_spill[["s_GPT", "logfreq", "length",
                                                              "s_GPT-1", "logfreq-1", "length-1",
                                                              "s_GPT-2", "logfreq-2", "length-2"]]), gpt_filt_spill), axis=1)
gpt2_small_all_spill = np.concatenate((np.array(df_spill[["s_GPT2_small", "logfreq", "length",
                                                              "s_GPT2_small-1", "logfreq-1", "length-1",
                                                              "s_GPT2_small-2", "logfreq-2", "length-2"]]), small_filt_spill), axis=1)
gpt2_med_all_spill   = np.concatenate((np.array(df_spill[["s_GPT2_med", "logfreq", "length",
                                                              "s_GPT2_med-1", "logfreq-1", "length-1",
                                                              "s_GPT2_med-2", "logfreq-2", "length-2"]]), med_filt_spill), axis=1)
gpt2_large_all_spill = np.concatenate((np.array(df_spill[["s_GPT2_large", "logfreq", "length",
                                                              "s_GPT2_large-1", "logfreq-1", "length-1",
                                                              "s_GPT2_large-2", "logfreq-2", "length-2"]]), large_filt_spill), axis=1)
gpt2_xl_all_spill    = np.concatenate((np.array(df_spill[["s_GPT2_xl", "logfreq", "length",
                                                              "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                              "s_GPT2_xl-2", "logfreq-2", "length-2"]]), xl_filt_spill), axis=1)

m_gpt_spr_all, sd_gpt_spr_all     = test_model_Ridge(gpt_all_spill, spr_spill, 10, "UCL_spr_gpt_all")
m_small_spr_all, sd_small_spr_all = test_model_Ridge(gpt2_small_all_spill, spr_spill, 10, "UCL_spr_small_all")
m_med_spr_all, sd_med_spr_all     = test_model_Ridge(gpt2_med_all_spill, spr_spill, 10, "UCL_spr_med_all")
m_larg_spr_all, sd_larg_spr_all   = test_model_Ridge(gpt2_large_all_spill, spr_spill, 10, "UCL_spr_large_all")
m_xl_spr_all, sd_xl_spr_all       = test_model_Ridge(gpt2_xl_all_spill, spr_spill, 10, "UCL_spr_xl_all")
print(m_gpt_spr_all, m_small_spr_all, m_med_spr_all, m_larg_spr_all, m_xl_spr_all)


df_results = pd.DataFrame({
    "base_fpass":[m_gpt_fpass_b, m_small_fpass_b, m_med_fpass_b, m_larg_fpass_b, m_xl_fpass_b],
    "base_fpass_sd": [sd_gpt_fpass_b, sd_small_fpass_b, sd_med_fpass_b, sd_larg_fpass_b, sd_xl_fpass_b],
    "base_n400":  [m_gpt_n400_b, m_small_n400_b, m_med_n400_b, m_larg_n400_b, m_xl_n400_b],
    "base_n400_sd":  [sd_gpt_n400_b, sd_small_n400_b, sd_med_n400_b, sd_larg_n400_b, sd_xl_n400_b],
    "base_p600":  [m_gpt_p600_b, m_small_p600_b, m_med_p600_b, m_larg_p600_b, m_xl_p600_b],
    "base_p600_sd":  [sd_gpt_p600_b, sd_small_p600_b, sd_med_p600_b, sd_larg_p600_b, sd_xl_p600_b],
    "base_spr": [m_gpt_spr_b, m_small_spr_b, m_med_spr_b, m_larg_spr_b, m_xl_spr_b],
    "base_spr_sd":[sd_gpt_spr_b, sd_small_spr_b, sd_med_spr_b, sd_larg_spr_b, sd_xl_spr_b],
    "act_fpass":[m_gpt_fpass, m_small_fpass, m_med_fpass, m_larg_fpass, m_xl_fpass],
    "act_fpass_sd": [sd_gpt_fpass, sd_small_fpass, sd_med_fpass, sd_larg_fpass, sd_xl_fpass],
    "act_n400":  [m_gpt_n400, m_small_n400, m_med_n400, m_larg_n400, m_xl_n400],
    "act_n400_sd":  [sd_gpt_n400, sd_small_n400, sd_med_n400, sd_larg_n400, sd_xl_n400],
    "act_p600":  [m_gpt_p600, m_small_p600, m_med_p600, m_larg_p600, m_xl_p600],
    "act_p600_sd":  [sd_gpt_p600, sd_small_p600, sd_med_p600, sd_larg_p600, sd_xl_p600],
    "act_spr": [m_gpt_spr, m_small_spr, m_med_spr, m_larg_spr, m_xl_spr],
    "act_spr_sd":[sd_gpt_spr, sd_small_spr, sd_med_spr, sd_larg_spr, sd_xl_spr],
    "all_fpass":[m_gpt_fpass_all, m_small_fpass_all, m_med_fpass_all, m_larg_fpass_all, m_xl_fpass_all],
    "all_fpass_sd":[sd_gpt_fpass_all, sd_small_fpass_all, sd_med_fpass_all, sd_larg_fpass_all, sd_xl_fpass_all],
    "all_n400":[m_gpt_n400_all, m_small_n400_all, m_med_n400_all, m_larg_n400_all, m_xl_n400_all],
    "all_n400_sd":[sd_gpt_n400_all, sd_small_n400_all, sd_med_n400_all, sd_larg_n400_all, sd_xl_n400_all],
    "all_p600":[m_gpt_p600_all, m_small_p600_all, m_med_p600_all, m_larg_p600_all, m_xl_p600_all],
    "all_p600_sd":[sd_gpt_p600_all, sd_small_p600_all, sd_med_p600_all, sd_larg_p600_all, sd_xl_p600_all],
    "all_spr":[m_gpt_spr_all, m_small_spr_all, m_med_spr_all, m_larg_spr_all, m_xl_spr_all],
    "all_spr_sd":[sd_gpt_spr_all, sd_small_spr_all, sd_med_spr_all, sd_larg_spr_all, sd_xl_spr_all]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
# df_results.to_csv("results/results_UCL.csv", index=True)

df_results = pd.read_csv("results/results_UCL.csv")
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

########################
# significance_testing #
########################
thebase_fpass = ["UCL_fpass_gpt_base", "UCL_fpass_small_base", "UCL_fpass_med_base", "UCL_fpass_large_base", "UCL_fpass_xl_base"]
theemb_fpass  = ["UCL_fpass_gpt", "UCL_fpass_small", "UCL_fpass_med", "UCL_fpass_large", "UCL_fpass_xl"]
all_fpass     = ["UCL_fpass_gpt_all", "UCL_fpass_small_all", "UCL_fpass_med_all", "UCL_fpass_large_all", "UCL_fpass_xl_all"]

thebase_n400 = ["UCL_n400_gpt_base", "UCL_n400_small_base", "UCL_n400_med_base", "UCL_n400_large_base", "UCL_n400_xl_base"]
theemb_n400  = ["UCL_n400_gpt", "UCL_n400_small", "UCL_n400_med", "UCL_n400_large", "UCL_n400_xl"]
all_n400     = ["UCL_n400_gpt_all", "UCL_n400_small_all", "UCL_n400_med_all", "UCL_n400_large_all", "UCL_n400_xl_all"]

thebase_p600 = ["UCL_p600_gpt_base", "UCL_p600_small_base", "UCL_p600_med_base", "UCL_p600_large_base", "UCL_p600_xl_base"]
theemb_p600  = ["UCL_p600_gpt", "UCL_p600_small", "UCL_p600_med", "UCL_p600_large", "UCL_p600_xl"]
all_p600     = ["UCL_p600_gpt_all", "UCL_p600_small_all", "UCL_p600_med_all", "UCL_p600_large_all", "UCL_p600_xl_all"]

thebase_spr = ["UCL_spr_gpt_base", "UCL_spr_small_base", "UCL_spr_med_base", "UCL_spr_large_base", "UCL_spr_xl_base"]
theemb_spr  = ["UCL_spr_gpt", "UCL_spr_small", "UCL_spr_med", "UCL_spr_large", "UCL_spr_xl"]
all_spr     = ["UCL_spr_gpt_all", "UCL_spr_small_all", "UCL_spr_med_all", "UCL_spr_large_all", "UCL_spr_xl_all"]

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

############
# plotting #
############

from utils import base_plot

base_plot(df_results, "act_n400", "base_n400", "all_n400", title="UCL, N400")
base_plot(df_results, "act_p600", "base_p600", "all_p600", title="UCL, P600")
base_plot(df_results, "act_fpass", "base_fpass", "all_fpass", title="UCL, FPass")
base_plot(df_results, "act_spr", "base_spr", "all_spr", title="UCL, SPR")

################
# plot WEIGHTS #
################

from utils import test_model_Ridge, plot_weights_nospill, plot_weights_nospill_withEMB, plot_weights_spill, plot_weights_spill_withEMB

plot_weights_nospill("UCL_fpass_gpt_base", "UCL, FPass", "GPT", ymin = -5.5, ymax = 25)
plot_weights_nospill("UCL_fpass_small_base", "UCL, FPass", "GPT-2$_{124M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill("UCL_fpass_med_base", "UCL, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill("UCL_fpass_large_base", "UCL, FPass", "GPT-2$_{774M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill("UCL_fpass_xl_base", "UCL, FPass", "GPT-2$_{1.5B}$", ymin = -5.5, ymax = 25)

plot_weights_nospill_withEMB("UCL_fpass_gpt_all", "UCL, FPass", "GPT", ymin = -5.5, ymax = 25)
plot_weights_nospill_withEMB("UCL_fpass_small_all", "UCL, FPass", "GPT-2$_{124M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill_withEMB("UCL_fpass_med_all", "UCL, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill_withEMB("UCL_fpass_large_all", "UCL, FPass", "GPT-2$_{774M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill_withEMB("UCL_fpass_xl_all", "UCL, FPass", "GPT-2$_{774M}$",ymin = -5.5, ymax = 25)

plot_weights_spill("UCL_spr_gpt_base", "UCL, SPR", ymin = -0.55, ymax = 3.1, modelname="GPT")
plot_weights_spill("UCL_spr_small_base", "UCL, SPR", ymin = -0.55, ymax = 3.1, modelname="GPT-2$_{124M}$")
plot_weights_spill("UCL_spr_med_base", "UCL, SPR", ymin = -0.55, ymax = 3.1, modelname="GPT-2$_{355M}$")
plot_weights_spill("UCL_spr_large_base", "UCL, SPR", ymin = -0.55, ymax = 3.1, modelname="GPT-2$_{774M}$")
plot_weights_spill("UCL_spr_xl_base", "UCL, SPR", ymin = -0.55, ymax = 3.1, modelname="GPT-2$_{1.5B}$")

plot_weights_spill_withEMB("UCL_spr_gpt_all", " UCL, SPR", ymin = -0.4, ymax = .7, modelname="GPT")
plot_weights_spill_withEMB("UCL_spr_small_all", " UCL, SPR", ymin = -0.4, ymax = .7, modelname="GPT-2$_{124M}$")
plot_weights_spill_withEMB("UCL_spr_med_all", " UCL, SPR", ymin = -0.4, ymax = .7, modelname="GPT-2$_{355M}$")
plot_weights_spill_withEMB("UCL_spr_large_all", " UCL, SPR", ymin = -0.4, ymax = .7, modelname="GPT-2$_{774M}$")
plot_weights_spill_withEMB("UCL_spr_xl_all", " UCL, SPR", ymin = -0.4, ymax = .7, modelname="GPT-2$_{1.5B}$")



# [removed: SEM follow-up — semantic-norm analysis is not part of the manuscript]
#################################
#            #######            #
#            # SEM #            #
#            #######            #
#################################

# sem = get_sem(list(df_noinit["word"]))
# Missing affective = 70.97
# Missing perceptual = 11.24
# sem_spill = spill_embeddings(sem, to_filt)

###########
# NoSpill #
###########

# baseline
# gpt_base_sem   = np.concatenate((sem, np.array(df_noinit[["s_GPT", "logfreq", "length"]])), axis=1)
# small_base_sem = np.concatenate((sem, np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]])), axis=1)
# med_base_sem   = np.concatenate((sem, np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]])), axis=1)
# large_base_sem = np.concatenate((sem, np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]])), axis=1)
# xl_base_sem    = np.concatenate((sem, np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]])), axis=1)

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

#########
# spill #
#########

# baseline
# gpt_base_sem_spill     = np.concatenate((sem_spill, np.array(df_spill[["s_GPT", "logfreq", "length",
#                                                           "s_GPT-1", "logfreq-1", "length-1",
#                                                           "s_GPT-2", "logfreq-2", "length-2"]])), axis=1)
# small_base_sem_spill   = np.concatenate((sem_spill, np.array(df_spill[["s_GPT2_small", "logfreq", "length",
#                                                           "s_GPT2_small-1", "logfreq-1", "length-1",
#                                                           "s_GPT2_small-2", "logfreq-2", "length-2"]])), axis=1)
# med_base_sem_spill     = np.concatenate((sem_spill, np.array(df_spill[["s_GPT2_med", "logfreq", "length",
#                                                           "s_GPT2_med-1", "logfreq-1", "length-1",
#                                                           "s_GPT2_med-2", "logfreq-2", "length-2"]])), axis=1)
# large_base_sem_spill   = np.concatenate((sem_spill, np.array(df_spill[["s_GPT2_large", "logfreq", "length",
#                                                           "s_GPT2_large-1", "logfreq-1", "length-1",
#                                                           "s_GPT2_large-2", "logfreq-2", "length-2"]])), axis=1)
# xl_base_sem_spill      = np.concatenate((sem_spill, np.array(df_spill[["s_GPT2_xl", "logfreq", "length",
#                                                           "s_GPT2_xl-1", "logfreq-1", "length-1",
#                                                           "s_GPT2_xl-2", "logfreq-2", "length-2"]])), axis=1)

# embedding
# gpt_sem_spill   = np.concatenate((sem_spill, gpt_filt_spill), axis=1)
# small_sem_spill = np.concatenate((sem_spill, small_filt_spill), axis=1)
# med_sem_spill   = np.concatenate((sem_spill, med_filt_spill), axis=1)
# large_sem_spill = np.concatenate((sem_spill, large_filt_spill), axis=1)
# xl_sem_spill    = np.concatenate((sem_spill, xl_filt_spill), axis=1)

# all
# gpt_all_sem_spill   = np.concatenate((sem_spill, gpt_all_spill), axis=1)
# small_all_sem_spill = np.concatenate((sem_spill, gpt2_small_all_spill), axis=1)
# med_all_sem_spill   = np.concatenate((sem_spill, gpt2_med_all_spill), axis=1)
# large_all_sem_spill = np.concatenate((sem_spill, gpt2_large_all_spill), axis=1)
# xl_all_sem_spill    = np.concatenate((sem_spill, gpt2_xl_all_spill), axis=1)

####################
# ACTIVATIONS ONLY #
####################

# first-pass, activation only
# m_gpt_fpass_sem, sd_gpt_fpass_sem     = test_model_Ridge(gpt_sem, fp, 10, "UCL_fpass_gpt_sem")
# m_small_fpass_sem, sd_small_fpass_sem = test_model_Ridge(small_sem, fp, 10, "UCL_fpass_small_sem")
# m_med_fpass_sem, sd_med_fpas_sems     = test_model_Ridge(med_sem, fp, 10, "UCL_fpass_med_sem")
# m_larg_fpass_sem, sd_larg_fpass_sem   = test_model_Ridge(large_sem, fp, 10, "UCL_fpass_large_sem")
# m_xl_fpass_sem, sd_xl_fpass_sem       = test_model_Ridge(xl_sem, fp, 10, "UCL_fpass_xl_sem")
# print(m_gpt_fpass_sem, m_small_fpass_sem, m_med_fpass_sem, m_larg_fpass_sem, m_xl_fpass_sem)

# n400, activation only
# m_gpt_n400_sem, sd_gpt_n400_sem     = test_model_Ridge(gpt_sem[filtr], n400, 10, "UCL_n400_gpt_sem")
# m_small_n400_sem, sd_small_n400_sem = test_model_Ridge(small_sem[filtr], n400, 10, "UCL_n400_small_sem")
# m_med_n400_sem, sd_med_n400_sem     = test_model_Ridge(med_sem[filtr], n400, 10, "UCL_n400_med_sem")
# m_larg_n400_sem, sd_larg_n400_sem   = test_model_Ridge(large_sem[filtr], n400, 10, "UCL_n400_large_sem")
# m_xl_n400_sem, sd_xl_n400_sem       = test_model_Ridge(xl_sem[filtr], n400, 10, "UCL_n400_xl_sem")
# print(m_gpt_n400_sem, m_small_n400_sem, m_med_n400_sem, m_larg_n400_sem, m_xl_n400_sem)

# p600, activation only
# m_gpt_p600_sem, sd_gpt_p600_sem     = test_model_Ridge(gpt_sem[filtr], p600, 10, "UCL_p600_gpt_sem")
# m_small_p600_sem, sd_small_p600_sem = test_model_Ridge(small_sem[filtr], p600, 10, "UCL_p600_small_sem")
# m_med_p600_sem, sd_med_p600_sem     = test_model_Ridge(med_sem[filtr], p600, 10, "UCL_p600_med_sem")
# m_larg_p600_sem, sd_larg_p600_sem   = test_model_Ridge(large_sem[filtr], p600, 10, "UCL_p600_large_sem")
# m_xl_p600_sem, sd_xl_p600_sem       = test_model_Ridge(xl_sem[filtr], p600, 10, "UCL_p600_xl_sem")
# print(m_gpt_p600_sem, m_small_p600_sem, m_med_p600_sem, m_larg_p600_sem, m_xl_p600_sem)

# SPR, spill
# m_gpt_spr_sem, sd_gpt_spr_sem     = test_model_Ridge(gpt_sem_spill, spr_spill, 10, "UCL_spr_gpt_sem")
# m_small_spr_sem, sd_small_spr_sem = test_model_Ridge(small_sem_spill, spr_spill, 10, "UCL_spr_small_sem")
# m_med_spr_sem, sd_med_spr_sem     = test_model_Ridge(med_sem_spill, spr_spill, 10, "UCL_spr_med_sem")
# m_larg_spr_sem, sd_larg_spr_sem   = test_model_Ridge(large_sem_spill, spr_spill, 10, "UCL_spr_large_sem")
# m_xl_spr_sem, sd_xl_spr_sem       = test_model_Ridge(xl_sem_spill, spr_spill, 10, "UCL_spr_xl_sem")
# print(m_gpt_spr_sem, m_small_spr_sem, m_med_spr_sem, m_larg_spr_sem, m_xl_spr_sem)

#################
# BASELINE ONLY #
#################

# first-pass, baseline only
# m_gpt_fpass_b_sem, sd_gpt_fpass_b_sem     = test_model_Ridge(gpt_base_sem, fp, 10, "UCL_fpass_gpt_base_sem")
# m_small_fpass_b_sem, sd_small_fpass_b_sem = test_model_Ridge(small_base_sem, fp, 10, "UCL_fpass_small_base_sem")
# m_med_fpass_b_sem, sd_med_fpass_b_sem     = test_model_Ridge(med_base_sem, fp, 10, "UCL_fpass_med_base_sem")
# m_larg_fpass_b_sem, sd_larg_fpass_b_sem   = test_model_Ridge(large_base_sem, fp, 10, "UCL_fpass_large_base_sem")
# m_xl_fpass_b_sem, sd_xl_fpass_b_sem       = test_model_Ridge(xl_base_sem, fp, 10, "UCL_fpass_xl_base_sem")
# print(m_gpt_fpass_b_sem, m_small_fpass_b_sem, m_med_fpass_b_sem, m_larg_fpass_b_sem, m_xl_fpass_b_sem)

# n400, baseline only
# m_gpt_n400_b_sem, sd_gpt_n400_b_sem     = test_model_Ridge(gpt_base_sem[filtr], n400, 10, "UCL_n400_gpt_base_sem")
# m_small_n400_b_sem, sd_small_n400_b_sem = test_model_Ridge(small_base_sem[filtr], n400, 10, "UCL_n400_small_base_sem")
# m_med_n400_b_sem, sd_med_n400_b_sem     = test_model_Ridge(med_base_sem[filtr], n400, 10, "UCL_n400_med_base_sem")
# m_larg_n400_b_sem, sd_larg_n400_b_sem   = test_model_Ridge(large_base_sem[filtr], n400, 10, "UCL_n400_large_base_sem")
# m_xl_n400_b_sem, sd_xl_n400_b_sem       = test_model_Ridge(xl_base_sem[filtr], n400, 10, "UCL_n400_xl_base_sem")
# print(m_gpt_n400_b_sem, m_small_n400_b_sem, m_med_n400_b_sem, m_larg_n400_b_sem, m_xl_n400_b_sem)

# p600, baseline only
# m_gpt_p600_b_sem, sd_gpt_p600_b_sem     = test_model_Ridge(gpt_base_sem[filtr], p600, 10, "UCL_p600_gpt_base_sem")
# m_small_p600_b_sem, sd_small_p600_b_sem = test_model_Ridge(small_base_sem[filtr], p600, 10, "UCL_p600_small_base_sem")
# m_med_p600_b_sem, sd_med_p600_b_sem     = test_model_Ridge(med_base_sem[filtr], p600, 10, "UCL_p600_med_base_sem")
# m_larg_p600_b_sem, sd_larg_p600_b_sem   = test_model_Ridge(large_base_sem[filtr], p600, 10, "UCL_p600_large_base_sem")
# m_xl_p600_b_sem, sd_xl_p600_b_sem       = test_model_Ridge(xl_base_sem[filtr], p600, 10, "UCL_p600_xl_base_sem")
# print(m_gpt_p600_b_sem, m_small_p600_b_sem, m_med_p600_b_sem, m_larg_p600_b_sem, m_xl_p600_b_sem)

# spr_spill, baseline only
# m_gpt_spr_b_sem, sd_gpt_spr_b_sem     = test_model_Ridge(gpt_base_sem_spill, spr_spill, 10, "UCL_spr_gpt_base_sem")
# m_small_spr_b_sem, sd_small_spr_b_sem = test_model_Ridge(small_base_sem_spill, spr_spill, 10, "UCL_spr_small_base_sem")
# m_med_spr_b_sem, sd_med_spr_b_sem     = test_model_Ridge(med_base_sem_spill, spr_spill, 10, "UCL_spr_med_base_sem")
# m_larg_spr_b_sem, sd_larg_spr_b_sem   = test_model_Ridge(large_base_sem_spill, spr_spill, 10, "UCL_spr_large_base_sem")
# m_xl_spr_b_sem, sd_xl_spr_b_sem       = test_model_Ridge(xl_base_sem_spill, spr_spill, 10, "UCL_spr_xl_base_sem")
# print(m_gpt_spr_b_sem, m_small_spr_b_sem, m_med_spr_b_sem, m_larg_spr_b_sem, m_xl_spr_b_sem)

#######
# ALL #
#######

# first-pass
# m_gpt_fpass_all_sem, sd_gpt_fpass_all_sem     = test_model_Ridge(gpt_all_sem, fp, 10, "UCL_fpass_gpt_all_sem")
# m_small_fpass_all_sem, sd_small_fpass_all_sem = test_model_Ridge(small_all_sem, fp, 10, "UCL_fpass_small_all_sem")
# m_med_fpass_all_sem, sd_med_fpass_all_sem     = test_model_Ridge(med_all_sem, fp, 10, "UCL_fpass_med_all_sem")
# m_larg_fpass_all_sem, sd_larg_fpass_all_sem   = test_model_Ridge(large_all_sem, fp, 10, "UCL_fpass_large_all_sem")
# m_xl_fpass_all_sem, sd_xl_fpass_all_sem       = test_model_Ridge(xl_all_sem, fp, 10, "UCL_fpass_xl_all_sem")
# print(m_gpt_fpass_all_sem, m_small_fpass_all_sem, m_med_fpass_all_sem, m_larg_fpass_all_sem, m_xl_fpass_all_sem)

# n400
# m_gpt_n400_all_sem, sd_gpt_n400_all_sem     = test_model_Ridge(gpt_all_sem[filtr], n400, 10, "UCL_n400_gpt_all_sem")
# m_small_n400_all_sem, sd_small_n400_all_sem = test_model_Ridge(small_all_sem[filtr], n400, 10, "UCL_n400_small_all_sem")
# m_med_n400_all_sem, sd_med_n400_all_sem     = test_model_Ridge(med_all_sem[filtr], n400, 10, "UCL_n400_med_all_sem")
# m_larg_n400_all_sem, sd_larg_n400_all_sem   = test_model_Ridge(large_all_sem[filtr], n400, 10, "UCL_n400_large_all_sem")
# m_xl_n400_all_sem, sd_xl_n400_all_sem       = test_model_Ridge(xl_all_sem[filtr], n400, 10, "UCL_n400_xl_all_sem")
# print(m_gpt_n400_all_sem, m_small_n400_all_sem, m_med_n400_all_sem, m_larg_n400_all_sem, m_xl_n400_all_sem)

# p600
# m_gpt_p600_all_sem, sd_gpt_p600_all_sem     = test_model_Ridge(gpt_all_sem[filtr], p600, 10, "UCL_p600_gpt_all_sem")
# m_small_p600_all_sem, sd_small_p600_all_sem = test_model_Ridge(small_all_sem[filtr], p600, 10, "UCL_p600_small_all_sem")
# m_med_p600_all_sem, sd_med_p600_all_sem     = test_model_Ridge(med_all_sem[filtr], p600, 10, "UCL_p600_med_all_sem")
# m_larg_p600_all_sem, sd_larg_p600_all_sem   = test_model_Ridge(large_all_sem[filtr], p600, 10, "UCL_p600_large_all_sem")
# m_xl_p600_all_sem, sd_xl_p600_all_sem       = test_model_Ridge(xl_all_sem[filtr], p600, 10, "UCL_p600_xl_all_sem")
# print(m_gpt_p600_all_sem, m_small_p600_all_sem, m_med_p600_all_sem, m_larg_p600_all_sem, m_xl_p600_all_sem)

# spr
# m_gpt_spr_all_sem, sd_gpt_spr_all_sem     = test_model_Ridge(gpt_all_sem_spill, spr_spill, 10, "UCL_spr_gpt_all_sem")
# m_small_spr_all_sem, sd_small_spr_all_sem = test_model_Ridge(small_all_sem_spill, spr_spill, 10, "UCL_spr_small_all_sem")
# m_med_spr_all_sem, sd_med_spr_all_sem     = test_model_Ridge(med_all_sem_spill, spr_spill, 10, "UCL_spr_med_all_sem")
# m_larg_spr_all_sem, sd_larg_spr_all_sem   = test_model_Ridge(large_all_sem_spill, spr_spill, 10, "UCL_spr_large_all_sem")
# m_xl_spr_all_sem, sd_xl_spr_all_sem       = test_model_Ridge(xl_all_sem_spill, spr_spill, 10, "UCL_spr_xl_all_sem")
# print(m_gpt_spr_all_sem, m_small_spr_all_sem, m_med_spr_all_sem, m_larg_spr_all_sem, m_xl_spr_all_sem)

# df_results_sem = pd.DataFrame({
#     "base_fpass_sem":[m_gpt_fpass_b_sem, m_small_fpass_b_sem, m_med_fpass_b_sem, m_larg_fpass_b_sem, m_xl_fpass_b_sem],
#     "base_fpass_sem_sd": [sd_gpt_fpass_b_sem, sd_small_fpass_b_sem, sd_med_fpass_b_sem, sd_larg_fpass_b_sem, sd_xl_fpass_b_sem],
#     "base_n400_sem":  [m_gpt_n400_b_sem, m_small_n400_b_sem, m_med_n400_b_sem, m_larg_n400_b_sem, m_xl_n400_b_sem],
#     "base_n400_sem_sd":  [sd_gpt_n400_b_sem, sd_small_n400_b_sem, sd_med_n400_b_sem, sd_larg_n400_b_sem, sd_xl_n400_b_sem],
#     "base_p600_sem":  [m_gpt_p600_b_sem, m_small_p600_b_sem, m_med_p600_b_sem, m_larg_p600_b_sem, m_xl_p600_b_sem],
#     "base_p600_sem_sd":  [sd_gpt_p600_b_sem, sd_small_p600_b_sem, sd_med_p600_b_sem, sd_larg_p600_b_sem, sd_xl_p600_b_sem],
#     "base_spr_sem": [m_gpt_spr_b_sem, m_small_spr_b_sem, m_med_spr_b_sem, m_larg_spr_b_sem, m_xl_spr_b_sem],
#     "base_spr_sem_sd":[sd_gpt_spr_b_sem, sd_small_spr_b_sem, sd_med_spr_b_sem, sd_larg_spr_b_sem, sd_xl_spr_b_sem],
#     "act_fpass_sem":[m_gpt_fpass_sem, m_small_fpass_sem, m_med_fpass_sem, m_larg_fpass_sem, m_xl_fpass_sem],
#     "act_fpass_sem_sd": [sd_gpt_fpass_sem, sd_small_fpass_sem, sd_med_fpas_sems, sd_larg_fpass_sem, sd_xl_fpass_sem],
#     "act_n400_sem":  [m_gpt_n400_sem, m_small_n400_sem, m_med_n400_sem, m_larg_n400_sem, m_xl_n400_sem],
#     "act_n400_sem_sd":  [sd_gpt_n400_sem, sd_small_n400_sem, sd_med_n400_sem, sd_larg_n400_sem, sd_xl_n400_sem],
#     "act_p600_sem":  [m_gpt_p600_sem, m_small_p600_sem, m_med_p600_sem, m_larg_p600_sem, m_xl_p600_sem],
#     "act_p600_sem_sd":  [sd_gpt_p600_sem, sd_small_p600_sem, sd_med_p600_sem, sd_larg_p600_sem, sd_xl_p600_sem],
#     "act_spr_sem": [m_gpt_spr_sem, m_small_spr_sem, m_med_spr_sem, m_larg_spr_sem, m_xl_spr_sem],
#     "act_spr_sem_sd":[sd_gpt_spr_sem, sd_small_spr_sem, sd_med_spr_sem, sd_larg_spr_sem, sd_xl_spr_sem],
#     "all_fpass_sem":[m_gpt_fpass_all_sem, m_small_fpass_all_sem, m_med_fpass_all_sem, m_larg_fpass_all_sem, m_xl_fpass_all_sem],
#     "all_fpass_sem_sd":[sd_gpt_fpass_all_sem, sd_small_fpass_all_sem, sd_med_fpass_all_sem, sd_larg_fpass_all_sem, sd_xl_fpass_all_sem],
#     "all_n400_sem":[m_gpt_n400_all_sem, m_small_n400_all_sem, m_med_n400_all_sem, m_larg_n400_all_sem, m_xl_n400_all_sem],
#     "all_n400_sem_sd":[sd_gpt_n400_all_sem, sd_small_n400_all_sem, sd_med_n400_all_sem, sd_larg_n400_all_sem, sd_xl_n400_all_sem],
#     "all_p600_sem":[m_gpt_p600_all_sem, m_small_p600_all_sem, m_med_p600_all_sem, m_larg_p600_all_sem, m_xl_p600_all_sem],
#     "all_p600_sem_sd":[sd_gpt_p600_all_sem, sd_small_p600_all_sem, sd_med_p600_all_sem, sd_larg_p600_all_sem, sd_xl_p600_all_sem],
#     "all_spr_sem":[m_gpt_spr_all_sem, m_small_spr_all_sem, m_med_spr_all_sem, m_larg_spr_all_sem, m_xl_spr_all_sem],
#     "all_spr_sem_sd":[sd_gpt_spr_all_sem, sd_small_spr_all_sem, sd_med_spr_all_sem, sd_larg_spr_all_sem, sd_xl_spr_all_sem]
#     })
# df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_results_sem.to_csv("SEM/results_UCL_SEM.csv", index=True)


########################
# significance_testing #
########################
# thebase_fpass = [x+"_sem" for x in ["UCL_fpass_gpt_base", "UCL_fpass_small_base", "UCL_fpass_med_base", "UCL_fpass_large_base", "UCL_fpass_xl_base"]]
# theemb_fpass  = [x+"_sem" for x in["UCL_fpass_gpt", "UCL_fpass_small", "UCL_fpass_med", "UCL_fpass_large", "UCL_fpass_xl"]]
# all_fpass     = [x+"_sem" for x in["UCL_fpass_gpt_all", "UCL_fpass_small_all", "UCL_fpass_med_all", "UCL_fpass_large_all", "UCL_fpass_xl_all"]]

# thebase_n400 = [x+"_sem" for x in["UCL_n400_gpt_base", "UCL_n400_small_base", "UCL_n400_med_base", "UCL_n400_large_base", "UCL_n400_xl_base"]]
# theemb_n400  = [x+"_sem" for x in["UCL_n400_gpt", "UCL_n400_small", "UCL_n400_med", "UCL_n400_large", "UCL_n400_xl"]]
# all_n400     = [x+"_sem" for x in["UCL_n400_gpt_all", "UCL_n400_small_all", "UCL_n400_med_all", "UCL_n400_large_all", "UCL_n400_xl_all"]]

# thebase_p600 = [x+"_sem" for x in["UCL_p600_gpt_base", "UCL_p600_small_base", "UCL_p600_med_base", "UCL_p600_large_base", "UCL_p600_xl_base"]]
# theemb_p600  = [x+"_sem" for x in["UCL_p600_gpt", "UCL_p600_small", "UCL_p600_med", "UCL_p600_large", "UCL_p600_xl"]]
# all_p600     = [x+"_sem" for x in["UCL_p600_gpt_all", "UCL_p600_small_all", "UCL_p600_med_all", "UCL_p600_large_all", "UCL_p600_xl_all"]]

# thebase_spr = [x+"_sem" for x in["UCL_spr_gpt_base", "UCL_spr_small_base", "UCL_spr_med_base", "UCL_spr_large_base", "UCL_spr_xl_base"]]
# theemb_spr  = [x+"_sem" for x in["UCL_spr_gpt", "UCL_spr_small", "UCL_spr_med", "UCL_spr_large", "UCL_spr_xl"]]
# all_spr     = [x+"_sem" for x in["UCL_spr_gpt_all", "UCL_spr_small_all", "UCL_spr_med_all", "UCL_spr_large_all", "UCL_spr_xl_all"]]

# p_fpass = []
# for base, emb, all_ in zip(thebase_fpass, theemb_fpass, all_fpass):
#     p_base_emb = significance_testing(base, emb)
#     p_all_base = significance_testing(all_, base)
#     p_all_emb  = significance_testing(all_, emb)
#     p_fpass.append([p_base_emb, p_all_base, p_all_emb])
# p_fpass = pd.DataFrame(p_fpass, columns=["base_emb", "all_base", "all_emb"])

# p_spr = []
# for base, emb, all_ in zip(thebase_spr, theemb_spr, all_spr):
#     p_base_emb = significance_testing(base, emb)
#     p_all_base = significance_testing(all_, base)
#     p_all_emb  = significance_testing(all_, emb)
#     p_spr.append([p_base_emb, p_all_base, p_all_emb])
# p_spr = pd.DataFrame(p_spr, columns=["base_emb", "all_base", "all_emb"])

# p_n400 = []
# for base, emb, all_ in zip(thebase_n400, theemb_n400, all_n400):
#     p_emb_base = significance_testing(emb, base)
#     p_all_base = significance_testing(all_, base)
#     p_all_emb  = significance_testing(all_, emb)
#     p_n400.append([p_base_emb, p_all_base, p_all_emb])
# p_n400 = pd.DataFrame(p_n400, columns=["base_emb", "all_base", "all_emb"])

# p_p600 = []
# for base, emb, all_ in zip(thebase_p600, theemb_p600, all_p600):
#     p_emb_base = significance_testing(emb, base)
#     p_all_base = significance_testing(all_, base)
#     p_all_emb  = significance_testing(all_, emb)
#     p_p600.append([p_base_emb, p_all_base, p_all_emb])
# p_p600 = pd.DataFrame(p_p600, columns=["base_emb", "all_base", "all_emb"])

# for df, dfname in zip([p_fpass, p_spr, p_n400, p_p600], ["fpass", "spr", "n400", "p600"]):
#     for df_column in df.columns:
#         df_results_sem["p_"+dfname+"_"+df_column+"_sem"] = list(df[df_column])

# merged = pd.concat([df_results, df_results_sem], axis=1)
# merged.columns
# merged.to_csv("SEM/merged_UCL.csv")

# merged = pd.read_csv("SEM/merged_UCL.csv")
# merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

############
# plotting #
############

# plot_sem(merged, "act_n400", "base_n400", "all_n400", "act_n400_sem", "base_n400_sem", "all_n400_sem", title="N400", subtract=True)

# plot_sem(merged, "act_p600", "base_p600", "all_p600", "act_p600_sem", "base_p600_sem", "all_p600_sem", title="P600", subtract=True)

# plot_sem(merged, "act_fpass", "base_fpass", "all_fpass", "act_fpass_sem", "base_fpass_sem", "all_fpass_sem", title="fpass", subtract=True)

# plot_sem(merged, "act_spr", "base_spr", "all_spr", "act_spr_sem", "base_spr_sem", "all_spr_sem", title="spr", subtract=True)

# plot_weights_sem("UCL_n400_small_base_sem", "UCL N400", "weights", ymin = -.1, ymax = .1)


#######################
# for weight analysis #
#######################

# first-pass, activation only
probe(med_filt, fp, 10, "UCL_fpass_med_probing")

# n400, activation only
probe(med_filt_f, n400, 10, "UCL_n400_med_probing")

# p600, activations only
probe(med_filt_f, p600, 10, "UCL_p600_med_probing")

# SPR, spill
probe(med_filt_spill, spr_spill, 10, "UCL_spr_med_probing")


# # first-pass, activation only
# m_small_fpass, sd_small_fpass     = probe(small_filt, fp, 10, "UCL_fpass_small_probing")

# # n400, activation only
# m_small_n400, sd_small_n400     = probe(small_filt_f, n400, 10, "UCL_n400_small_probing")

# # p600, activations only
# m_small_p600, sd_small_p600     = probe(small_filt_f, p600, 10, "UCL_p600_small_probing")

# # SPR, spill
# m_small_spr, sd_small_spr     = probe(small_filt_spill, spr_spill, 10, "UCL_small_med_probing")




# from joblib import load
# pca = load('probing_LMs/pca.pkl')
# med_filt_pca = pca.transform(med_filt)
# med_filt_f_pca = pca.transform(med_filt_f)

# med_filt_spill_pca = spill_embeddings(med_filt_pca, to_filt)

# # first-pass, activation only
# m_med_fpass, sd_med_fpass = test_model_Ridge(med_filt_pca, fp, 10, "UCL_fpass_med_probing_pca")
# m_med_n400, sd_med_n400   = test_model_Ridge(med_filt_f_pca, n400, 10, "UCL_n400_med_probing_pca")
# m_med_p600, sd_med_p600   = test_model_Ridge(med_filt_f_pca, p600, 10, "UCL_p600_med_probing_pca")
# m_med_spr, sd_med_spr     = test_model_Ridge(med_filt_spill_pca, spr_spill, 10, "UCL_spr_med_probing_pca")



##########################
# end of weight analysis #
##########################
# save med_filt, y
dataname = "ucl_fpass"
tosave = [med_filt, fp]
# with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# save med_filt, y
dataname = "ucl_n400"
tosave = [med_filt_f, n400]
# with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# save med_filt, y
dataname = "ucl_p600"
tosave = [med_filt_f, n400]
# with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# save med_filt, y
dataname = "ucl_spr"
tosave = [med_filt_spill, spr_spill]
# with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
##############################################################################
###############################
# additional fix measurements #
###############################

####################
# ACTIVATIONS ONLY #
####################

# ffix
m_gpt_ffix, sd_gpt_ffix     = test_model_Ridge(gpt_filt, ff, 10, "UCL_ffix_gpt")
m_small_ffix, sd_small_ffix = test_model_Ridge(small_filt, ff, 10, "UCL_ffix_small")
m_med_ffix, sd_med_ffix     = test_model_Ridge(med_filt, ff, 10, "UCL_ffix_med")
m_larg_ffix, sd_larg_ffix   = test_model_Ridge(large_filt, ff, 10, "UCL_ffix_large")
m_xl_ffix, sd_xl_ffix       = test_model_Ridge(xl_filt, ff, 10, "UCL_ffix_xl")
print(m_gpt_ffix, m_small_ffix, m_med_ffix, m_larg_ffix, m_xl_ffix)

# totfix
m_gpt_totfix, sd_gpt_totfix     = test_model_Ridge(gpt_filt, tf, 10, "UCL_totfix_gpt")
m_small_totfix, sd_small_totfix = test_model_Ridge(small_filt, tf, 10, "UCL_totfix_small")
m_med_totfix, sd_med_totfix     = test_model_Ridge(med_filt, tf, 10, "UCL_totfix_med")
m_larg_totfix, sd_larg_totfix   = test_model_Ridge(large_filt, tf, 10, "UCL_totfix_large")
m_xl_totfix, sd_xl_totfix       = test_model_Ridge(xl_filt, tf, 10, "UCL_totfix_xl")
print(m_gpt_totfix, m_small_totfix, m_med_totfix, m_larg_totfix, m_xl_totfix)

#################
# BASELINE ONLY #
#################

# ffix, baseline only
m_gpt_ffix_b, sd_gpt_ffix_b     = test_model_Ridge(np.array(df_noinit[["s_GPT", "logfreq", "length"]]), ff, 10, "UCL_ffix_gpt_base")
m_small_ffix_b, sd_small_ffix_b = test_model_Ridge(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]]), ff, 10, "UCL_ffix_small_base")
m_med_ffix_b, sd_med_ffix_b     = test_model_Ridge(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), ff, 10, "UCL_ffix_med_base")
m_larg_ffix_b, sd_larg_ffix_b   = test_model_Ridge(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]]), ff, 10, "UCL_ffix_large_base")
m_xl_ffix_b, sd_xl_ffix_b       = test_model_Ridge(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]]), ff, 10, "UCL_ffix_xl_base")
print(m_gpt_ffix_b, m_small_ffix_b, m_med_ffix_b, m_larg_ffix_b, m_xl_ffix_b)

# totfix
m_gpt_totfix_b, sd_gpt_totfix_b     = test_model_Ridge(np.array(df_noinit[["s_GPT", "logfreq", "length"]]), tf, 10, "UCL_totfix_gpt_base")
m_small_totfix_b, sd_small_totfix_b = test_model_Ridge(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]]), tf, 10, "UCL_totfix_small_base")
m_med_totfix_b, sd_med_totfix_b     = test_model_Ridge(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), tf, 10, "UCL_totfix_med_base")
m_larg_totfix_b, sd_larg_totfix_b   = test_model_Ridge(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]]), tf, 10, "UCL_totfix_large_base")
m_xl_totfix_b, sd_xl_totfix_b       = test_model_Ridge(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]]), tf, 10, "UCL_totfix_xl_base")
print(m_gpt_totfix_b, m_small_totfix_b, m_med_totfix_b, m_larg_totfix_b, m_xl_totfix_b)

# decompose SFL
decompose_sfl(np.array(df_noinit[["s_GPT", "logfreq", "length"]]), ff, "UCL_ffix_gpt_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]]), ff, "UCL_ffix_small_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), ff, "UCL_ffix_med_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]]), ff, "UCL_ffix_large_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]]), ff, "UCL_ffix_xl_base")

decompose_sfl(np.array(df_noinit[["s_GPT", "logfreq", "length"]]), tf, "UCL_totfix_gpt_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]]), tf, "UCL_totfix_small_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), tf, "UCL_totfix_med_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]]), tf, "UCL_totfix_large_base")
decompose_sfl(np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]]), tf, "UCL_totfix_xl_base")

##########################
# BASELINE + ACTIVATIONS #
##########################

gpt_all        = np.concatenate((np.array(df_noinit[["s_GPT", "logfreq", "length"]]), gpt_filt), axis=1)
gpt2_small_all = np.concatenate((np.array(df_noinit[["s_GPT2_small", "logfreq", "length"]]), small_filt), axis=1)
gpt2_med_all   = np.concatenate((np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), med_filt), axis=1)
gpt2_large_all = np.concatenate((np.array(df_noinit[["s_GPT2_large", "logfreq", "length"]]), large_filt), axis=1)
gpt2_xl_all    = np.concatenate((np.array(df_noinit[["s_GPT2_xl", "logfreq", "length"]]), xl_filt), axis=1)

# ffix
m_gpt_ffix_all, sd_gpt_ffix_all     = test_model_Ridge(gpt_all, ff, 10, "UCL_ffix_gpt_all")
m_small_ffix_all, sd_small_ffix_all = test_model_Ridge(gpt2_small_all, ff, 10, "UCL_ffix_small_all")
m_med_ffix_all, sd_med_ffix_all     = test_model_Ridge(gpt2_med_all, ff, 10, "UCL_ffix_med_all")
m_larg_ffix_all, sd_larg_ffix_all   = test_model_Ridge(gpt2_large_all, ff, 10, "UCL_ffix_large_all")
m_xl_ffix_all, sd_xl_ffix_all       = test_model_Ridge(gpt2_xl_all, ff, 10, "UCL_ffix_xl_all")
print(m_gpt_ffix_all, m_small_ffix_all, m_med_ffix_all, m_larg_ffix_all, m_xl_ffix_all)

# totfix
m_gpt_totfix_all, sd_gpt_totfix_all     = test_model_Ridge(gpt_all, tf, 10, "UCL_totfix_gpt_all")
m_small_totfix_all, sd_small_totfix_all = test_model_Ridge(gpt2_small_all, tf, 10, "UCL_totfix_small_all")
m_med_totfix_all, sd_med_totfix_all     = test_model_Ridge(gpt2_med_all, tf, 10, "UCL_totfix_med_all")
m_larg_totfix_all, sd_larg_totfix_all   = test_model_Ridge(gpt2_large_all, tf, 10, "UCL_totfix_large_all")
m_xl_totfix_all, sd_xl_totfix_all       = test_model_Ridge(gpt2_xl_all, tf, 10, "UCL_totfix_xl_all")
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
    "all_totfix_sd":[sd_gpt_totfix_all, sd_small_totfix_all, sd_med_totfix_all, sd_larg_totfix_all, sd_xl_totfix_all],
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
# df_results.to_csv("results/results_UCL.csv", index=True)


########################
# significance_testing #
########################

thebase_ffix = ["UCL_ffix_gpt_base", "UCL_ffix_small_base", "UCL_ffix_med_base", "UCL_ffix_large_base", "UCL_ffix_xl_base"]
theemb_ffix  = ["UCL_ffix_gpt", "UCL_ffix_small", "UCL_ffix_med", "UCL_ffix_large", "UCL_ffix_xl"]
all_ffix     = ["UCL_ffix_gpt_all", "UCL_ffix_small_all", "UCL_ffix_med_all", "UCL_ffix_large_all", "UCL_ffix_xl_all"]

p_ffix = []
for base, emb, all_ in zip(thebase_ffix, theemb_ffix, all_ffix):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_ffix.append([p_base_emb, p_all_base, p_all_emb])
p_ffix = pd.DataFrame(p_ffix, columns=["base_emb", "all_base", "all_emb"])

thebase_totfix = ["UCL_totfix_gpt_base", "UCL_totfix_small_base", "UCL_totfix_med_base", "UCL_totfix_large_base", "UCL_totfix_xl_base"]
theemb_totfix  = ["UCL_totfix_gpt", "UCL_totfix_small", "UCL_totfix_med", "UCL_totfix_large", "UCL_totfix_xl"]
all_totfix     = ["UCL_totfix_gpt_all", "UCL_totfix_small_all", "UCL_totfix_med_all", "UCL_totfix_large_all", "UCL_totfix_xl_all"]

p_totfix = []
for base, emb, all_ in zip(thebase_totfix, theemb_totfix, all_totfix):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_totfix.append([p_base_emb, p_all_base, p_all_emb])
p_totfix = pd.DataFrame(p_totfix, columns=["base_emb", "all_base", "all_emb"])

for df, dfname in zip([p_ffix, p_totfix], ["ffix", "totfix"]):
    for df_column in df.columns:
        df_results["p_"+dfname+"_"+df_column] = list(df[df_column])

df_results.to_csv("results/results_UCL_ffix.csv", index=True)
