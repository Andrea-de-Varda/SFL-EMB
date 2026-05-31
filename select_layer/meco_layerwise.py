import neurox
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from scipy.special import softmax
import torch.nn.functional as F
import numpy as np
from os import chdir, path
import pandas as pd
from tqdm import tqdm
import pickle
import re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from glmnet import ElasticNet
import matplotlib.patches as mpatches

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, decompose_sfl

########
# MECO #
########

meco = pd.read_csv("/home/dev/Documents/PhD/side-projects/eye-tracking/Meco_l1.csv", index_col=None)
meco = meco.iloc[: , 1:]; print("N (with NA) =", len(meco))
meco = meco[meco['ia'].notna()]; print("N (no NA) =",len(meco))
meco["trialid"] = meco.trialid.astype(int)
meco["itemid"] = meco.itemid.astype(int)
meco["ianum"] = meco.ianum.astype(int)
meco["unique"] = meco.trialid.astype(str)+"-"+meco.ianum.astype(str)+"-"+meco.lang
meco = meco[~meco["subid"].isin(["ee_09", "ee_22", "ru_8"])] # wrong itemid assignment (as in 1.2 release)

languages = ['it', 'ee', 'en', 'fi', 'ge', 'gr', 'he', 'du', 'ko', 'no', 'ru', 'sp', 'tr']
out = {lang:[] for lang in languages}
for language in languages: 
    print("\n\n", language.upper())
    for textid in set(meco.trialid):
        print(textid)
        temp = meco[(meco["lang"] == language) & (meco["trialid"] == textid)]
        therange = range(1, max(temp.ianum)+1)
        temp_out = []
        temp_out_unique = []
        for itemindex in therange:
            try:
                token = list(set(temp[temp.ianum == itemindex].ia))
                unique = list(set(temp[temp.ianum == itemindex].unique))
                if len(token) > 1:
                    print(textid, itemindex, token)
                temp_out_unique.append(unique[0])
                temp_out.append(token[0])
            except IndexError:
                print(language, textid, itemindex)
        out[language].append([textid, temp_out, temp_out_unique])
# out = lang: [textN, words, unique]

# with open('corpora/meco_en_sentences.txt', 'w') as f:
#     for text in out["en"]:
#         line = " ".join(text[1])
#         f.write(f"{line}\n")

###########
# get SFL #
###########

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

# length     
def get_l(word):
    return len(word)

def get_surprisal(prompt):
    inputs = toker(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return [-item[0] for item in logprobs.tolist()[0]]

def tok_maker(a, sep, cased = False):
    # Credit to Ben S. https://stackoverflow.com/questions/74458282/match-strings-of-different-length-in-two-lists-of-different-length
    plainseq = " ".join(a)
    b = [re.sub(sep, "", item) for item in toker.tokenize(plainseq)]
    c = []
    if cased:
        for element in a:
            temp_list = []
            while "".join(temp_list) != element:
                temp_list.append(b.pop(0))
            c.append(temp_list)
    else:
        for element in a:
            temp_list = []
            while "".join(temp_list) != element.lower():
                temp_list.append(b.pop(0))
            c.append(temp_list)
    return c

def get_surprisal_tokens(tokens, sep, cased=False):
    s = get_surprisal(" ".join(tokens))
    toks = tok_maker(tokens, sep, cased)
    theindex = 0
    out = []
    for index, word in enumerate(toks[1:]):
        if len(word) == 1:
            surp = s[theindex]
            theindex += 1
            out.append(surp)
        else:
            surp = s[theindex:theindex+len(word)]
            theindex += len(word)
            out.append(sum(surp))
    return out

meco_sent = [re.sub("\n", "", s).split() for s in open("corpora/meco_en_sentences.txt").readlines()]
freq_meco = [get_f(t) for w in meco_sent for t in w[1:]]
len_meco =  [get_l(t) for w in meco_sent for t in w[1:]]

# GPT2 medium
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-medium")
s_med_meco = [s for sent in tqdm(meco_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

del model, toker

###################
# behavioral data #
###################

ff = []; fp = []; tf = []
for text in tqdm(out["en"]):
    ids = text[2]
    for uniq in ids[1:]:
        temp = meco[meco.unique == uniq] # fpass, count NaN (i.e. skips) as 0
        temp['firstfix.dur'] = temp['firstfix.dur'].fillna(0)
        temp['firstrun.dur'] = temp['firstrun.dur'].fillna(0)
        temp['dur'] = temp['dur'].fillna(0)
        rt_ff = np.nanmean(temp["firstfix.dur"])
        rt = np.nanmean(temp["firstrun.dur"])
        rt_tf = np.nanmean(temp["dur"])
        ff.append(rt_ff); fp.append(rt); tf.append(rt_tf)
fp = np.array(fp)
tf = np.array(tf)
ff = np.array(ff)

##################
# load model act #
##################

import neurox.data.loader as data_loader

act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium_meco.json', 1024, is_brnn=False)

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

##########################################

###########
# testing #
###########

m_med_fpass, sd_med_fpass     = test_model_Ridge(med_filt, fp, 10, "meco_med_layer3")

baseline_med   = np.array(list(zip(s_med_meco, freq_meco, len_meco)))
m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(baseline_med, fp, 10, "meco_med_base_layer3")

gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(gpt2_med_all, fp, 10, "meco_med_all_layer3")

df_results = pd.DataFrame({
    "base_fpass":[m_med_fpass_b],
    "base_fpass_sd": [sd_med_fpass_b],
    "act_fpass": [m_med_fpass],
    "act_fpass_sd": [sd_med_fpass],
    "all_fpass":[m_med_fpass_all],
    "all_fpass_sd":[sd_med_fpass_all]
    })
df_results.index = ["GPT-2$_{355M}$"]

thebase_fpass = ["meco_med_base_layer3"]
theemb_fpass  = ["meco_med_layer3"]
all_fpass     = ["meco_med_all_layer3"]

p_fpass = []
for base, emb, all_ in zip(thebase_fpass, theemb_fpass, all_fpass):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_fpass.append([p_base_emb, p_all_base, p_all_emb])
p_fpass = pd.DataFrame(p_fpass, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_fpass.columns:
    df_results["p_fpass_"+df_column] = list(p_fpass[df_column])
    
df_results.to_csv("select_layer/results_meco.csv", index=False)
