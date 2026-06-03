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

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT)
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, decompose_sfl, probe

########
# MECO #
########

meco = pd.read_csv(PROJECT_ROOT / "main_data/Meco/Meco_l1.csv", index_col=None)
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

import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

#######
# GPT #
#######

# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt_meco.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "corpora/meco_en_sentences.txt",
        dirName+'/activations_gpt_meco.json',
        aggregation="average"
    )

################
# GPT-2 FAMILY #
################

# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small_meco.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "corpora/meco_en_sentences.txt",
        dirName+'/activations_gpt2_small_meco.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium_meco.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "corpora/meco_en_sentences.txt",
        dirName+'/activations_gpt2_medium_meco.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large_meco.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "corpora/meco_en_sentences.txt",
        dirName+'/activations_gpt2_large_meco.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl_meco.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "corpora/meco_en_sentences.txt",
        dirName+'/activations_gpt2_xl_meco.json',
        aggregation="average"
    )

###########
# get SFL #
###########

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

# GPT
model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt_meco = [s for sent in tqdm(meco_sent) for s in get_surprisal_tokens(sent, sep = "</w>")]

# GPT2 SMALL (124M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2")
s_small_meco = [s for sent in tqdm(meco_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 medium
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-medium")
s_med_meco = [s for sent in tqdm(meco_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 large
model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-large")
s_large_meco = [s for sent in tqdm(meco_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 xl
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl_meco = [s for sent in tqdm(meco_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

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

act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt_meco.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small_meco.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium_meco.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large_meco.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl_meco.json', 1600, is_brnn=False)

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

##########################################
# from sklearn.model_selection import KFold

# def test_model(X, y, n):
#     kf = KFold(n_splits=n)
#     out_reg = []
#     for train_index, test_index in tqdm(kf.split(X), total=n):
#         X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#         reg = ElasticNet()
#         reg.fit(X_train, y_train)
#         y_pred = reg.predict(X_test)
#         r, _ = pearsonr(y_test, y_pred)
#         out_reg.append(r)
#     return np.mean(out_reg), np.std(out_reg)

###########
# testing #
###########
    
# first-pass, activation only
m_gpt_fpass, sd_gpt_fpass     = test_model_Ridge(gpt_filt, fp, 10, "meco_gpt")
m_small_fpass, sd_small_fpass = test_model_Ridge(small_filt, fp, 10, "meco_small")
m_med_fpass, sd_med_fpass     = test_model_Ridge(med_filt, fp, 10, "meco_med")
m_larg_fpass, sd_larg_fpass   = test_model_Ridge(large_filt, fp, 10, "meco_large")
m_xl_fpass, sd_xl_fpass       = test_model_Ridge(xl_filt, fp, 10, "meco_xl")
print(m_gpt_fpass, m_small_fpass, m_med_fpass, m_larg_fpass)

# first-pass, baseline only
baseline_gpt   = np.array(list(zip(s_gpt_meco, freq_meco, len_meco)))
baseline_small = np.array(list(zip(s_small_meco, freq_meco, len_meco)))
baseline_med   = np.array(list(zip(s_med_meco, freq_meco, len_meco)))
baseline_large = np.array(list(zip(s_large_meco, freq_meco, len_meco)))
baseline_xl    = np.array(list(zip(s_xl_meco, freq_meco, len_meco)))

m_gpt_fpass_b, sd_gpt_fpass_b     = test_model_Ridge(baseline_gpt, fp, 10, "meco_gpt_base")
m_small_fpass_b, sd_small_fpass_b = test_model_Ridge(baseline_small, fp, 10, "meco_small_base")
m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(baseline_med, fp, 10, "meco_med_base")
m_larg_fpass_b, sd_larg_fpass_b   = test_model_Ridge(baseline_large, fp, 10, "meco_large_base")
m_xl_fpass_b, sd_xl_fpass_b       = test_model_Ridge(baseline_xl, fp, 10, "meco_xl_base")
print(m_gpt_fpass_b, m_small_fpass_b, m_med_fpass_b, m_larg_fpass_b, m_xl_fpass_b)

# decompose_sfl
decompose_sfl(baseline_gpt, fp, "meco_gpt_base")
decompose_sfl(baseline_small, fp, "meco_small_base")
decompose_sfl(baseline_med, fp, "meco_med_base")
decompose_sfl(baseline_large, fp, "meco_large_base")
decompose_sfl(baseline_xl, fp, "meco_xl_base")

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_fpass_all, sd_gpt_fpass_all     = test_model_Ridge(gpt_all, fp, 10, "meco_gpt_all")
m_small_fpass_all, sd_small_fpass_all = test_model_Ridge(gpt2_small_all, fp, 10, "meco_small_all")
m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(gpt2_med_all, fp, 10, "meco_med_all")
m_larg_fpass_all, sd_larg_fpass_all   = test_model_Ridge(gpt2_large_all, fp, 10, "meco_large_all")
m_xl_fpass_all, sd_xl_fpass_all       = test_model_Ridge(gpt2_xl_all, fp, 10, "meco_xl_all")
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

base_plot(df_results, "act_fpass", "base_fpass", "all_fpass")

thebase_fpass = ["meco_gpt_base", "meco_small_base", "meco_med_base", "meco_large_base", "meco_xl_base"]
theemb_fpass  = ["meco_gpt", "meco_small", "meco_med", "meco_large", "meco_xl"]
all_fpass     = ["meco_gpt_all", "meco_small_all", "meco_med_all", "meco_large_all", "meco_xl_all"]

p_fpass = []
for base, emb, all_ in zip(thebase_fpass, theemb_fpass, all_fpass):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_fpass.append([p_base_emb, p_all_base, p_all_emb])
p_fpass = pd.DataFrame(p_fpass, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_fpass.columns:
    df_results["p_fpass_"+df_column] = list(p_fpass[df_column])
    
# df_results.to_csv("results_meco.csv", index=False)
df_results = pd.read_csv("results_meco.csv")
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

#################################
# [removed: SEM follow-up — semantic-norm analysis is not part of the manuscript]
# FOLLOW-UP: SEMANTIC VARIABLES #
#################################

# from utils import plot_sem, get_sem, plot_weights_sem

# wordlist = [t for w in meco_sent for t in w[1:]]
# sem = get_sem(wordlist)
# print(sem.shape)

# Missing affective = 69.04
# Missing perceptual = 19.37

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

# activation only
# m_gpt_fpass, sd_gpt_fpass     = test_model_Ridge(gpt_sem, fp, 10, "meco_gpt_sem")
# m_small_fpass, sd_small_fpass = test_model_Ridge(small_sem, fp, 10, "meco_small_sem")
# m_med_fpass, sd_med_fpass     = test_model_Ridge(med_sem, fp, 10, "meco_med_sem")
# m_larg_fpass, sd_larg_fpass   = test_model_Ridge(large_sem, fp, 10, "meco_large_sem")
# m_xl_fpass, sd_xl_fpass       = test_model_Ridge(xl_sem, fp, 10, "meco_xl_sem")
# print(m_gpt_fpass, m_small_fpass, m_med_fpass, m_larg_fpass)

# baseline only
# m_gpt_fpass_b, sd_gpt_fpass_b     = test_model_Ridge(gpt_base_sem, fp, 10, "meco_gpt_base_sem")
# m_small_fpass_b, sd_small_fpass_b = test_model_Ridge(small_base_sem, fp, 10, "meco_small_base_sem")
# m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(med_base_sem, fp, 10, "meco_med_base_sem")
# m_larg_fpass_b, sd_larg_fpass_b   = test_model_Ridge(large_base_sem, fp, 10, "meco_large_base_sem")
# m_xl_fpass_b, sd_xl_fpass_b       = test_model_Ridge(xl_base_sem, fp, 10, "meco_xl_base_sem")
# print(m_gpt_fpass_b, m_small_fpass_b, m_med_fpass_b, m_larg_fpass_b, m_xl_fpass_b)

# all
# m_gpt_fpass_all, sd_gpt_fpass_all     = test_model_Ridge(gpt_all_sem, fp, 10, "meco_gpt_all_sem")
# m_small_fpass_all, sd_small_fpass_all = test_model_Ridge(small_all_sem, fp, 10, "meco_small_all_sem")
# m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(med_all_sem, fp, 10, "meco_med_all_sem")
# m_larg_fpass_all, sd_larg_fpass_all   = test_model_Ridge(large_all_sem, fp, 10, "meco_large_all_sem")
# m_xl_fpass_all, sd_xl_fpass_all       = test_model_Ridge(xl_all_sem, fp, 10, "meco_xl_all_sem")
# print(m_gpt_fpass_all, m_small_fpass_all, m_med_fpass_all, m_larg_fpass_all, m_xl_fpass_all)

# df_results_sem = pd.DataFrame({
#     "base_fpass_sem":[m_gpt_fpass_b, m_small_fpass_b, m_med_fpass_b, m_larg_fpass_b, m_xl_fpass_b],
#     "base_fpass_sem_sd": [sd_gpt_fpass_b, sd_small_fpass_b, sd_med_fpass_b, sd_larg_fpass_b, sd_xl_fpass_b],
#     "act_fpass_sem":[m_gpt_fpass, m_small_fpass, m_med_fpass, m_larg_fpass, m_xl_fpass],
#     "act_fpass_sem_sd": [sd_gpt_fpass, sd_small_fpass, sd_med_fpass, sd_larg_fpass, sd_xl_fpass],
#     "all_fpass_sem":[m_gpt_fpass_all, m_small_fpass_all, m_med_fpass_all, m_larg_fpass_all, m_xl_fpass_all],
#     "all_fpass_sem_sd":[sd_gpt_fpass_all, sd_small_fpass_all, sd_med_fpass_all, sd_larg_fpass_all, sd_xl_fpass_all]
#     })
# df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

# df_results = pd.read_csv("results_meco.csv")
# df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

# merged = pd.concat([df_results, df_results_sem], axis=1)
# merged.columns
#merged.to_csv("SEM/merged_meco.csv")

# plot_sem(merged, "act_fpass", "base_fpass", "all_fpass", "act_fpass_sem", "base_fpass_sem", "all_fpass_sem", title="fpass", subtract=True)

########################
# for last probing exp #
########################


probe(med_filt, fp, 10, "meco_med_probing")
# m_small_fpass, sd_small_fpass = test_model_Ridge(zscore(small_filt), zscore(fp), 10, "meco_small_probing")

# from joblib import load
# pca = load('probing_LMs/pca.pkl')
# med_filt_pca = pca.transform(med_filt)

# # first-pass, activation only
# m_med_fpass, sd_med_fpass = test_model_Ridge(med_filt_pca, fp, 10, "meco_med_probing_pca")


##########################
# end of weight analysis #
##########################
dataname = "meco"
tosave = [med_filt, fp]
# with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

######################
# Other fix measures #
######################

m_gpt_ffix, sd_gpt_ffix     = test_model_Ridge(gpt_filt, ff, 10, "meco_ffix_gpt")
m_small_ffix, sd_small_ffix = test_model_Ridge(small_filt, ff, 10, "meco_ffix_small")
m_med_ffix, sd_med_ffix     = test_model_Ridge(med_filt, ff, 10, "meco_ffix_med")
m_larg_ffix, sd_larg_ffix   = test_model_Ridge(large_filt, ff, 10, "meco_ffix_large")
m_xl_ffix, sd_xl_ffix       = test_model_Ridge(xl_filt, ff, 10, "meco_ffix_xl")
print(m_gpt_ffix, m_small_ffix, m_med_ffix, m_larg_ffix)

m_gpt_totfix, sd_gpt_totfix     = test_model_Ridge(gpt_filt, tf, 10, "meco_totfix_gpt")
m_small_totfix, sd_small_totfix = test_model_Ridge(small_filt, tf, 10, "meco_totfix_small")
m_med_totfix, sd_med_totfix     = test_model_Ridge(med_filt, tf, 10, "meco_totfix_med")
m_larg_totfix, sd_larg_totfix   = test_model_Ridge(large_filt, tf, 10, "meco_totfix_large")
m_xl_totfix, sd_xl_totfix       = test_model_Ridge(xl_filt, tf, 10, "meco_totfix_xl")
print(m_gpt_totfix, m_small_totfix, m_med_totfix, m_larg_totfix)

baseline_gpt   = np.array(list(zip(s_gpt_meco, freq_meco, len_meco)))
baseline_small = np.array(list(zip(s_small_meco, freq_meco, len_meco)))
baseline_med   = np.array(list(zip(s_med_meco, freq_meco, len_meco)))
baseline_large = np.array(list(zip(s_large_meco, freq_meco, len_meco)))
baseline_xl    = np.array(list(zip(s_xl_meco, freq_meco, len_meco)))

m_gpt_ffix_b, sd_gpt_ffix_b     = test_model_Ridge(baseline_gpt, ff, 10, "meco_ffix_gpt_base")
m_small_ffix_b, sd_small_ffix_b = test_model_Ridge(baseline_small, ff, 10, "meco_ffix_small_base")
m_med_ffix_b, sd_med_ffix_b     = test_model_Ridge(baseline_med, ff, 10, "meco_ffix_med_base")
m_larg_ffix_b, sd_larg_ffix_b   = test_model_Ridge(baseline_large, ff, 10, "meco_ffix_large_base")
m_xl_ffix_b, sd_xl_ffix_b       = test_model_Ridge(baseline_xl, ff, 10, "meco_ffix_xl_base")
print(m_gpt_ffix_b, m_small_ffix_b, m_med_ffix_b, m_larg_ffix_b, m_xl_ffix_b)

m_gpt_totfix_b, sd_gpt_totfix_b     = test_model_Ridge(baseline_gpt, tf, 10, "meco_totfix_gpt_base")
m_small_totfix_b, sd_small_totfix_b = test_model_Ridge(baseline_small, tf, 10, "meco_totfix_small_base")
m_med_totfix_b, sd_med_totfix_b     = test_model_Ridge(baseline_med, tf, 10, "meco_totfix_med_base")
m_larg_totfix_b, sd_larg_totfix_b   = test_model_Ridge(baseline_large, tf, 10, "meco_totfix_large_base")
m_xl_totfix_b, sd_xl_totfix_b       = test_model_Ridge(baseline_xl, tf, 10, "meco_totfix_xl_base")
print(m_gpt_totfix_b, m_small_totfix_b, m_med_totfix_b, m_larg_totfix_b, m_xl_totfix_b)

# decompose_sfl
decompose_sfl(baseline_gpt, ff, "meco_ffix_gpt_base")
decompose_sfl(baseline_small, ff, "meco_ffix_small_base")
decompose_sfl(baseline_med, ff, "meco_ffix_med_base")
decompose_sfl(baseline_large, ff, "meco_ffix_large_base")
decompose_sfl(baseline_xl, ff, "meco_ffix_xl_base")

decompose_sfl(baseline_gpt, tf, "meco_totfix_gpt_base")
decompose_sfl(baseline_small, tf, "meco_totfix_small_base")
decompose_sfl(baseline_med, tf, "meco_totfix_med_base")
decompose_sfl(baseline_large, tf, "meco_totfix_large_base")
decompose_sfl(baseline_xl, tf, "meco_totfix_xl_base")

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_ffix_all, sd_gpt_ffix_all     = test_model_Ridge(gpt_all, ff, 10, "meco_ffix_gpt_all")
m_small_ffix_all, sd_small_ffix_all = test_model_Ridge(gpt2_small_all, ff, 10, "meco_ffix_small_all")
m_med_ffix_all, sd_med_ffix_all     = test_model_Ridge(gpt2_med_all, ff, 10, "meco_ffix_med_all")
m_larg_ffix_all, sd_larg_ffix_all   = test_model_Ridge(gpt2_large_all, ff, 10, "meco_ffix_large_all")
m_xl_ffix_all, sd_xl_ffix_all       = test_model_Ridge(gpt2_xl_all, ff, 10, "meco_ffix_xl_all")
print(m_gpt_ffix_all, m_small_ffix_all, m_med_ffix_all, m_larg_ffix_all, m_xl_ffix_all)

m_gpt_totfix_all, sd_gpt_totfix_all     = test_model_Ridge(gpt_all, tf, 10, "meco_totfix_gpt_all")
m_small_totfix_all, sd_small_totfix_all = test_model_Ridge(gpt2_small_all, tf, 10, "meco_totfix_small_all")
m_med_totfix_all, sd_med_totfix_all     = test_model_Ridge(gpt2_med_all, tf, 10, "meco_totfix_med_all")
m_larg_totfix_all, sd_larg_totfix_all   = test_model_Ridge(gpt2_large_all, tf, 10, "meco_totfix_large_all")
m_xl_totfix_all, sd_xl_totfix_all       = test_model_Ridge(gpt2_xl_all, tf, 10, "meco_totfix_xl_all")
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

thebase_ffix = ["meco_ffix_gpt_base", "meco_ffix_small_base", "meco_ffix_med_base", "meco_ffix_large_base", "meco_xl_base"]
theemb_ffix  = ["meco_ffix_gpt", "meco_ffix_small", "meco_ffix_med", "meco_ffix_large", "meco_ffix_xl"]
all_ffix     = ["meco_ffix_gpt_all", "meco_ffix_small_all", "meco_ffix_med_all", "meco_ffix_large_all", "meco_ffix_xl_all"]

p_ffix = []
for base, emb, all_ in zip(thebase_ffix, theemb_ffix, all_ffix):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_ffix.append([p_base_emb, p_all_base, p_all_emb])
p_ffix = pd.DataFrame(p_ffix, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_ffix.columns:
    df_results["p_ffix_"+df_column] = list(p_ffix[df_column])
    
thebase_totfix = ["meco_totfix_gpt_base", "meco_totfix_small_base", "meco_totfix_med_base", "meco_totfix_large_base", "meco_xl_base"]
theemb_totfix  = ["meco_totfix_gpt", "meco_totfix_small", "meco_totfix_med", "meco_totfix_large", "meco_totfix_xl"]
all_totfix     = ["meco_totfix_gpt_all", "meco_totfix_small_all", "meco_totfix_med_all", "meco_totfix_large_all", "meco_totfix_xl_all"]

p_totfix = []
for base, emb, all_ in zip(thebase_totfix, theemb_totfix, all_totfix):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_totfix.append([p_base_emb, p_all_base, p_all_emb])
p_totfix = pd.DataFrame(p_totfix, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_totfix.columns:
    df_results["p_totfix_"+df_column] = list(p_totfix[df_column])
    
df_results.to_csv("results_meco_ff.csv", index=False)
