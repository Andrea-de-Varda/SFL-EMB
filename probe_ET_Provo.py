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

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, decompose_sfl, probe

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

import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

#######
# GPT #
#######

# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt_provo.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "corpora/provo_sentences.txt",
        dirName+'/activations_gpt_provo.json',
        aggregation="average"
    )

################
# GPT-2 FAMILY #
################

# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small_provo.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "corpora/provo_sentences.txt",
        dirName+'/activations_gpt2_small_provo.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium_provo.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "corpora/provo_sentences.txt",
        dirName+'/activations_gpt2_medium_provo.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large_provo.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "corpora/provo_sentences.txt",
        dirName+'/activations_gpt2_large_provo.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl_provo.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "corpora/provo_sentences.txt",
        dirName+'/activations_gpt2_xl_provo.json',
        aggregation="average"
    )

##########################################################

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

provo_sent = [re.sub("\n", "", s).split() for s in open("corpora/provo_sentences.txt").readlines()]

# GPT
model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt_provo = [s for sent in tqdm(provo_sent) for s in get_surprisal_tokens(sent, sep = "</w>")]

# GPT2 SMALL (124M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2")
s_small_provo = [s for sent in tqdm(provo_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 medium
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-medium")
s_med_provo = [s for sent in tqdm(provo_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 large
model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-large")
s_large_provo = [s for sent in tqdm(provo_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 xl
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl_provo = [s for sent in tqdm(provo_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

df = pd.DataFrame({"s_GPT":s_gpt_provo,
                   "s_small":s_small_provo,
                   "s_med":s_med_provo,
                   "s_large":s_large_provo,
                   "s_xl":s_xl_provo,
                   "freq":provo_freq,
                   "len":provo_len})
#df.to_csv("provo_SFL.csv")
df = pd.read_csv("provo_SFL.csv")
s_gpt_provo = np.array(df["s_GPT"])
s_small_provo = np.array(df["s_small"])
s_med_provo = np.array(df["s_med"])
s_large_provo = np.array(df["s_large"])
s_xl_provo = np.array(df["s_xl"])
provo_freq = np.array(df["freq"])
provo_len = np.array(df["len"])

# loading activations
act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt_provo.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small_provo.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium_provo.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large_provo.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl_provo.json', 1600, is_brnn=False)

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
m_gpt_fpass, sd_gpt_fpass     = test_model_Ridge(gpt_filt, fp, 10, "provo_gpt")
m_small_fpass, sd_small_fpass = test_model_Ridge(small_filt, fp, 10, "provo_small")
m_med_fpass, sd_med_fpass     = test_model_Ridge(med_filt, fp, 10, "provo_med")
m_larg_fpass, sd_larg_fpass   = test_model_Ridge(large_filt, fp, 10, "provo_large")
m_xl_fpass, sd_xl_fpass       = test_model_Ridge(xl_filt, fp, 10, "provo_xl")
print(m_gpt_fpass, m_small_fpass, m_med_fpass, m_larg_fpass)

# first-pass, baseline only
baseline_gpt   = np.array(list(zip(s_gpt_provo, provo_freq, provo_len)))
baseline_small = np.array(list(zip(s_small_provo, provo_freq, provo_len)))
baseline_med   = np.array(list(zip(s_med_provo, provo_freq, provo_len)))
baseline_large = np.array(list(zip(s_large_provo, provo_freq, provo_len)))
baseline_xl    = np.array(list(zip(s_xl_provo, provo_freq, provo_len)))

m_gpt_fpass_b, sd_gpt_fpass_b     = test_model_Ridge(baseline_gpt, fp, 10, "provo_gpt_base")
m_small_fpass_b, sd_small_fpass_b = test_model_Ridge(baseline_small, fp, 10, "provo_small_base")
m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(baseline_med, fp, 10, "provo_med_base")
m_larg_fpass_b, sd_larg_fpass_b   = test_model_Ridge(baseline_large, fp, 10, "provo_large_base")
m_xl_fpass_b, sd_xl_fpass_b       = test_model_Ridge(baseline_xl, fp, 10, "provo_xl_base")
print(m_gpt_fpass_b, m_small_fpass_b, m_med_fpass_b, m_larg_fpass_b, m_xl_fpass_b)

# decompose_sfl
decompose_sfl(baseline_gpt, fp, "provo_gpt_base")
decompose_sfl(baseline_small, fp, "provo_small_base")
decompose_sfl(baseline_med, fp, "provo_med_base")
decompose_sfl(baseline_large, fp, "provo_large_base")
decompose_sfl(baseline_xl, fp, "provo_xl_base")

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_fpass_all, sd_gpt_fpass_all     = test_model_Ridge(gpt_all, fp, 10, "provo_gpt_all")
m_small_fpass_all, sd_small_fpass_all = test_model_Ridge(gpt2_small_all, fp, 10, "provo_small_all")
m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(gpt2_med_all, fp, 10, "provo_med_all")
m_larg_fpass_all, sd_larg_fpass_all   = test_model_Ridge(gpt2_large_all, fp, 10, "provo_large_all")
m_xl_fpass_all, sd_xl_fpass_all       = test_model_Ridge(gpt2_xl_all, fp, 10, "provo_xl_all")
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

########################
# Significance testing #
########################

thebase_fpass = ["provo_gpt_base", "provo_small_base", "provo_med_base", "provo_large_base", "provo_xl_base"]
theemb_fpass  = ["provo_gpt", "provo_small", "provo_med", "provo_large", "provo_xl"]
all_fpass     = ["provo_gpt_all", "provo_small_all", "provo_med_all", "provo_large_all", "provo_xl_all"]

p_fpass = []
for base, emb, all_ in zip(thebase_fpass, theemb_fpass, all_fpass):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_fpass.append([p_base_emb, p_all_base, p_all_emb])
p_fpass = pd.DataFrame(p_fpass, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_fpass.columns:
    df_results["p_fpass_"+df_column] = list(p_fpass[df_column])

# df_results.to_csv("results_provo.csv")

#################################
# FOLLOW-UP: SEMANTIC VARIABLES #
#################################

from utils import plot_sem, get_sem, plot_weights_sem

sem = get_sem(provo_word)
# Missing affective = 69.99
# Missing perceptual = 17.68

print(sem.shape)

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
gpt_all_sem   = np.concatenate((gpt_all, sem), axis=1)
small_all_sem = np.concatenate((gpt2_small_all, sem), axis=1)
med_all_sem   = np.concatenate((gpt2_med_all, sem), axis=1)
large_all_sem = np.concatenate((gpt2_large_all, sem), axis=1)
xl_all_sem    = np.concatenate((gpt2_xl_all, sem), axis=1)

# model testing

###############################
# first-pass, activation only #
###############################

# with sem
m_gpt_fpass_sem, sd_gpt_fpass_sem     = test_model_Ridge(gpt_sem, fp, 10, "provo_gpt_sem")
m_small_fpass_sem, sd_small_fpass_sem = test_model_Ridge(small_sem, fp, 10, "provo_small_sem")
m_med_fpass_sem, sd_med_fpass_sem     = test_model_Ridge(med_sem, fp, 10, "provo_med_sem")
m_larg_fpass_sem, sd_larg_fpass_sem   = test_model_Ridge(large_sem, fp, 10, "provo_large_sem")
m_xl_fpass_sem, sd_xl_fpass_sem       = test_model_Ridge(xl_sem, fp, 10, "provo_xl_sem")
print(m_gpt_fpass_sem, m_small_fpass_sem, m_med_fpass_sem, m_larg_fpass_sem)

########################
# first-pass, baseline #
########################

# with sem
m_gpt_fpass_b_sem, sd_gpt_fpass_b_sem     = test_model_Ridge(gpt_base_sem, fp, 10, "provo_gpt_base_sem")
m_small_fpass_b_sem, sd_small_fpass_b_sem = test_model_Ridge(small_base_sem, fp, 10, "provo_small_base_sem")
m_med_fpass_b_sem, sd_med_fpass_b_sem     = test_model_Ridge(med_base_sem, fp, 10, "provo_med_base_sem")
m_larg_fpass_b_sem, sd_larg_fpass_b_sem   = test_model_Ridge(large_base_sem, fp, 10, "provo_large_base_sem")
m_xl_fpass_b_sem, sd_xl_fpass_b_sem       = test_model_Ridge(xl_base_sem, fp, 10, "provo_xl_base_sem")
print(m_gpt_fpass_b_sem, m_small_fpass_b_sem, m_med_fpass_b_sem, m_larg_fpass_b_sem, m_xl_fpass_b_sem)

###################
# first-pass, all #
###################

# with sem
m_gpt_fpass_all_sem, sd_gpt_fpass_all_sem     = test_model_Ridge(gpt_all_sem, fp, 10, "provo_gpt_all_sem")
m_small_fpass_all_sem, sd_small_fpass_all_sem = test_model_Ridge(small_all_sem, fp, 10, "provo_small_all_sem")
m_med_fpass_all_sem, sd_med_fpass_all_sem     = test_model_Ridge(med_all_sem, fp, 10, "provo_med_all_sem")
m_larg_fpass_all_sem, sd_larg_fpass_all_sem   = test_model_Ridge(large_all_sem, fp, 10, "provo_large_all_sem")
m_xl_fpass_all_sem, sd_xl_fpass_all_sem       = test_model_Ridge(xl_all_sem, fp, 10, "provo_xl_all_sem")
print(m_gpt_fpass_all_sem, m_small_fpass_all_sem, m_med_fpass_all_sem, m_larg_fpass_all_sem, m_xl_fpass_all_sem)

df_results_sem = pd.DataFrame({
    "base_fpass_sem":[m_gpt_fpass_b_sem, m_small_fpass_b_sem, m_med_fpass_b_sem, m_larg_fpass_b_sem, m_xl_fpass_b_sem],
    "base_fpass_sem_sd": [sd_gpt_fpass_b_sem, sd_small_fpass_b_sem, sd_med_fpass_b_sem, sd_larg_fpass_b_sem, sd_xl_fpass_b_sem],
    "act_fpass_sem":[m_gpt_fpass_sem, m_small_fpass_sem, m_med_fpass_sem, m_larg_fpass_sem, m_xl_fpass_sem],
    "act_fpass_sem_sd": [sd_gpt_fpass_sem, sd_small_fpass_sem, sd_med_fpass_sem, sd_larg_fpass_sem, sd_xl_fpass_sem],
    "all_fpass_sem":[m_gpt_fpass_all_sem, m_small_fpass_all_sem, m_med_fpass_all_sem, m_larg_fpass_all_sem, m_xl_fpass_all_sem],
    "all_fpass_sem_sd":[sd_gpt_fpass_all_sem, sd_small_fpass_all_sem, sd_med_fpass_all_sem, sd_larg_fpass_all_sem, sd_xl_fpass_all_sem]
    })
df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

df_results = pd.read_csv("results_provo.csv")
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

merged = pd.concat([df_results, df_results_sem], axis=1)
# merged.to_csv("SEM/merged_provo.csv")

plot_sem(merged, "act_fpass", "base_fpass", "all_fpass", "act_fpass_sem", "base_fpass_sem", "all_fpass_sem", title="Provo", subtract=True)



########################
# for last probing exp #
########################

probe(med_filt, fp, 10, "provo_med_probing")
# print(m_med_fpass)
# m_small_fpass, sd_small_fpass     = test_model_Ridge(zscore(small_filt), zscore(fp), 10, "provo_small_probing")
# print(m_small_fpass)


# from joblib import load
# pca = load('probing_LMs/pca.pkl')
# med_filt_pca = pca.transform(med_filt)

# # first-pass, activation only
# m_med_fpass, sd_med_fpass = test_model_Ridge(med_filt_pca, fp, 10, "provo_med_probing_pca")


##########################
# end of weight analysis #
##########################
# save med_filt, y
dataname = "provo"
tosave = [med_filt, fp]
with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:
    pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################
# OTHER FIXATION MEASUREMENTS #
###############################

##########
# TOTFIX #
##########
   
# tot fix, activation only
m_gpt_totfix, sd_gpt_totfix     = test_model_Ridge(gpt_filt, tf, 10, "provo_tf_gpt")
m_small_totfix, sd_small_totfix = test_model_Ridge(small_filt, tf, 10, "provo_tf_small")
m_med_totfix, sd_med_totfix     = test_model_Ridge(med_filt, tf, 10, "provo_tf_med")
m_larg_totfix, sd_larg_totfix   = test_model_Ridge(large_filt, tf, 10, "provo_tf_large")
m_xl_totfix, sd_xl_totfix       = test_model_Ridge(xl_filt, tf, 10, "provo_tf_xl")
print(m_gpt_totfix, m_small_totfix, m_med_totfix, m_larg_totfix)

# tot fix, baseline only
baseline_gpt   = np.array(list(zip(s_gpt_provo, provo_freq, provo_len)))
baseline_small = np.array(list(zip(s_small_provo, provo_freq, provo_len)))
baseline_med   = np.array(list(zip(s_med_provo, provo_freq, provo_len)))
baseline_large = np.array(list(zip(s_large_provo, provo_freq, provo_len)))
baseline_xl    = np.array(list(zip(s_xl_provo, provo_freq, provo_len)))

m_gpt_totfix_b, sd_gpt_totfix_b     = test_model_Ridge(baseline_gpt, tf, 10, "provo_tf_gpt_base")
m_small_totfix_b, sd_small_totfix_b = test_model_Ridge(baseline_small, tf, 10, "provo_tf_small_base")
m_med_totfix_b, sd_med_totfix_b     = test_model_Ridge(baseline_med, tf, 10, "provo_tf_med_base")
m_larg_totfix_b, sd_larg_totfix_b   = test_model_Ridge(baseline_large, tf, 10, "provo_tf_large_base")
m_xl_totfix_b, sd_xl_totfix_b       = test_model_Ridge(baseline_xl, tf, 10, "provo_tf_xl_base")
print(m_gpt_totfix_b, m_small_totfix_b, m_med_totfix_b, m_larg_totfix_b, m_xl_totfix_b)

# decompose_sfl
decompose_sfl(baseline_gpt, tf, "provo_tf_gpt_base")
decompose_sfl(baseline_small, tf, "provo_tf_small_base")
decompose_sfl(baseline_med, tf, "provo_tf_med_base")
decompose_sfl(baseline_large, tf, "provo_tf_large_base")
decompose_sfl(baseline_xl, tf, "provo_tf_xl_base")

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_totfix_all, sd_gpt_totfix_all     = test_model_Ridge(gpt_all, tf, 10, "provo_tf_gpt_all")
m_small_totfix_all, sd_small_totfix_all = test_model_Ridge(gpt2_small_all, tf, 10, "provo_tf_small_all")
m_med_totfix_all, sd_med_totfix_all     = test_model_Ridge(gpt2_med_all, tf, 10, "provo_tf_med_all")
m_larg_totfix_all, sd_larg_totfix_all   = test_model_Ridge(gpt2_large_all, tf, 10, "provo_tf_large_all")
m_xl_totfix_all, sd_xl_totfix_all       = test_model_Ridge(gpt2_xl_all, tf, 10, "provo_tf_xl_all")
print(m_gpt_totfix_all, m_small_totfix_all, m_med_totfix_all, m_larg_totfix_all, m_xl_totfix_all)

df_results = pd.DataFrame({
    "base_totfix":[m_gpt_totfix_b, m_small_totfix_b, m_med_totfix_b, m_larg_totfix_b, m_xl_totfix_b],
    "base_totfix_sd": [sd_gpt_totfix_b, sd_small_totfix_b, sd_med_totfix_b, sd_larg_totfix_b, sd_xl_totfix_b],
    "act_totfix":[m_gpt_totfix, m_small_totfix, m_med_totfix, m_larg_totfix, m_xl_totfix],
    "act_totfix_sd": [sd_gpt_totfix, sd_small_totfix, sd_med_totfix, sd_larg_totfix, sd_xl_totfix],
    "all_totfix":[m_gpt_totfix_all, m_small_totfix_all, m_med_totfix_all, m_larg_totfix_all, m_xl_totfix_all],
    "all_totfix_sd":[sd_gpt_totfix_all, sd_small_totfix_all, sd_med_totfix_all, sd_larg_totfix_all, sd_xl_totfix_all]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

thebase_totfix = ["provo_tf_gpt_base", "provo_tf_small_base", "provo_tf_med_base", "provo_tf_large_base", "provo_tf_xl_base"]
theemb_totfix  = ["provo_tf_gpt", "provo_tf_small", "provo_tf_med", "provo_tf_large", "provo_tf_xl"]
all_totfix     = ["provo_tf_gpt_all", "provo_tf_small_all", "provo_tf_med_all", "provo_tf_large_all", "provo_tf_xl_all"]

p_totfix = []
for base, emb, all_ in zip(thebase_totfix, theemb_totfix, all_totfix):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_totfix.append([p_base_emb, p_all_base, p_all_emb])
p_totfix = pd.DataFrame(p_totfix, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_totfix.columns:
    df_results["p_totfix_"+df_column] = list(p_totfix[df_column])

df_results.to_csv("results_provo_tf.csv")

########
# FFIX #
########

# ffix, activation only
m_gpt_ffix, sd_gpt_ffix     = test_model_Ridge(gpt_filt, ff, 10, "provo_ff_gpt")
m_small_ffix, sd_small_ffix = test_model_Ridge(small_filt, ff, 10, "provo_ff_small")
m_med_ffix, sd_med_ffix     = test_model_Ridge(med_filt, ff, 10, "provo_ff_med")
m_larg_ffix, sd_larg_ffix   = test_model_Ridge(large_filt, ff, 10, "provo_ff_large")
m_xl_ffix, sd_xl_ffix       = test_model_Ridge(xl_filt, ff, 10, "provo_ff_xl")
print(m_gpt_ffix, m_small_ffix, m_med_ffix, m_larg_ffix)

# ffix, baseline only
baseline_gpt   = np.array(list(zip(s_gpt_provo, provo_freq, provo_len)))
baseline_small = np.array(list(zip(s_small_provo, provo_freq, provo_len)))
baseline_med   = np.array(list(zip(s_med_provo, provo_freq, provo_len)))
baseline_large = np.array(list(zip(s_large_provo, provo_freq, provo_len)))
baseline_xl    = np.array(list(zip(s_xl_provo, provo_freq, provo_len)))

m_gpt_ffix_b, sd_gpt_ffix_b     = test_model_Ridge(baseline_gpt, ff, 10, "provo_ff_gpt_base")
m_small_ffix_b, sd_small_ffix_b = test_model_Ridge(baseline_small, ff, 10, "provo_ff_small_base")
m_med_ffix_b, sd_med_ffix_b     = test_model_Ridge(baseline_med, ff, 10, "provo_ff_med_base")
m_larg_ffix_b, sd_larg_ffix_b   = test_model_Ridge(baseline_large, ff, 10, "provo_ff_large_base")
m_xl_ffix_b, sd_xl_ffix_b       = test_model_Ridge(baseline_xl, ff, 10, "provo_ff_xl_base")
print(m_gpt_ffix_b, m_small_ffix_b, m_med_ffix_b, m_larg_ffix_b, m_xl_ffix_b)

# decompose_sfl
decompose_sfl(baseline_gpt, ff, "provo_ff_gpt_base")
decompose_sfl(baseline_small, ff, "provo_ff_small_base")
decompose_sfl(baseline_med, ff, "provo_ff_med_base")
decompose_sfl(baseline_large, ff, "provo_ff_large_base")
decompose_sfl(baseline_xl, ff, "provo_ff_xl_base")

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_ffix_all, sd_gpt_ffix_all     = test_model_Ridge(gpt_all, ff, 10, "provo_ff_gpt_all")
m_small_ffix_all, sd_small_ffix_all = test_model_Ridge(gpt2_small_all, ff, 10, "provo_ff_small_all")
m_med_ffix_all, sd_med_ffix_all     = test_model_Ridge(gpt2_med_all, ff, 10, "provo_ff_med_all")
m_larg_ffix_all, sd_larg_ffix_all   = test_model_Ridge(gpt2_large_all, ff, 10, "provo_ff_large_all")
m_xl_ffix_all, sd_xl_ffix_all       = test_model_Ridge(gpt2_xl_all, ff, 10, "provo_ff_xl_all")
print(m_gpt_ffix_all, m_small_ffix_all, m_med_ffix_all, m_larg_ffix_all, m_xl_ffix_all)

df_results = pd.DataFrame({
    "base_ffix":[m_gpt_ffix_b, m_small_ffix_b, m_med_ffix_b, m_larg_ffix_b, m_xl_ffix_b],
    "base_ffix_sd": [sd_gpt_ffix_b, sd_small_ffix_b, sd_med_ffix_b, sd_larg_ffix_b, sd_xl_ffix_b],
    "act_ffix":[m_gpt_ffix, m_small_ffix, m_med_ffix, m_larg_ffix, m_xl_ffix],
    "act_ffix_sd": [sd_gpt_ffix, sd_small_ffix, sd_med_ffix, sd_larg_ffix, sd_xl_ffix],
    "all_ffix":[m_gpt_ffix_all, m_small_ffix_all, m_med_ffix_all, m_larg_ffix_all, m_xl_ffix_all],
    "all_ffix_sd":[sd_gpt_ffix_all, sd_small_ffix_all, sd_med_ffix_all, sd_larg_ffix_all, sd_xl_ffix_all]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

thebase_ffix = ["provo_ff_gpt_base", "provo_ff_small_base", "provo_ff_med_base", "provo_ff_large_base", "provo_ff_xl_base"]
theemb_ffix  = ["provo_ff_gpt", "provo_ff_small", "provo_ff_med", "provo_ff_large", "provo_ff_xl"]
all_ffix     = ["provo_ff_gpt_all", "provo_ff_small_all", "provo_ff_med_all", "provo_ff_large_all", "provo_ff_xl_all"]

p_ffix = []
for base, emb, all_ in zip(thebase_ffix, theemb_ffix, all_ffix):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_ffix.append([p_base_emb, p_all_base, p_all_emb])
p_ffix = pd.DataFrame(p_ffix, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_ffix.columns:
    df_results["p_ffix_"+df_column] = list(p_ffix[df_column])

df_results.to_csv("results_provo_ff.csv")

