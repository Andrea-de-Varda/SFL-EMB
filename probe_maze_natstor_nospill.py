import neurox
import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from glmnet import ElasticNet
import re
from scipy.special import softmax
import torch.nn.functional as F
import scipy.io
from researchpy import corr_pair
from transformers import AutoModelForCausalLM, AutoTokenizer, TransfoXLLMHeadModel
import torch
from scipy.special import softmax
import matplotlib.patches as mpatches
import pyreadr
import pickle

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge, base_plot, significance_testing, decompose_sfl, probe

#####################
# reading maze data #
#####################

maze = pyreadr.read_r('/home/dev/Documents/Datasets/NaturalStories/natstor_maze.rds')
df_maze = maze[None]

maze = df_maze[df_maze["native"].isin(['ENG', 'ENGLISH', 'English','english'])]

sent_n_dict = {sent : idx for idx, sent in enumerate(set(maze["sentence"]))}
sent_n_d_rev = {value : key for key, value in sent_n_dict.items()}
maze["sent_n"] = maze.sentence.map(sent_n_dict)
maze["unique"] = maze["sent_n"].astype(str)+"-"+maze["word_num"].astype(str)

# with open('corpora/maze_sentences.txt', 'w') as f:
#     for n in range(0, 481):
#     #for line in set(maze["sentence"]):
#         line = sent_n_d_rev[n]
#         f.write(f"{line}\n")

maze_sent = [re.sub("\n", "", s).split() for s in open("corpora/maze_sentences.txt").readlines()]
maze_sent_dict_recover = {" ".join(l):idx for idx, l in enumerate(maze_sent)}
maze["sent_n"] = maze.sentence.map(maze_sent_dict_recover)

part_dict = {}
for subj in set(maze["subject"]):
    temp = maze[maze["subject"] == subj]
    corr = len(temp[temp["correct"] == "yes"])/len(temp) > .8
    part_dict[subj] = corr
    
error = []
previous_idx = 0
previous_cor = True
for index, row in tqdm(maze.iterrows()):
    current_idx = row["word_num"]
    current_cor = row["correct"] == "yes"
    if current_idx - previous_idx == 1:
        if current_cor and previous_cor:
            error.append(0)
        else:
            error.append(1)
    else:
        if current_cor:
            error.append(0)
        else:
            error.append(1)
    previous_idx = current_idx
    previous_cor = current_cor
maze["error"] = error

maze["is_attentive"] = maze["subject"].map(part_dict)
maze = maze[(maze["type"] != "practice") & (maze["rt"] < 5000) & (maze["rt"] > 100) & (maze["is_attentive"] == True) & (maze["word_num"] !=0) & (maze["error"] == 0)]

maze = maze.groupby(["sent_n", "word_num", "word", "unique"], as_index=False, sort=False).agg({'rt':"mean", 'word_num':"max", 'sent_n':"max", "word":"max", "unique":"max"})

# getting relevant psychometric data

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

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act_eye(act_list, layer, dim):
    act_list = [act_list[i] for i in list(set(maze.sent_n))]
    out = [get_layer(s, layer, dim) for sent in act_list for s in sent]
    return np.array(out)

def filt_act(act, filt):
    out = []
    for activ, thefilt in zip(act, filt):
        if thefilt == 0:
            out.append(activ)
    return np.array(out)

maze_sent_len = {idx : length for idx, length in enumerate([len(re.sub("\n", "", s).split()) for s in open("corpora/maze_sentences.txt").readlines()])}

filt = []
rt = []
length = []
freq = []
theact_gpt_cleaned = []
for sent_n in tqdm(list(set(maze.sent_n))):
    temp = maze[maze.sent_n == sent_n]
    for n in range(maze_sent_len[sent_n]):
        therow = temp[temp.word_num == n]
        if therow.empty:
            filt.append(1)
        else:
            if n == 0:
                filt.append(1)
            else:
                rt.append(float(therow.iloc[0, 0]))
                filt.append(0)
                length.append(get_l(therow.iloc[0, 3]))
                freq.append(get_f(therow.iloc[0, 3]))
                theact_gpt_cleaned.append(get_layer(act_gpt[sent_n][n], 12, 768))
rt = np.array(rt)
length = np.array(length)
freq = np.array(freq)

##############################
# EXTRACTING REPRESENTATIONS #
##############################

import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

###################
# NATURAL STORIES #
###################

# GPT
# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt_maze.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "corpora/maze_sentences.txt",
        dirName+'/activations_gpt_maze.json',
        aggregation="average"
    )

# GPT-2 FAMILY 
# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small_maze.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "corpora/maze_sentences.txt",
        dirName+'/activations_gpt2_small_maze.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium_maze.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "corpora/maze_sentences.txt",
        dirName+'/activations_gpt2_medium_maze.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large_maze.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "corpora/maze_sentences.txt",
        dirName+'/activations_gpt2_large_maze.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl_maze.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "corpora/maze_sentences.txt",
        dirName+'/activations_gpt2_xl_maze.json',
        aggregation="average"
    )

# loading activations
act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt_maze.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small_maze.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium_maze.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large_maze.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl_maze.json', 1600, is_brnn=False)

gpt_filt   = get_act_eye(act_gpt, 12, 768)
small_filt = get_act_eye(act_gpt2_small, 12, 768)
med_filt   = get_act_eye(act_gpt2_medium, 24, 1024)
large_filt = get_act_eye(act_gpt2_large, 36, 1280)
xl_filt    = get_act_eye(act_gpt2_xl, 48, 1600)

gpt_filt   = filt_act(gpt_filt, filt)
small_filt = filt_act(small_filt, filt)
med_filt   = filt_act(med_filt, filt)
large_filt = filt_act(large_filt, filt)
xl_filt    = filt_act(xl_filt, filt)

maze_sent = [re.sub("\n", "", s).split() for s in open("corpora/maze_sentences.txt").readlines()]

# some checks
sum([len(act_gpt[i]) for i in set(maze.sent_n)])
sum([len(maze_sent[i]) for i in set(maze.sent_n)])

for a, b in zip(maze_sent, act_gpt):
    print(len(a) == len(b))

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

# GPT
model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt_maze = [get_surprisal_tokens(sent, sep = "</w>") for sent in maze_sent]

# GPT2 SMALL (124M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2")
s_small_maze = [get_surprisal_tokens(sent, sep = "Ġ", cased=True) for sent in maze_sent]

# GPT2 medium
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-medium")
s_med_maze = [get_surprisal_tokens(sent, sep = "Ġ", cased=True) for sent in maze_sent]

# GPT2 large
model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-large")
s_large_maze = [get_surprisal_tokens(sent, sep = "Ġ", cased=True) for sent in maze_sent]

# GPT2 xl
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl_maze = [get_surprisal_tokens(sent, sep = "Ġ", cased=True) for sent in maze_sent]

def get_surprisal(surp_list):
    surp_list = [surp_list[i] for i in set(maze.sent_n)]
    out = [s for sent in surp_list for s in [np.nan]+sent]
    return np.array(out)

def clean_surprisal(surp_list):
    surp_list = get_surprisal(surp_list) # excluding sentences with all NaN
    out = filt_act(surp_list, filt)
    return out

s_gpt = clean_surprisal(s_gpt_maze)
s_small = clean_surprisal(s_small_maze)
s_med = clean_surprisal(s_med_maze)
s_large = clean_surprisal(s_large_maze)
s_xl = clean_surprisal(s_xl_maze)

df = pd.DataFrame({"s_GPT":s_gpt,
                    "s_small":s_small,
                    "s_med":s_med,
                    "s_large":s_large,
                    "s_xl":s_xl,
                    "freq":freq,
                    "len":length})

#df.to_csv("maze_SFL_nospill.csv")
df = pd.read_csv("maze_SFL_nospill.csv")
s_gpt_maze = np.array(df["s_GPT"])
s_small_maze = np.array(df["s_small"])
s_med_maze = np.array(df["s_med"])
s_large_maze = np.array(df["s_large"])
s_xl_maze = np.array(df["s_xl"])
maze_freq = np.array(df["freq"])
maze_len = np.array(df["len"])


##########################################

###########
# testing #
###########

# first-pass, activation only
m_gpt, sd_gpt     = test_model_Ridge(gpt_filt, rt, 10, "maze_gpt")
m_small, sd_small = test_model_Ridge(small_filt, rt, 10, "maze_small")
m_med, sd_med     = test_model_Ridge(med_filt, rt, 10, "maze_med")
m_larg, sd_larg   = test_model_Ridge(large_filt, rt, 10, "maze_large")
m_xl, sd_xl       = test_model_Ridge(xl_filt, rt, 10, "maze_xl")
print(m_gpt, m_small, m_med, m_larg, m_xl)

# first-pass, baseline only

baseline_gpt   = np.array(df[["s_GPT", "len", "freq"]])
baseline_small = np.array(df[["s_small", "len", "freq"]])
baseline_med   = np.array(df[["s_med", "len", "freq"]])
baseline_large = np.array(df[["s_GPT", "len", "freq"]])
baseline_xl    = np.array(df[["s_xl", "len", "freq"]])

m_gpt_b, sd_gpt_b     = test_model_Ridge(baseline_gpt, rt, 10, "maze_gpt_base")
m_small_b, sd_small_b = test_model_Ridge(baseline_small, rt, 10, "maze_small_base")
m_med_b, sd_med_b     = test_model_Ridge(baseline_med, rt, 10, "maze_med_base")
m_larg_b, sd_larg_b   = test_model_Ridge(baseline_large, rt, 10, "maze_large_base")
m_xl_b, sd_xl_b       = test_model_Ridge(baseline_xl, rt, 10, "maze_xl_base")
print(m_gpt_b, m_small_b, m_med_b, m_larg_b, m_xl_b)

# decompose_sfl
decompose_sfl(baseline_gpt, rt, "maze_gpt_base", idx_s = 0, idx_f = 2, idx_l = 1)
decompose_sfl(baseline_small, rt, "maze_small_base", idx_s = 0, idx_f = 2, idx_l = 1)
decompose_sfl(baseline_med, rt, "maze_med_base", idx_s = 0, idx_f = 2, idx_l = 1)
decompose_sfl(baseline_large, rt, "maze_large_base", idx_s = 0, idx_f = 2, idx_l = 1)
decompose_sfl(baseline_xl, rt, "maze_xl_base", idx_s = 0, idx_f = 2, idx_l = 1)


# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_all, sd_gpt_all     = test_model_Ridge(gpt_all, rt, 10, "maze_gpt_all")
m_small_all, sd_small_all = test_model_Ridge(gpt2_small_all, rt, 10, "maze_small_all")
m_med_all, sd_med_all     = test_model_Ridge(gpt2_med_all, rt, 10, "maze_med_all")
m_larg_all, sd_larg_all   = test_model_Ridge(gpt2_large_all, rt, 10, "maze_large_all")
m_xl_all, sd_xl_all       = test_model_Ridge(gpt2_xl_all, rt, 10, "maze_xl_all")
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

# df_results.to_csv("results_maze_nospill.csv")

base_plot(df_results, "act", "base", "all")

# significance testing

thebase_maze = ["maze_gpt_base", "maze_small_base", "maze_med_base", "maze_large_base", "maze_xl_base"]
theemb_maze  = ["maze_gpt", "maze_small", "maze_med", "maze_large", "maze_xl"]
all_maze     = ["maze_gpt_all", "maze_small_all", "maze_med_all", "maze_large_all", "maze_xl_all"]

p_maze = []
for base, emb, all_ in zip(thebase_maze, theemb_maze, all_maze):
    p_base_emb = significance_testing(emb, base)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_maze.append([p_base_emb, p_all_base, p_all_emb])
p_maze = pd.DataFrame(p_maze, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_maze.columns:
    df_results["p_maze_"+df_column] = list(p_maze[df_column])
    
    
#################################
# FOLLOW-UP: SEMANTIC VARIABLES #
#################################

from utils import plot_sem, get_sem, plot_weights_sem

words = [maze_sent[i] for i in set(maze.sent_n)]
out = [s for sent in words for s in sent]
wordlist = list(filt_act(out, filt))
wordlist = [str(w) for w in wordlist]

sem = get_sem(wordlist)

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

# model testing

###############################
# first-pass, activation only #
###############################

# with sem
m_gpt_sem, sd_gpt_sem     = test_model_Ridge(gpt_sem, rt, 10, "maze_gpt_sem")
m_small_sem, sd_small_sem = test_model_Ridge(small_sem, rt, 10, "maze_small_sem")
m_med_sem, sd_med_sem     = test_model_Ridge(med_sem, rt, 10, "maze_med_sem")
m_larg_sem, sd_larg_sem   = test_model_Ridge(large_sem, rt, 10, "maze_large_sem")
m_xl_sem, sd_xl_sem       = test_model_Ridge(xl_sem, rt, 10, "maze_xl_sem")
print(m_gpt_sem, m_small_sem, m_med_sem, m_larg_sem)

########################
# first-pass, baseline #
########################

# with sem
m_gpt_b_sem, sd_gpt_b_sem     = test_model_Ridge(gpt_base_sem, rt, 10, "maze_gpt_base_sem")
m_small_b_sem, sd_small_b_sem = test_model_Ridge(small_base_sem, rt, 10, "maze_small_base_sem")
m_med_b_sem, sd_med_b_sem     = test_model_Ridge(med_base_sem, rt, 10, "maze_med_base_sem")
m_larg_b_sem, sd_larg_b_sem   = test_model_Ridge(large_base_sem, rt, 10, "maze_large_base_sem")
m_xl_b_sem, sd_xl_b_sem       = test_model_Ridge(xl_base_sem, rt, 10, "maze_xl_base_sem")
print(m_gpt_b_sem, m_small_b_sem, m_med_b_sem, m_larg_b_sem, m_xl_b_sem)

###################
# first-pass, all #
###################

# with sem
m_gpt_all_sem, sd_gpt_all_sem     = test_model_Ridge(gpt_all_sem, rt, 10, "maze_gpt_all_sem")
m_small_all_sem, sd_small_all_sem = test_model_Ridge(small_all_sem, rt, 10, "maze_small_all_sem")
m_med_all_sem, sd_med_all_sem     = test_model_Ridge(med_all_sem, rt, 10, "maze_med_all_sem")
m_larg_all_sem, sd_larg_all_sem   = test_model_Ridge(large_all_sem, rt, 10, "maze_large_all_sem")
m_xl_all_sem, sd_xl_all_sem       = test_model_Ridge(xl_all_sem, rt, 10, "maze_xl_all_sem")
print(m_gpt_all_sem, m_small_all_sem, m_med_all_sem, m_larg_all_sem, m_xl_all_sem)

df_results_sem = pd.DataFrame({
    "base_sem":[m_gpt_b_sem, m_small_b_sem, m_med_b_sem, m_larg_b_sem, m_xl_b_sem],
    "base_sem_sd": [sd_gpt_b_sem, sd_small_b_sem, sd_med_b_sem, sd_larg_b_sem, sd_xl_b_sem],
    "act_sem":[m_gpt_sem, m_small_sem, m_med_sem, m_larg_sem, m_xl_sem],
    "act_sem_sd": [sd_gpt_sem, sd_small_sem, sd_med_sem, sd_larg_sem, sd_xl_sem],
    "all_sem":[m_gpt_all_sem, m_small_all_sem, m_med_all_sem, m_larg_all_sem, m_xl_all_sem],
    "all_sem_sd":[sd_gpt_all_sem, sd_small_all_sem, sd_med_all_sem, sd_larg_all_sem, sd_xl_all_sem]
    })
df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

df_results = pd.read_csv("results_maze_nospill.csv")
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

merged = pd.concat([df_results, df_results_sem], axis=1)
#merged.to_csv("SEM/merged_maze.csv")

df_results = pd.read_csv("results_maze_nospill.csv")

plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", title="maze", subtract=True)


########################
# for last probing exp #
########################

probe(med_filt, rt, 10, "maze_med_probing")

# from joblib import load
# pca = load('probing_LMs/pca.pkl')
# med_filt_pca = pca.transform(med_filt)

# # first-pass, activation only
# m_med, sd_med = test_model_Ridge(med_filt_pca, rt, 10, "maze_med_probing_pca")

##########################
# end of weight analysis #
##########################
dataname = "maze"
tosave = [med_filt, rt]
with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:
    pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)