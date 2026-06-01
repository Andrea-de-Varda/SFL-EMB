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
# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
from utils import test_model_Ridge, base_plot

dirName = str(PROJECT_ROOT)
chdir(dirName)

#####################
# reading maze data #
#####################

maze = pyreadr.read_r(PROJECT_ROOT / "main_data/NaturalStories/natstor_maze.rds")
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
freq = pd.read_excel(PROJECT_ROOT / "main_data/subtlex.xlsx")
f = {row.Word : row.Lg10WF for index, row in freq.iterrows()}
minfreq = min(freq["Lg10WF"])

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

maze_sent_len = {idx : length for idx, length in enumerate([len(re.sub("\n", "", s).split()) for s in open("corpora/maze_sentences.txt").readlines()])}

# filt = []
# rt = []
# length = []
# freq = []
# for sent_n in tqdm(set(maze.sent_n)):
#     temp = maze[maze.sent_n == sent_n]
#     for n in range(maze_sent_len[sent_n]):
#         therow = temp[temp.word_num == n]
#         if therow.empty:
#             filt.append(1)
#         else:
#             rt.append(float(therow.iloc[0, 0]))
#             filt.append(0)
#             length.append(get_l(therow.iloc[0, 3]))
#             freq.append(get_f(therow.iloc[0, 3]))
# rt = np.array(rt)
# length = np.array(length)
# freq = np.array(freq)

out = []
filt = []
for sent in set(maze.sent_n):
    temp = maze[maze.sent_n == sent]
    idxmax = max(temp.word_num)+1
    idxmin = min(temp.word_num)
    filt.extend([1,1,1])
    for idx in range(3, idxmax):
        twobef = temp[temp.word_num == idx-2]
        onebef = temp[temp.word_num == idx-1]
        actual = temp[temp.word_num == idx]
        if actual.empty or twobef.empty or onebef.empty:
            filt.append(1)
        else:
            a = twobef.values.tolist()[0]+onebef.values.tolist()[0]+actual.values.tolist()[0]
            out.append(a)
            filt.append(0)
        
cols = [colname + "-2" for colname in maze.columns.tolist()] + [colname + "-1" for colname in maze.columns.tolist()] + maze.columns.tolist()
df_spill = pd.DataFrame(out, columns = cols)
df_spill["len"] = [get_l(w) for w in df_spill["word"]]
df_spill["len-1"] = [get_l(w) for w in df_spill["word-1"]]
df_spill["len-2"] = [get_l(w) for w in df_spill["word-2"]]
df_spill["freq"] = [get_f(w) for w in df_spill["word"]]
df_spill["freq-1"] = [get_f(w) for w in df_spill["word-1"]]
df_spill["freq-2"] = [get_f(w) for w in df_spill["word-2"]]

rt = np.array(df_spill["rt"])

# some checks 
# print(len(filt))
# print(len(maze))
# print(sum([len(x) for x in act_gpt]))
# len(maze)-sum(filt)
# print(len(filt)-sum(filt))

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

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act_eye(act_list, layer, dim):
    act_list = [act_list[i] for i in set(maze.sent_n)]
    out = []
    for act_sent in act_list:
        for idx, array in enumerate(act_sent):
            out.append(np.concatenate([get_layer(act_sent[idx], layer, dim), get_layer(act_sent[idx-1], layer, dim), get_layer(act_sent[idx-2], layer, dim)]))
    return np.array(out)

def filt_act(act, filt):
    out = []
    for activ, thefilt in zip(act, filt):
        if thefilt == 0:
            out.append(activ)
    return np.array(out)

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

def filter_surprisal(surp, name):
    i, i1, i2 = [],[],[]
    for sent in tqdm(set(maze.sent_n)):
        temp = maze[maze.sent_n == sent]
        idxmax = max(temp.word_num)+1
        idxmin = min(temp.word_num)
        for idx in range(3, idxmax):
            twobef = temp[temp.word_num == idx-2]
            onebef = temp[temp.word_num == idx-1]
            actual = temp[temp.word_num == idx]
            twobef_s = surp[sent][idx-3]
            onebef_s = surp[sent][idx-2]
            actual_s = surp[sent][idx-1]
            if actual.empty or twobef.empty or onebef.empty:
                pass
            else:
                i.append(actual_s)
                i1.append(onebef_s)
                i2.append(twobef_s)
    df_spill["s_"+name] = i
    df_spill["s_"+name+"-1"] = i1
    df_spill["s_"+name+"-2"] = i2

filter_surprisal(s_gpt_maze, "GPT")
filter_surprisal(s_small_maze, "small")
filter_surprisal(s_med_maze, "med")
filter_surprisal(s_large_maze, "large")
filter_surprisal(s_xl_maze, "xl")

#df_spill.to_csv("DF_spill_maze.csv") ###############



# # filtering surp values for words that were absent in the RT data (excluded before)
# filt_2 = []
# for sent_n in range(len(maze_sent)):
#     temp = maze[maze.sent_n == sent_n]
#     for n in range(maze_sent_len[sent_n]):
#         therow = temp[temp.word_num == n]
#         twobef = temp[temp.word_num == n-2]
#         onebef = temp[temp.word_num == n-1]
#         if therow.empty or onebef.empty or twobef.empty: 
#             if n == 0:
#                 pass
#             else:
#                 filt_2.append(1)
#         else:
#             filt_2.append(0)

# def filter_surprisal(surprisal):
#     out = []
#     for s, f in zip(surprisal, filt_2):
#         if f == 0:
#             out.append(s)
#         else:
#             pass
#     return out

# s_gpt_maze1 = filter_surprisal(s_gpt_maze)
# s_small_maze1 = filter_surprisal(s_small_maze)
# s_med_maze1 = filter_surprisal(s_med_maze)
# s_large_maze1 = filter_surprisal(s_large_maze)
# s_xl_maze1 = filter_surprisal(s_xl_maze)

# df = pd.DataFrame({"s_GPT":s_gpt_maze1,
#                    "s_small":s_small_maze1,
#                    "s_med":s_med_maze1,
#                    "s_large":s_large_maze1,
#                    "s_xl":s_xl_maze1,
#                    "freq":freq,
#                    "len":length})

# df.to_csv("maze_SFL.csv")
# df = pd.read_csv("maze_SFL.csv")
# s_gpt_maze = np.array(df["s_GPT"])
# s_small_maze = np.array(df["s_small"])
# s_med_maze = np.array(df["s_med"])
# s_large_maze = np.array(df["s_large"])
# s_xl_maze = np.array(df["s_xl"])
# maze_freq = np.array(df["freq"])
# maze_len = np.array(df["len"])

##########################################


###########
# testing #
###########
    
# first-pass, activation only
m_gpt, sd_gpt     = test_model_Ridge(gpt_filt, rt, 5)
m_small, sd_small = test_model_Ridge(small_filt, rt, 10)
m_med, sd_med     = test_model_Ridge(med_filt, rt, 10)
m_larg, sd_larg   = test_model_Ridge(large_filt, rt, 10)
m_xl, sd_xl       = test_model_Ridge(xl_filt, rt, 10)
print(m_gpt, m_small, m_med, m_larg)

# first-pass, baseline only

baseline_gpt   = np.array(df_spill[["s_GPT", "len", "freq",
                   "s_GPT-1", "len-1", "freq-1",
                   "s_GPT-2", "len-2", "freq-2"]])
baseline_small = np.array(df_spill[["s_small", "len", "freq",
                   "s_small-1", "len-1", "freq-1",
                   "s_small-2", "len-2", "freq-2"]])
baseline_med   = np.array(df_spill[["s_med", "len", "freq",
                   "s_med-1", "len-1", "freq-1",
                   "s_med-2", "len-2", "freq-2"]])
baseline_large = np.array(df_spill[["s_GPT", "len", "freq",
                   "s_large-1", "len-1", "freq-1",
                   "s_large-2", "len-2", "freq-2"]])
baseline_xl    = np.array(df_spill[["s_xl", "len", "freq",
                   "s_xl-1", "len-1", "freq-1",
                   "s_xl-2", "len-2", "freq-2"]])


m_gpt_b, sd_gpt_b     = test_model_Ridge(baseline_gpt, rt, 5)
m_small_b, sd_small_b = test_model_Ridge(baseline_small, rt, 5)
m_med_b, sd_med_b     = test_model_Ridge(baseline_med, rt, 5)
m_larg_b, sd_larg_b   = test_model_Ridge(baseline_large, rt, 5)
m_xl_b, sd_xl_b       = test_model_Ridge(baseline_xl, rt, 5)
print(m_gpt_b, m_small_b, m_med_b, m_larg_b, m_xl_b)

# all
gpt_all        = np.concatenate((baseline_gpt, gpt_filt), axis=1)
gpt2_small_all = np.concatenate((baseline_small, small_filt), axis=1)
gpt2_med_all   = np.concatenate((baseline_med, med_filt), axis=1)
gpt2_large_all = np.concatenate((baseline_large, large_filt), axis=1)
gpt2_xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

m_gpt_all, sd_gpt_all     = test_model_Ridge(gpt_all, rt, 5)
m_small_all, sd_small_all = test_model_Ridge(gpt2_small_all, rt, 5)
m_med_all, sd_med_all     = test_model_Ridge(gpt2_med_all, rt, 5)
m_larg_all, sd_larg_all   = test_model_Ridge(gpt2_large_all, rt, 5)
m_xl_all, sd_xl_all       = test_model_Ridge(gpt2_xl_all, rt, 5)
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

# df_results.to_csv("results_maze_spill.csv")

plt.rcParams.update({
    "text.usetex": False})
fig = plt.figure(figsize = (5/1.2, 3.5/1.2), dpi=300)
plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
plt.minorticks_on()
plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
plt.bar([1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], 
        [df_results.loc["GPT", "act"], df_results.loc["GPT", "base"], df_results.loc["GPT", "all"],
         df_results.loc["GPT-2$_{124M}$", "act"], df_results.loc["GPT-2$_{124M}$", "base"], df_results.loc["GPT-2$_{124M}$", "all"],
         df_results.loc["GPT-2$_{355M}$", "act"], df_results.loc["GPT-2$_{355M}$", "base"], df_results.loc["GPT-2$_{355M}$", "all"],
         df_results.loc["GPT-2$_{774M}$", "act"], df_results.loc["GPT-2$_{774M}$", "base"], df_results.loc["GPT-2$_{774M}$", "all"],
         df_results.loc["GPT-2$_{1.5B}$", "act"], df_results.loc["GPT-2$_{1.5B}$", "base"], df_results.loc["GPT-2$_{1.5B}$", "all"]],
        color = ['lightblue', 'lightgreen', "indianred"]*5,width=0.9,
        yerr=[df_results.loc["GPT", "act_sd"], df_results.loc["GPT", "base_sd"], df_results.loc["GPT", "all_sd"],
         df_results.loc["GPT-2$_{124M}$", "act_sd"], df_results.loc["GPT-2$_{124M}$", "base_sd"], df_results.loc["GPT-2$_{124M}$", "all_sd"],
         df_results.loc["GPT-2$_{355M}$", "act_sd"], df_results.loc["GPT-2$_{355M}$", "base_sd"], df_results.loc["GPT-2$_{355M}$", "all_sd"],
         df_results.loc["GPT-2$_{774M}$", "act_sd"], df_results.loc["GPT-2$_{774M}$", "base_sd"], df_results.loc["GPT-2$_{774M}$", "all_sd"],
         df_results.loc["GPT-2$_{1.5B}$", "act_sd"], df_results.loc["GPT-2$_{1.5B}$", "base_sd"], df_results.loc["GPT-2$_{1.5B}$", "all_sd"]],capsize=2,
        tick_label=["", "GPT", "", "", "GPT-2$_{124M}$", "", "", "GPT-2$_{355M}$","", "",  "GPT-2$_{774M}$", "", "", "GPT-2$_{1.5B}$", ""]
        )
plt.ylabel("r")
plt.xticks(fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
plt.ylim(0.1, 0.9)
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
plt.show()
