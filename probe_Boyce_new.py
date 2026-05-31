import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from glmnet import ElasticNet
import re
from scipy.special import softmax
from sklearn.linear_model import RidgeCV
import torch.nn.functional as F
import scipy.io
from researchpy import corr_pair
from transformers import AutoModelForCausalLM, AutoTokenizer, TransfoXLLMHeadModel
import torch
from scipy.special import softmax
import matplotlib.patches as mpatches
import pickle
import pyreadr

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge, base_plot, significance_testing, decompose_sfl

#####################
# reading maze data #
#####################

def preprocess_maze(path):
    maze = pyreadr.read_r(path)
    df_maze = maze[None]
    
    maze = df_maze[(df_maze["native"] == "yes") & (df_maze["citizen"] == "yes") & (df_maze["resident"] == "yes")]
    maze_no_na = maze[maze['rt'].notnull()]
    maze_cleaned = maze_no_na[maze_no_na['correct'] == 'yes'].drop(columns=['correct'])
    maze_no_0 = maze_cleaned[maze_cleaned['rt'] != 0]
    return maze_no_0
    
def preprocess_spr(path):
    spr = pyreadr.read_r(path)
    df_spr = spr[None]
    
    data_filtered = df_spr[(df_spr['citizen'] == 'yes') & (df_spr['native'] == 'yes') & (df_spr['resident'] == 'yes')]
    data_filtered['word_num'] = data_filtered['word_num'] - 1
    
    data_accurate = data_filtered[data_filtered['accuracy'] > 0.8]
    data_no_na = data_accurate[data_accurate['rt'].notna()]
    data_no_0 = data_no_na[data_no_na['rt'] != 0]
    return data_no_0

maze = preprocess_maze('Boyce/gulo_maze.rds') # datasets are obtained from https://github.com/vboyce/Maze/tree/master/experiment
                                              # preprocessing code is also Python adaptation from R code in the repository
spr  = preprocess_spr('Boyce/spr.rds')

maze_grouped = maze.groupby(["sentence", "word_num"], as_index=False, sort=False).agg({'rt':"mean", 'word':"max","word_num":"max", "type":"max"}) 
spr_grouped  = spr.groupby(["sentence", "word_num"], as_index=False, sort=False).agg({'rt':"mean", 'word':"max","word_num":"max", "type":"max"}) 

shared_sentences = sorted(list(set(maze_grouped["sentence"]).intersection(set(spr_grouped["sentence"]))))

len(set(maze_grouped["sentence"]).intersection(spr_grouped["sentence"]))

maze_grouped_shared = maze_grouped[maze_grouped.sentence.isin(shared_sentences)]
spr_grouped_shared  = spr_grouped[spr_grouped.sentence.isin(shared_sentences)]

# extracting average per-word RTs

def get_rt(dataset):
    sents = []
    rts   = []
    for sent in shared_sentences:
        df = dataset[dataset.sentence == sent]
        thesent = []
        rt   = []
        for idx, word in enumerate(sent.split()):
            therow = df[df.word_num == idx]
            if len(therow["word"]) == 1 and therow["word"].item() == word:
                thesent.append(word)
                rt.append(therow["rt"].item())
            else:
                print(therow, idx)
        rts.append(rt)
        sents.append(thesent)
    return sents, rts

maze_sent, maze_rt = get_rt(maze_grouped_shared)
spr_sent, spr_rt = get_rt(spr_grouped_shared)

# with open('corpora/boyce_sentences.txt', 'w') as f:
#     for line in maze_sent:
#         l = " ".join(line).strip()
#         f.write(f"{l}\n")

maze_sent = [re.sub("\n", "", s).split() for s in open("corpora/boyce_sentences.txt").readlines()]

######################################
# getting relevant psychometric data #
######################################

freq = pd.read_excel("/home/dev/Documents/Datasets/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}
minfreq = min(f.values())

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

length = [get_l(w) for sent in maze_sent for w in sent[1:]]
frequency = [get_f(w) for sent in maze_sent for w in sent[1:]]
rt_maze = [w for sent in maze_rt for w in sent[1:]]
rt_spr  = [w for sent in spr_rt for w in sent[1:]]

# GPT
model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt = [s for sent in tqdm(maze_sent) for s in get_surprisal_tokens(sent, sep = "</w>")]

# GPT2 SMALL (124M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2")
s_small = [s for sent in tqdm(maze_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 medium
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-medium")
s_med = [s for sent in tqdm(maze_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 large
model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-large")
s_large = [s for sent in tqdm(maze_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 xl
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl = [s for sent in tqdm(maze_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

boyce = pd.DataFrame({"spr" : rt_spr, "maze" : rt_maze, "logfreq" : frequency, "length" : length,
                      "s_GPT" : s_gpt, "s_GPT2_small" : s_small, "s_GPT2_med" : s_med, 
                      "s_GPT2_large" : s_large, "s_GPT2_xl" : s_xl})
# boyce.to_csv("dfs/boyce_surprisal.csv", index=False)
boyce = pd.read_csv("dfs/boyce_surprisal.csv")

###########################
# identify critical items #
###########################

noun = pyreadr.read_r("Boyce/Labelled/noun_clause_labelled.rds")[None]
adv  = pyreadr.read_r("Boyce/Labelled/adv_clause_labelled.rds")[None]
rel  = pyreadr.read_r("Boyce/Labelled/rel_clause_labelled.rds")[None]

merged = pd.merge(maze_grouped_shared, rel, on=["word_num", "sentence", "type", "word"], how="left")
merged = pd.merge(merged, adv, on=["word_num", "sentence", "type", "word"], how="left")
merged = pd.merge(merged, noun, on=["word_num", "sentence", "type", "word"], how="left")
merged['wn'] = merged['mod_word_number_x'].combine_first(merged['mod_word_number_y']).combine_first(merged['mod_word_number'])

def get_condition(dataset):
    sents = []
    conditions   = []
    for sent in shared_sentences:
        df = dataset[dataset.sentence == sent]
        condition   = []
        for idx, word in enumerate(sent.split()):
            therow = df[df.word_num == idx]
            if len(therow["word"]) == 1 and therow["word"].item() == word:
                condition.append(therow["wn"].item() == 0)
            else:
                print(therow, idx)
        conditions.append(condition)
    return conditions

maze_critical = get_condition(merged)
boyce["critical"] = [v for sent in maze_critical for v in sent[1:]]

##########################################
# need to account for spill-over effects #
##########################################

def create_word_pos(sentences):
    word_pos = []
    sent_nr = []
    for sentnum, sent in enumerate(sentences):
        for toknum, tok in enumerate(sent[1:]):
            word_pos.append(toknum+2)
            sent_nr.append(sentnum)
    return word_pos, sent_nr

boyce["word_pos"], boyce["sent_nr"] = create_word_pos(maze_sent)

def spill_maker(df):
    out = []
    for sent in set(df.sent_nr):
        temp = df[df.sent_nr == sent]
        idxmax = max(temp.word_pos)+1
        for idx in range(4, idxmax):
            twobef = temp[temp.word_pos == idx-2]
            onebef = temp[temp.word_pos == idx-1]
            actual = temp[temp.word_pos == idx]
            a = twobef.values.tolist()[0]+onebef.values.tolist()[0]+actual.values.tolist()[0]
            out.append(a)
    cols = [colname + "-2" for colname in df.columns.tolist()] + [colname + "-1" for colname in df.columns.tolist()] + df.columns.tolist()
    df_spill = pd.DataFrame(out, columns = cols) 
    return df_spill

boyce_spill = spill_maker(boyce)

##################
# get embeddings #
##################

# GPT 
# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt_boyce.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "corpora/boyce_sentences.txt",
        dirName+'/activations_gpt_boyce.json',
        aggregation="average"
    )

# GPT-2 FAMILY 
# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small_boyce.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "corpora/boyce_sentences.txt",
        dirName+'/activations_gpt2_small_boyce.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium_boyce.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "corpora/boyce_sentences.txt",
        dirName+'/activations_gpt2_medium_boyce.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large_boyce.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "corpora/boyce_sentences.txt",
        dirName+'/activations_gpt2_large_boyce.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl_boyce.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "corpora/boyce_sentences.txt",
        dirName+'/activations_gpt2_xl_boyce.json',
        aggregation="average"
    )


# load embeddings
act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt_boyce.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small_boyce.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium_boyce.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large_boyce.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl_boyce.json', 1600, is_brnn=False)

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

####################################
# filtering embeddings (spillover) #
####################################

def filter_embeddings_spill(df, emb):
    count = 0
    to_filt = []
    for sent in set(df.sent_nr):
        temp = df[df.sent_nr == sent]
        idxmax = max(temp.word_pos)+1
        for idx in range(2, idxmax):
            if idx > 3:
                to_filt.append([count-2, count-1, count])
            count += 1
    filtered = []
    for idx1, idx2, idx3 in to_filt:
        filtered.append(np.concatenate([emb[idx1], emb[idx2], emb[idx3]]))
    return np.array(filtered)
       
gpt_filt_spill   = filter_embeddings_spill(boyce, gpt_filt)
small_filt_spill = filter_embeddings_spill(boyce, small_filt)
med_filt_spill   = filter_embeddings_spill(boyce, med_filt)
large_filt_spill = filter_embeddings_spill(boyce, large_filt)
xl_filt_spill    = filter_embeddings_spill(boyce, xl_filt)

####################
# ACTIVATIONS ONLY #
####################

maze_y = np.array(boyce_spill["maze"])
spr_y  = np.array(boyce_spill["spr"])

# maze
m_gpt_maze, sd_gpt_maze     = test_model_Ridge(gpt_filt_spill, maze_y, 10, "boyce_maze_gpt")
m_small_maze, sd_small_maze = test_model_Ridge(small_filt_spill, maze_y, 10, "boyce_maze_small")
m_med_maze, sd_med_maze     = test_model_Ridge(med_filt_spill, maze_y, 10, "boyce_maze_med")
m_larg_maze, sd_larg_maze   = test_model_Ridge(large_filt_spill, maze_y, 10, "boyce_maze_large")
m_xl_maze, sd_xl_maze       = test_model_Ridge(xl_filt_spill, maze_y, 10, "boyce_maze_xl")
print(m_gpt_maze, m_small_maze, m_med_maze, m_larg_maze, m_xl_maze)

# spr
m_gpt_spr, sd_gpt_spr     = test_model_Ridge(gpt_filt_spill, spr_y, 10, "boyce_spr_gpt")
m_small_spr, sd_small_spr = test_model_Ridge(small_filt_spill, spr_y, 10, "boyce_spr_small")
m_med_spr, sd_med_spr     = test_model_Ridge(med_filt_spill, spr_y, 10, "boyce_spr_med")
m_larg_spr, sd_larg_spr   = test_model_Ridge(large_filt_spill, spr_y, 10, "boyce_spr_large")
m_xl_spr, sd_xl_spr       = test_model_Ridge(xl_filt_spill, spr_y, 10, "boyce_spr_xl")
print(m_gpt_spr, m_small_spr, m_med_spr, m_larg_spr, m_xl_spr)

#################
# BASELINE ONLY #
#################

# maze, baseline only
m_gpt_maze_b, sd_gpt_maze_b     = test_model_Ridge(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]]), maze_y, 10, 
                                                   "boyce_maze_gpt_base")
m_small_maze_b, sd_small_maze_b = test_model_Ridge(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2"]]), maze_y, 10, 
                                                   "boyce_maze_small_base")
m_med_maze_b, sd_med_maze_b     = test_model_Ridge(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2"]]), maze_y, 10, 
                                                   "boyce_maze_med_base")
m_larg_maze_b, sd_larg_maze_b   = test_model_Ridge(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2"]]), maze_y, 10, 
                                                   "boyce_maze_large_base")
m_xl_maze_b, sd_xl_maze_b       = test_model_Ridge(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2"]]), maze_y, 10, 
                                                   "boyce_maze_xl_base")
print(m_gpt_maze_b, m_small_maze_b, m_med_maze_b, m_larg_maze_b, m_xl_maze_b)

# decompose_sfl
decompose_sfl(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                              "s_GPT-1", "logfreq-1", "length-1",
                              "s_GPT-2", "logfreq-2", "length-2"]]), maze_y, "boyce_maze_gpt_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                              "s_GPT2_small-1", "logfreq-1", "length-1",
                              "s_GPT2_small-2", "logfreq-2", "length-2"]]), maze_y, "boyce_maze_small_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                              "s_GPT2_med-1", "logfreq-1", "length-1",
                              "s_GPT2_med-2", "logfreq-2", "length-2"]]), maze_y, "boyce_maze_med_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                              "s_GPT2_large-1", "logfreq-1", "length-1",
                              "s_GPT2_large-2", "logfreq-2", "length-2"]]), maze_y, "boyce_maze_large_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                              "s_GPT2_xl-1", "logfreq-1", "length-1",
                              "s_GPT2_xl-2", "logfreq-2", "length-2"]]), maze_y, "boyce_maze_xl_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])

# spr, baseline only
m_gpt_spr_b, sd_gpt_spr_b     = test_model_Ridge(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]]), spr_y, 10, 
                                                   "boyce_spr_gpt_base")
m_small_spr_b, sd_small_spr_b = test_model_Ridge(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2"]]), spr_y, 10, 
                                                   "boyce_spr_small_base")
m_med_spr_b, sd_med_spr_b     = test_model_Ridge(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2"]]), spr_y, 10, 
                                                   "boyce_spr_med_base")
m_larg_spr_b, sd_larg_spr_b   = test_model_Ridge(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2"]]), spr_y, 10, 
                                                   "boyce_spr_large_base")
m_xl_spr_b, sd_xl_spr_b       = test_model_Ridge(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2"]]), spr_y, 10, 
                                                   "boyce_spr_xl_base")
print(m_gpt_spr_b, m_small_spr_b, m_med_spr_b, m_larg_spr_b, m_xl_spr_b)

# decompose_sfl
decompose_sfl(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                              "s_GPT-1", "logfreq-1", "length-1",
                              "s_GPT-2", "logfreq-2", "length-2"]]), spr_y, "boyce_spr_gpt_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                              "s_GPT2_small-1", "logfreq-1", "length-1",
                              "s_GPT2_small-2", "logfreq-2", "length-2"]]), spr_y, "boyce_spr_small_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                              "s_GPT2_med-1", "logfreq-1", "length-1",
                              "s_GPT2_med-2", "logfreq-2", "length-2"]]), spr_y, "boyce_spr_med_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                              "s_GPT2_large-1", "logfreq-1", "length-1",
                              "s_GPT2_large-2", "logfreq-2", "length-2"]]), spr_y, "boyce_spr_large_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                              "s_GPT2_xl-1", "logfreq-1", "length-1",
                              "s_GPT2_xl-2", "logfreq-2", "length-2"]]), spr_y, "boyce_spr_xl_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])

##########################
# BASELINE + ACTIVATIONS #
##########################

gpt_all        = np.concatenate((np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                          "s_GPT-1", "logfreq-1", "length-1",
                                                          "s_GPT-2", "logfreq-2", "length-2"]]), gpt_filt_spill), axis=1)
gpt2_small_all = np.concatenate((np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                          "s_GPT2_small-1", "logfreq-1", "length-1",
                                                          "s_GPT2_small-2", "logfreq-2", "length-2"]]), small_filt_spill), axis=1)
gpt2_med_all   = np.concatenate((np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                          "s_GPT2_med-1", "logfreq-1", "length-1",
                                                          "s_GPT2_med-2", "logfreq-2", "length-2"]]), med_filt_spill), axis=1)
gpt2_large_all = np.concatenate((np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                          "s_GPT2_large-1", "logfreq-1", "length-1",
                                                          "s_GPT2_large-2", "logfreq-2", "length-2"]]), large_filt_spill), axis=1)
gpt2_xl_all    = np.concatenate((np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                          "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                          "s_GPT2_xl-2", "logfreq-2", "length-2"]]), xl_filt_spill), axis=1)

# all, maze
m_gpt_maze_all, sd_gpt_maze_all     = test_model_Ridge(gpt_all, maze_y, 10, "boyce_maze_gpt_all")
m_small_maze_all, sd_small_maze_all = test_model_Ridge(gpt2_small_all, maze_y, 10, "boyce_maze_small_all")
m_med_maze_all, sd_med_maze_all     = test_model_Ridge(gpt2_med_all, maze_y, 10, "boyce_maze_med_all")
m_larg_maze_all, sd_larg_maze_all   = test_model_Ridge(gpt2_large_all, maze_y, 10, "boyce_maze_large_all")
m_xl_maze_all, sd_xl_maze_all       = test_model_Ridge(gpt2_xl_all, maze_y, 10, "boyce_maze_xl_all")
print(m_gpt_maze_all, m_small_maze_all, m_med_maze_all, m_larg_maze_all, m_xl_maze_all)

# all, spr
m_gpt_spr_all, sd_gpt_spr_all     = test_model_Ridge(gpt_all, spr_y, 10, "boyce_spr_gpt_all")
m_small_spr_all, sd_small_spr_all = test_model_Ridge(gpt2_small_all, spr_y, 10, "boyce_spr_small_all")
m_med_spr_all, sd_med_spr_all     = test_model_Ridge(gpt2_med_all, spr_y, 10, "boyce_spr_med_all")
m_larg_spr_all, sd_larg_spr_all   = test_model_Ridge(gpt2_large_all, spr_y, 10, "boyce_spr_large_all")
m_xl_spr_all, sd_xl_spr_all       = test_model_Ridge(gpt2_xl_all, spr_y, 10, "boyce_spr_xl_all")
print(m_gpt_spr_all, m_small_spr_all, m_med_spr_all, m_larg_spr_all, m_xl_spr_all)


df_results = pd.DataFrame({
    "base_maze":[m_gpt_maze_b, m_small_maze_b, m_med_maze_b, m_larg_maze_b, m_xl_maze_b],
    "base_maze_sd": [sd_gpt_maze_b, sd_small_maze_b, sd_med_maze_b, sd_larg_maze_b, sd_xl_maze_b],
    "base_spr":  [m_gpt_spr_b, m_small_spr_b, m_med_spr_b, m_larg_spr_b, m_xl_spr_b],
    "base_spr_sd":  [sd_gpt_spr_b, sd_small_spr_b, sd_med_spr_b, sd_larg_spr_b, sd_xl_spr_b],
    "act_maze":[m_gpt_maze, m_small_maze, m_med_maze, m_larg_maze, m_xl_maze],
    "act_maze_sd": [sd_gpt_maze, sd_small_maze, sd_med_maze, sd_larg_maze, sd_xl_maze],
    "act_spr":  [m_gpt_spr, m_small_spr, m_med_spr, m_larg_spr, m_xl_spr],
    "act_spr_sd":  [sd_gpt_spr, sd_small_spr, sd_med_spr, sd_larg_spr, sd_xl_spr],
    "all_maze":[m_gpt_maze_all, m_small_maze_all, m_med_maze_all, m_larg_maze_all, m_xl_maze_all],
    "all_maze_sd":[sd_gpt_maze_all, sd_small_maze_all, sd_med_maze_all, sd_larg_maze_all, sd_xl_maze_all],
    "all_spr":[m_gpt_spr_all, m_small_spr_all, m_med_spr_all, m_larg_spr_all, m_xl_spr_all],
    "all_spr_sd":[sd_gpt_spr_all, sd_small_spr_all, sd_med_spr_all, sd_larg_spr_all, sd_xl_spr_all]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_results.to_csv("results_boyce.csv")
df_results = pd.read_csv("results_boyce.csv")

base_plot(df_results,"act_maze", "base_maze", "all_maze")
base_plot(df_results,"act_spr", "base_spr", "all_spr")

###############################################################################
###########################################
# train on non-critical, test on critical #
###########################################
###############################################################################

filt = np.array(boyce_spill["critical"])

def test_model_critical(X, y, saveto=None, save = True): # mimick test_model_Ridge()
    out_reg = []
    out_coefs = []
    out_predictions = []
    X_train, X_test, y_train, y_test = X[~filt], X[filt], y[~filt], y[filt]
    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r, _ = pearsonr(y_test, y_pred)
    out_predictions.append([y_test, y_pred])
    coefs = reg.coef_#; print(coefs)
    out_coefs.append(coefs)
    out_reg.append(r)
    coefs_avg = out_coefs
    coefs_std = out_coefs
    if save:
        with open("results/"+saveto+"_critical_coefs_avg", 'wb') as handle:
            pickle.dump(coefs_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_critical_coefs", 'wb') as handle:
            pickle.dump(out_coefs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_critical_coefs_std", 'wb') as handle:
            pickle.dump(coefs_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_critical_predictions", "wb") as handle:
            pickle.dump(out_reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return np.mean(out_reg), np.std(out_reg)

def decompose_sfl_critical(X, y, saveto, idx_s = 0, idx_f = 1, idx_l = 2, save=False):
    if type(idx_s) == int:
        s = X[:, idx_s].reshape(-1, 1)
        f = X[:, idx_f].reshape(-1, 1)
        l = X[:, idx_l].reshape(-1, 1)
        
        combs = [(s), (f), (l), (s, f), (f, l), (s, l), (s, f, l)]
        names = ["S", "F", "L", "SF",   "FL",   "SL",   "SFL"]
        
        out = {}
        for c, n in zip(combs, names):
            the_x = np.hstack(c)
            if len(the_x.shape) == 1:
                m, sd = test_model_critical(the_x.reshape(-1, 1), y, 10, save=False)
                out[n] = [m, sd]
            else:
                m, sd = test_model_critical(the_x, y, 10, save=False)
                out[n] = [m, sd]
        df = pd.DataFrame(out)
        df.index = ["m", "sd"]
        if save:
            df.to_csv("results/"+saveto+"_sfl_decomposed.csv")
        print("\n", df)
    else:
        s = X[:, idx_s]
        f = X[:, idx_f]
        l = X[:, idx_l]
        
        combs = [(s), (f), (l), (s, f), (f, l), (s, l), (s, f, l)]
        names = ["S", "F", "L", "SF",   "FL",   "SL",   "SFL"]
        
        out = {}
        for c, n in zip(combs, names):
            if n in ["S", "F", "L"]:
                m, sd = test_model_critical(c, y, save=False)
                out[n] = [m, sd]
            else:
                the_x = np.hstack(c)
                m, sd = test_model_critical(the_x, y, save=False)
                out[n] = [m, sd]
        df = pd.DataFrame(out)
        df.index = ["m", "sd"]
        if save:
            df.to_csv("results/"+saveto+"_sfl_decomposed.csv")
        print("\n", df)

# maze
m_gpt_maze, sd_gpt_maze     = test_model_critical(gpt_filt_spill, maze_y, "boyce_maze_gpt")
m_small_maze, sd_small_maze = test_model_critical(small_filt_spill, maze_y, "boyce_maze_small")
m_med_maze, sd_med_maze     = test_model_critical(med_filt_spill, maze_y, "boyce_maze_med")
m_larg_maze, sd_larg_maze   = test_model_critical(large_filt_spill, maze_y, "boyce_maze_large")
m_xl_maze, sd_xl_maze       = test_model_critical(xl_filt_spill, maze_y, "boyce_maze_xl")
print(m_gpt_maze, m_small_maze, m_med_maze, m_larg_maze, m_xl_maze)

# spr
m_gpt_spr, sd_gpt_spr     = test_model_critical(gpt_filt_spill, spr_y, "boyce_spr_gpt")
m_small_spr, sd_small_spr = test_model_critical(small_filt_spill, spr_y, "boyce_spr_small")
m_med_spr, sd_med_spr     = test_model_critical(med_filt_spill, spr_y,"boyce_spr_med")
m_larg_spr, sd_larg_spr   = test_model_critical(large_filt_spill, spr_y,"boyce_spr_large")
m_xl_spr, sd_xl_spr       = test_model_critical(xl_filt_spill, spr_y,"boyce_spr_xl")
print(m_gpt_spr, m_small_spr, m_med_spr, m_larg_spr, m_xl_spr)

#################
# BASELINE ONLY #
#################

# maze, baseline only
m_gpt_maze_b, sd_gpt_maze_b     = test_model_critical(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]]), maze_y,
                                                   "boyce_maze_gpt_base")
m_small_maze_b, sd_small_maze_b = test_model_critical(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2"]]), maze_y,
                                                   "boyce_maze_small_base")
m_med_maze_b, sd_med_maze_b     = test_model_critical(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2"]]), maze_y,
                                                   "boyce_maze_med_base")
m_larg_maze_b, sd_larg_maze_b   = test_model_critical(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2"]]), maze_y,
                                                   "boyce_maze_large_base")
m_xl_maze_b, sd_xl_maze_b       = test_model_critical(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2"]]), maze_y,
                                                   "boyce_maze_xl_base")
print(m_gpt_maze_b, m_small_maze_b, m_med_maze_b, m_larg_maze_b, m_xl_maze_b)


# spr, baseline only
m_gpt_spr_b, sd_gpt_spr_b     = test_model_critical(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]]), spr_y,
                                                   "boyce_spr_gpt_base")
m_small_spr_b, sd_small_spr_b = test_model_critical(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2"]]), spr_y,
                                                   "boyce_spr_small_base")
m_med_spr_b, sd_med_spr_b     = test_model_critical(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2"]]), spr_y,
                                                   "boyce_spr_med_base")
m_larg_spr_b, sd_larg_spr_b   = test_model_critical(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2"]]), spr_y,
                                                   "boyce_spr_large_base")
m_xl_spr_b, sd_xl_spr_b       = test_model_critical(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2"]]), spr_y,
                                                   "boyce_spr_xl_base")
print(m_gpt_spr_b, m_small_spr_b, m_med_spr_b, m_larg_spr_b, m_xl_spr_b)


# decompose SFL critical
decompose_sfl_critical(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                             "s_GPT-1", "logfreq-1", "length-1",
                                             "s_GPT-2", "logfreq-2", "length-2",]]), maze_y,
                                              "boyce_maze_gpt_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])
decompose_sfl_critical(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                             "s_GPT2_small-1", "logfreq-1", "length-1",
                                             "s_GPT2_small-2", "logfreq-2", "length-2",]]), maze_y,
                                              "boyce_maze_small_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])
decompose_sfl_critical(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                             "s_GPT2_med-1", "logfreq-1", "length-1",
                                             "s_GPT2_med-2", "logfreq-2", "length-2",]]), maze_y,
                                              "boyce_maze_med_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])

decompose_sfl_critical(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                             "s_GPT2_large-1", "logfreq-1", "length-1",
                                             "s_GPT2_large-2", "logfreq-2", "length-2",]]), maze_y,
                                              "boyce_maze_large_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])
decompose_sfl_critical(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                             "s_GPT2_xl-1", "logfreq-1", "length-1",
                                             "s_GPT2_xl-2", "logfreq-2", "length-2",]]), maze_y,
                                              "boyce_maze_xl_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])


# decompose SFL critical, spr
decompose_sfl_critical(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                             "s_GPT-1", "logfreq-1", "length-1",
                                             "s_GPT-2", "logfreq-2", "length-2",]]), spr_y,
                                              "boyce_spr_gpt_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])
decompose_sfl_critical(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                             "s_GPT2_small-1", "logfreq-1", "length-1",
                                             "s_GPT2_small-2", "logfreq-2", "length-2",]]), spr_y,
                                              "boyce_spr_small_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])
decompose_sfl_critical(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                             "s_GPT2_med-1", "logfreq-1", "length-1",
                                             "s_GPT2_med-2", "logfreq-2", "length-2",]]), spr_y,
                                              "boyce_spr_med_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])

decompose_sfl_critical(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                             "s_GPT2_large-1", "logfreq-1", "length-1",
                                             "s_GPT2_large-2", "logfreq-2", "length-2",]]), spr_y,
                                              "boyce_spr_large_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])
decompose_sfl_critical(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                             "s_GPT2_xl-1", "logfreq-1", "length-1",
                                             "s_GPT2_xl-2", "logfreq-2", "length-2",]]), spr_y,
                                              "boyce_spr_xl_base", idx_s = [0, 3, 6], idx_f = [1, 4, 7], idx_l = [2, 5, 8])

##########################
# BASELINE + ACTIVATIONS #
##########################

gpt_all        = np.concatenate((np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                          "s_GPT-1", "logfreq-1", "length-1",
                                                          "s_GPT-2", "logfreq-2", "length-2"]]), gpt_filt_spill), axis=1)
gpt2_small_all = np.concatenate((np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                          "s_GPT2_small-1", "logfreq-1", "length-1",
                                                          "s_GPT2_small-2", "logfreq-2", "length-2"]]), small_filt_spill), axis=1)
gpt2_med_all   = np.concatenate((np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                          "s_GPT2_med-1", "logfreq-1", "length-1",
                                                          "s_GPT2_med-2", "logfreq-2", "length-2"]]), med_filt_spill), axis=1)
gpt2_large_all = np.concatenate((np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                          "s_GPT2_large-1", "logfreq-1", "length-1",
                                                          "s_GPT2_large-2", "logfreq-2", "length-2"]]), large_filt_spill), axis=1)
gpt2_xl_all    = np.concatenate((np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                          "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                          "s_GPT2_xl-2", "logfreq-2", "length-2"]]), xl_filt_spill), axis=1)

# all, maze
m_gpt_maze_all, sd_gpt_maze_all     = test_model_critical(gpt_all, maze_y, "boyce_maze_gpt_all")
m_small_maze_all, sd_small_maze_all = test_model_critical(gpt2_small_all, maze_y, "boyce_maze_small_all")
m_med_maze_all, sd_med_maze_all     = test_model_critical(gpt2_med_all, maze_y, "boyce_maze_med_all")
m_larg_maze_all, sd_larg_maze_all   = test_model_critical(gpt2_large_all, maze_y, "boyce_maze_large_all")
m_xl_maze_all, sd_xl_maze_all       = test_model_critical(gpt2_xl_all, maze_y, "boyce_maze_xl_all")
print(m_gpt_maze_all, m_small_maze_all, m_med_maze_all, m_larg_maze_all, m_xl_maze_all)

# all, spr
m_gpt_spr_all, sd_gpt_spr_all     = test_model_critical(gpt_all, spr_y, "boyce_spr_gpt_all")
m_small_spr_all, sd_small_spr_all = test_model_critical(gpt2_small_all, spr_y, "boyce_spr_small_all")
m_med_spr_all, sd_med_spr_all     = test_model_critical(gpt2_med_all, spr_y, "boyce_spr_med_all")
m_larg_spr_all, sd_larg_spr_all   = test_model_critical(gpt2_large_all, spr_y, "boyce_spr_large_all")
m_xl_spr_all, sd_xl_spr_all       = test_model_critical(gpt2_xl_all, spr_y, "boyce_spr_xl_all")
print(m_gpt_spr_all, m_small_spr_all, m_med_spr_all, m_larg_spr_all, m_xl_spr_all)

df_critical = pd.DataFrame({
    "base_maze":[m_gpt_maze_b, m_small_maze_b, m_med_maze_b, m_larg_maze_b, m_xl_maze_b],
    "base_maze_sd": [sd_gpt_maze_b, sd_small_maze_b, sd_med_maze_b, sd_larg_maze_b, sd_xl_maze_b],
    "base_spr":  [m_gpt_spr_b, m_small_spr_b, m_med_spr_b, m_larg_spr_b, m_xl_spr_b],
    "base_spr_sd":  [sd_gpt_spr_b, sd_small_spr_b, sd_med_spr_b, sd_larg_spr_b, sd_xl_spr_b],
    "act_maze":[m_gpt_maze, m_small_maze, m_med_maze, m_larg_maze, m_xl_maze],
    "act_maze_sd": [sd_gpt_maze, sd_small_maze, sd_med_maze, sd_larg_maze, sd_xl_maze],
    "act_spr":  [m_gpt_spr, m_small_spr, m_med_spr, m_larg_spr, m_xl_spr],
    "act_spr_sd":  [sd_gpt_spr, sd_small_spr, sd_med_spr, sd_larg_spr, sd_xl_spr],
    "all_maze":[m_gpt_maze_all, m_small_maze_all, m_med_maze_all, m_larg_maze_all, m_xl_maze_all],
    "all_maze_sd":[sd_gpt_maze_all, sd_small_maze_all, sd_med_maze_all, sd_larg_maze_all, sd_xl_maze_all],
    "all_spr":[m_gpt_spr_all, m_small_spr_all, m_med_spr_all, m_larg_spr_all, m_xl_spr_all],
    "all_spr_sd":[sd_gpt_spr_all, sd_small_spr_all, sd_med_spr_all, sd_larg_spr_all, sd_xl_spr_all]
    })
df_critical.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_critical.to_csv("results_boyce_critical.csv")
df_critical = pd.read_csv("results_boyce_critical.csv")

base_plot(df_critical,"act_maze", "base_maze", "all_maze", ylim1=0, legend=False, title = r"Boyce Maze, NC $\rightarrow$ C")
base_plot(df_critical,"act_spr", "base_spr", "all_spr", ylim1=0, legend=False, title = r"Boyce SPR, NC $\rightarrow$ C")

#################
# CRITICAL ONLY #
#################

def test_model_criticalonly(X, y, n, saveto=None, save = True): # mimick test_model_Ridge()
    out_reg = []
    out_coefs = []
    out_predictions = []
    X_train, X_test, y_train, y_test = X[~filt], X[filt], y[~filt], y[filt]
    m, sd = test_model_Ridge(X_test, y_test, 10, saveto+"_criticalonly")
    return m, sd

# maze
m_gpt_maze, sd_gpt_maze     = test_model_criticalonly(gpt_filt_spill, maze_y, 10, "boyce_maze_gpt")
m_small_maze, sd_small_maze = test_model_criticalonly(small_filt_spill, maze_y, 10, "boyce_maze_small")
m_med_maze, sd_med_maze     = test_model_criticalonly(med_filt_spill, maze_y, 10, "boyce_maze_med")
m_larg_maze, sd_larg_maze   = test_model_criticalonly(large_filt_spill, maze_y, 10, "boyce_maze_large")
m_xl_maze, sd_xl_maze       = test_model_criticalonly(xl_filt_spill, maze_y, 10, "boyce_maze_xl")
print(m_gpt_maze, m_small_maze, m_med_maze, m_larg_maze, m_xl_maze)

# spr
m_gpt_spr, sd_gpt_spr     = test_model_criticalonly(gpt_filt_spill, spr_y, 10, "boyce_spr_gpt")
m_small_spr, sd_small_spr = test_model_criticalonly(small_filt_spill, spr_y, 10, "boyce_spr_small")
m_med_spr, sd_med_spr     = test_model_criticalonly(med_filt_spill, spr_y, 10, "boyce_spr_med")
m_larg_spr, sd_larg_spr   = test_model_criticalonly(large_filt_spill, spr_y, 10, "boyce_spr_large")
m_xl_spr, sd_xl_spr       = test_model_criticalonly(xl_filt_spill, spr_y, 10, "boyce_spr_xl")
print(m_gpt_spr, m_small_spr, m_med_spr, m_larg_spr, m_xl_spr)

#################
# BASELINE ONLY #
#################

# maze, baseline only
m_gpt_maze_b, sd_gpt_maze_b     = test_model_criticalonly(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]]), maze_y, 10, 
                                                   "boyce_maze_gpt_base")
m_small_maze_b, sd_small_maze_b = test_model_criticalonly(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2"]]), maze_y, 10, 
                                                   "boyce_maze_small_base")
m_med_maze_b, sd_med_maze_b     = test_model_criticalonly(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2"]]), maze_y, 10, 
                                                   "boyce_maze_med_base")
m_larg_maze_b, sd_larg_maze_b   = test_model_criticalonly(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2"]]), maze_y, 10, 
                                                   "boyce_maze_large_base")
m_xl_maze_b, sd_xl_maze_b       = test_model_criticalonly(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2"]]), maze_y, 10, 
                                                   "boyce_maze_xl_base")
print(m_gpt_maze_b, m_small_maze_b, m_med_maze_b, m_larg_maze_b, m_xl_maze_b)

# spr, baseline only
m_gpt_spr_b, sd_gpt_spr_b     = test_model_criticalonly(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]]), spr_y, 10, 
                                                   "boyce_spr_gpt_base")
m_small_spr_b, sd_small_spr_b = test_model_criticalonly(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2"]]), spr_y, 10, 
                                                   "boyce_spr_small_base")
m_med_spr_b, sd_med_spr_b     = test_model_criticalonly(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2"]]), spr_y, 10, 
                                                   "boyce_spr_med_base")
m_larg_spr_b, sd_larg_spr_b   = test_model_criticalonly(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2"]]), spr_y, 10, 
                                                   "boyce_spr_large_base")
m_xl_spr_b, sd_xl_spr_b       = test_model_criticalonly(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2"]]), spr_y, 10, 
                                                   "boyce_spr_xl_base")
print(m_gpt_spr_b, m_small_spr_b, m_med_spr_b, m_larg_spr_b, m_xl_spr_b)

##########################
# BASELINE + ACTIVATIONS #
##########################

gpt_all        = np.concatenate((np.array(boyce_spill[["s_GPT", "logfreq", "length",
                                                          "s_GPT-1", "logfreq-1", "length-1",
                                                          "s_GPT-2", "logfreq-2", "length-2"]]), gpt_filt_spill), axis=1)
gpt2_small_all = np.concatenate((np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                                                          "s_GPT2_small-1", "logfreq-1", "length-1",
                                                          "s_GPT2_small-2", "logfreq-2", "length-2"]]), small_filt_spill), axis=1)
gpt2_med_all   = np.concatenate((np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                                                          "s_GPT2_med-1", "logfreq-1", "length-1",
                                                          "s_GPT2_med-2", "logfreq-2", "length-2"]]), med_filt_spill), axis=1)
gpt2_large_all = np.concatenate((np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                                                          "s_GPT2_large-1", "logfreq-1", "length-1",
                                                          "s_GPT2_large-2", "logfreq-2", "length-2"]]), large_filt_spill), axis=1)
gpt2_xl_all    = np.concatenate((np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                                                          "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                          "s_GPT2_xl-2", "logfreq-2", "length-2"]]), xl_filt_spill), axis=1)

# all, maze
m_gpt_maze_all, sd_gpt_maze_all     = test_model_criticalonly(gpt_all, maze_y, 10, "boyce_maze_gpt_all")
m_small_maze_all, sd_small_maze_all = test_model_criticalonly(gpt2_small_all, maze_y, 10, "boyce_maze_small_all")
m_med_maze_all, sd_med_maze_all     = test_model_criticalonly(gpt2_med_all, maze_y, 10, "boyce_maze_med_all")
m_larg_maze_all, sd_larg_maze_all   = test_model_criticalonly(gpt2_large_all, maze_y, 10, "boyce_maze_large_all")
m_xl_maze_all, sd_xl_maze_all       = test_model_criticalonly(gpt2_xl_all, maze_y, 10, "boyce_maze_xl_all")
print(m_gpt_maze_all, m_small_maze_all, m_med_maze_all, m_larg_maze_all, m_xl_maze_all)

# all, spr
m_gpt_spr_all, sd_gpt_spr_all     = test_model_criticalonly(gpt_all, spr_y, 10, "boyce_spr_gpt_all")
m_small_spr_all, sd_small_spr_all = test_model_criticalonly(gpt2_small_all, spr_y, 10, "boyce_spr_small_all")
m_med_spr_all, sd_med_spr_all     = test_model_criticalonly(gpt2_med_all, spr_y, 10, "boyce_spr_med_all")
m_larg_spr_all, sd_larg_spr_all   = test_model_criticalonly(gpt2_large_all, spr_y, 10, "boyce_spr_large_all")
m_xl_spr_all, sd_xl_spr_all       = test_model_criticalonly(gpt2_xl_all, spr_y, 10, "boyce_spr_xl_all")
print(m_gpt_spr_all, m_small_spr_all, m_med_spr_all, m_larg_spr_all, m_xl_spr_all)


df_results_criticalonly = pd.DataFrame({
    "base_maze":[m_gpt_maze_b, m_small_maze_b, m_med_maze_b, m_larg_maze_b, m_xl_maze_b],
    "base_maze_sd": [sd_gpt_maze_b, sd_small_maze_b, sd_med_maze_b, sd_larg_maze_b, sd_xl_maze_b],
    "base_spr":  [m_gpt_spr_b, m_small_spr_b, m_med_spr_b, m_larg_spr_b, m_xl_spr_b],
    "base_spr_sd":  [sd_gpt_spr_b, sd_small_spr_b, sd_med_spr_b, sd_larg_spr_b, sd_xl_spr_b],
    "act_maze":[m_gpt_maze, m_small_maze, m_med_maze, m_larg_maze, m_xl_maze],
    "act_maze_sd": [sd_gpt_maze, sd_small_maze, sd_med_maze, sd_larg_maze, sd_xl_maze],
    "act_spr":  [m_gpt_spr, m_small_spr, m_med_spr, m_larg_spr, m_xl_spr],
    "act_spr_sd":  [sd_gpt_spr, sd_small_spr, sd_med_spr, sd_larg_spr, sd_xl_spr],
    "all_maze":[m_gpt_maze_all, m_small_maze_all, m_med_maze_all, m_larg_maze_all, m_xl_maze_all],
    "all_maze_sd":[sd_gpt_maze_all, sd_small_maze_all, sd_med_maze_all, sd_larg_maze_all, sd_xl_maze_all],
    "all_spr":[m_gpt_spr_all, m_small_spr_all, m_med_spr_all, m_larg_spr_all, m_xl_spr_all],
    "all_spr_sd":[sd_gpt_spr_all, sd_small_spr_all, sd_med_spr_all, sd_larg_spr_all, sd_xl_spr_all]
    })
df_results_criticalonly.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_results_criticalonly.to_csv("results_boyce_criticalonly.csv")
df_results_criticalonly = pd.read_csv("results_boyce_criticalonly.csv")

base_plot(df_results_criticalonly,"act_maze", "base_maze", "all_maze", legend = True, title="Boyce Maze, critical-only")
base_plot(df_results_criticalonly,"act_spr", "base_spr", "all_spr", legend = True, title="Boyce SPR, critical-only")


decompose_sfl(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                              "s_GPT-1", "logfreq-1", "length-1",
                              "s_GPT-2", "logfreq-2", "length-2"]])[filt], spr_y[filt], "boyce_spr_gpt_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                              "s_GPT2_small-1", "logfreq-1", "length-1",
                              "s_GPT2_small-2", "logfreq-2", "length-2"]])[filt], spr_y[filt], "boyce_spr_small_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                              "s_GPT2_med-1", "logfreq-1", "length-1",
                              "s_GPT2_med-2", "logfreq-2", "length-2"]])[filt], spr_y[filt], "boyce_spr_med_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                              "s_GPT2_large-1", "logfreq-1", "length-1",
                              "s_GPT2_large-2", "logfreq-2", "length-2"]])[filt], spr_y[filt], "boyce_spr_large_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                              "s_GPT2_xl-1", "logfreq-1", "length-1",
                              "s_GPT2_xl-2", "logfreq-2", "length-2"]])[filt], spr_y[filt], "boyce_spr_xl_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])


decompose_sfl(np.array(boyce_spill[["s_GPT", "logfreq", "length",
                              "s_GPT-1", "logfreq-1", "length-1",
                              "s_GPT-2", "logfreq-2", "length-2"]])[filt], maze_y[filt], "boyce_maze_gpt_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_small", "logfreq", "length",
                              "s_GPT2_small-1", "logfreq-1", "length-1",
                              "s_GPT2_small-2", "logfreq-2", "length-2"]])[filt], maze_y[filt], "boyce_maze_small_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_med", "logfreq", "length",
                              "s_GPT2_med-1", "logfreq-1", "length-1",
                              "s_GPT2_med-2", "logfreq-2", "length-2"]])[filt], maze_y[filt], "boyce_maze_med_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_large", "logfreq", "length",
                              "s_GPT2_large-1", "logfreq-1", "length-1",
                              "s_GPT2_large-2", "logfreq-2", "length-2"]])[filt], maze_y[filt], "boyce_maze_large_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(boyce_spill[["s_GPT2_xl", "logfreq", "length",
                              "s_GPT2_xl-1", "logfreq-1", "length-1",
                              "s_GPT2_xl-2", "logfreq-2", "length-2"]])[filt], maze_y[filt], "boyce_maze_xl_base_criticalonly", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
