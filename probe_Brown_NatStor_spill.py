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
from joblib import dump, load
from scipy.special import softmax
import matplotlib.patches as mpatches
import pickle

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT)
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, decompose_sfl, probe

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
if path.isfile(dirName+'/activations_gpt_natstor.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "corpora/natural_stories.txt",
        dirName+'/activations_gpt_natstor.json',
        aggregation="average"
    )

# GPT-2 FAMILY 
# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small_natstor.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "corpora/natural_stories.txt",
        dirName+'/activations_gpt2_small_natstor.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium_natstor.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "corpora/natural_stories.txt",
        dirName+'/activations_gpt2_medium_natstor.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large_natstor.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "corpora/natural_stories.txt",
        dirName+'/activations_gpt2_large_natstor.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl_natstor.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "corpora/natural_stories.txt",
        dirName+'/activations_gpt2_xl_natstor.json',
        aggregation="average"
    )

# TRANSFORMER-XL
# 18-block, 1024 hidden, 8 heads
# if path.isfile(dirName+'/activations_transf_xl_natstor.json') == False:
#     transformers_extractor.extract_representations('transfo-xl-wt103',
#         "corpora/natural_stories.txt",
#         dirName+'/activations_transf_xl_natstor.json',
#         aggregation="average"
#     )
    
#########
# BROWN #
#########

# GPT 
# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt_brown.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "corpora/bnc_brown.txt",
        dirName+'/activations_gpt_brown.json',
        aggregation="average"
    )


# GPT-2 FAMILY 
# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small_brown.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "corpora/bnc_brown.txt",
        dirName+'/activations_gpt2_small_brown.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium_brown.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "corpora/bnc_brown.txt",
        dirName+'/activations_gpt2_medium_brown.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large_brown.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "corpora/bnc_brown.txt",
        dirName+'/activations_gpt2_large_brown.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl_brown.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "corpora/bnc_brown.txt",
        dirName+'/activations_gpt2_xl_brown.json',
        aggregation="average"
    )

# TRANSFORMER-XL 
# # 18-block, 1024 hidden, 8 heads
# if path.isfile(dirName+'/activations_transf_xl_brown.json') == False:
#     transformers_extractor.extract_representations('transfo-xl-wt103',
#         "corpora/bnc_brown.txt",
#         dirName+'/activations_transf_xl_brown.json',
#         aggregation="average"
#     )


###############################
# GET SURPRISAL, FREQ, LENGTH #
###############################

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


brown_sent = [re.sub("\n", "", s).split() for s in open("corpora/bnc_brown.txt").readlines()]
natst_sent = [re.sub("\n", "", s).split() for s in open("corpora/natural_stories.txt").readlines()]

freq_brown = [get_f(t) for w in brown_sent for t in w[1:]]
len_brown  = [get_l(t) for w in brown_sent for t in w[1:]]
freq_natst = [get_f(t) for w in natst_sent for t in w[1:]]
len_natst =  [get_l(t) for w in natst_sent for t in w[1:]]

# GPT
model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt_brown = [s for sent in tqdm(brown_sent) for s in get_surprisal_tokens(sent, sep = "</w>")]
s_gpt_natst = [s for sent in tqdm(natst_sent) for s in get_surprisal_tokens(sent, sep = "</w>")]

# GPT2 SMALL (124M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2")
s_small_brown = [s for sent in tqdm(brown_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]
s_small_natst = [s for sent in tqdm(natst_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 medium
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-medium")
s_med_brown = [s for sent in tqdm(brown_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]
s_med_natst = [s for sent in tqdm(natst_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 large
model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-large")
s_large_brown = [s for sent in tqdm(brown_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]
s_large_natst = [s for sent in tqdm(natst_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 xl
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl_brown = [s for sent in tqdm(brown_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]
s_xl_natst = [s for sent in tqdm(natst_sent) for s in get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# # Transformer-XL
# model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103", return_dict_in_generate=True)
# toker = AutoTokenizer.from_pretrained("transfo-xl-wt103")
# s_t_xl_brown = [s for sent in tqdm(brown_sent) for s in get_surprisal_tokens(sent, sep = "", cased=True)]
# s_t_xl_natst = [s for sent in tqdm(natst_sent) for s in get_surprisal_tokens(sent, sep = "", cased=True)]

################################################
# FILTERING behav data (no sent initial words) #
################################################

ns_spr = pd.read_csv(PROJECT_ROOT / "main_data/NaturalStories/SPR/natural-stories.csv", sep=",")
brown_spr = pd.read_csv(PROJECT_ROOT / "main_data/Brown/bnc-brown.csv", sep=",")

filt_brown = []
for sent in brown_sent:
    for idx, tok in enumerate(sent):
        if idx == 0:
            filt_brown.append(0)
        else:
            filt_brown.append(1)
brown_spr["filt"] = filt_brown
brown_spr = brown_spr[brown_spr["filt"]==1]

filt_natst = []
for sent in natst_sent:
    for idx, tok in enumerate(sent):
        if idx == 0:
            filt_natst.append(0)
        else:
            filt_natst.append(1) 
ns_spr["filt"] = filt_natst
ns_spr = ns_spr[ns_spr["filt"]==1]

ns_spr["s_GPT"] = s_gpt_natst
ns_spr["s_GPT2_small"] = s_small_natst
ns_spr["s_GPT2_med"] = s_med_natst
ns_spr["s_GPT2_large"] = s_large_natst
ns_spr["s_GPT2_xl"] = s_xl_natst
ns_spr["logfreq"] = freq_natst
ns_spr["length"] = len_natst
#ns_spr.to_csv("dfs/natstor_surprisal.csv", index=False)
ns_spr = pd.read_csv("dfs/natstor_surprisal.csv")

brown_spr["s_GPT"] = s_gpt_brown
brown_spr["s_GPT2_small"] = s_small_brown
brown_spr["s_GPT2_med"] = s_med_brown
brown_spr["s_GPT2_large"] = s_large_brown
brown_spr["s_GPT2_xl"] = s_xl_brown
brown_spr["logfreq"] = freq_brown
brown_spr["length"] = len_brown
#brown_spr.to_csv("dfs/brown_surprisal.csv", index=False)
brown_spr = pd.read_csv("dfs/brown_surprisal.csv")

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

brown_spr["word_pos"], brown_spr["sent_nr"] = create_word_pos(brown_sent)
ns_spr["word_pos"], ns_spr["sent_nr"] = create_word_pos(natst_sent)

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

brown_spill = spill_maker(brown_spr)
ns_spill    = spill_maker(ns_spr)

################
# loading data #
################

# natstor
act_gpt_natstor, num_layers_gpt_natstor = data_loader.load_activations(dirName+'/activations_gpt_natstor.json', 768, is_brnn=False)
act_gpt2_small_natstor, num_layers_gpt2_small_natstor = data_loader.load_activations(dirName+'/activations_gpt2_small_natstor.json', 768, is_brnn=False)
act_gpt2_medium_natstor, num_layers_gpt2_medium_natstor = data_loader.load_activations(dirName+'/activations_gpt2_medium_natstor.json', 1024, is_brnn=False)
act_gpt2_large_natstor, num_layers_gpt2_large_natstor = data_loader.load_activations(dirName+'/activations_gpt2_large_natstor.json', 1280, is_brnn=False)
act_gpt2_xl_natstor, num_layers_gpt2_xl_natstor = data_loader.load_activations(dirName+'/activations_gpt2_xl_natstor.json', 1600, is_brnn=False)

# brown
act_gpt_brown, num_layers_gpt_brown = data_loader.load_activations(dirName+'/activations_gpt_brown.json', 768, is_brnn=False)
act_gpt2_small_brown, num_layers_gpt2_small_brown = data_loader.load_activations(dirName+'/activations_gpt2_small_brown.json', 768, is_brnn=False)
act_gpt2_medium_brown, num_layers_gpt2_medium_brown = data_loader.load_activations(dirName+'/activations_gpt2_medium_brown.json', 1024, is_brnn=False)
act_gpt2_large_brown, num_layers_gpt2_large_brown = data_loader.load_activations(dirName+'/activations_gpt2_large_brown.json', 1280, is_brnn=False)
act_gpt2_xl_brown, num_layers_gpt2_xl_brown = data_loader.load_activations(dirName+'/activations_gpt2_xl_brown.json', 1600, is_brnn=False)

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act_eye(act_list, layer, dim):
    out = []
    for act_sent in act_list:
        out.extend([get_layer(v, layer, dim) for v in act_sent[1:]])
    return np.array(out)

gpt_filt_natstor   = get_act_eye(act_gpt_natstor, 12, 768)
small_filt_natstor = get_act_eye(act_gpt2_small_natstor, 12, 768)
med_filt_natstor   = get_act_eye(act_gpt2_medium_natstor, 24, 1024)
large_filt_natstor = get_act_eye(act_gpt2_large_natstor, 36, 1280)
xl_filt_natstor    = get_act_eye(act_gpt2_xl_natstor, 48, 1600)

gpt_filt_brown   = get_act_eye(act_gpt_brown, 12, 768)
small_filt_brown = get_act_eye(act_gpt2_small_brown, 12, 768)
med_filt_brown   = get_act_eye(act_gpt2_medium_brown, 24, 1024)
large_filt_brown = get_act_eye(act_gpt2_large_brown, 36, 1280)
xl_filt_brown    = get_act_eye(act_gpt2_xl_brown, 48, 1600)

# del act_gpt_natstor, act_gpt2_small_natstor, act_gpt2_medium_natstor, act_gpt2_large_natstor, act_gpt2_xl_natstor
# del act_gpt_brown, act_gpt2_small_brown, act_gpt2_medium_brown, act_gpt2_large_brown, act_gpt2_xl_brown

#####################################
# filtering embeddings (spill-over) #
#####################################

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
       
gpt_filt_brown_spill   = filter_embeddings_spill(brown_spr, gpt_filt_brown)
small_filt_brown_spill = filter_embeddings_spill(brown_spr, small_filt_brown)
med_filt_brown_spill   = filter_embeddings_spill(brown_spr, med_filt_brown)
large_filt_brown_spill = filter_embeddings_spill(brown_spr, large_filt_brown)
xl_filt_brown_spill    = filter_embeddings_spill(brown_spr, xl_filt_brown)

gpt_filt_ns_spill   = filter_embeddings_spill(ns_spr, gpt_filt_natstor)
small_filt_ns_spill = filter_embeddings_spill(ns_spr, small_filt_natstor)
med_filt_ns_spill   = filter_embeddings_spill(ns_spr, med_filt_natstor)
large_filt_ns_spill = filter_embeddings_spill(ns_spr, large_filt_natstor)
xl_filt_ns_spill    = filter_embeddings_spill(ns_spr, xl_filt_natstor)

del gpt_filt_brown, small_filt_brown, med_filt_brown, large_filt_brown, xl_filt_brown
del gpt_filt_natstor, small_filt_natstor, med_filt_natstor, large_filt_natstor, xl_filt_natstor

####################
# ACTIVATIONS ONLY #
####################

ns_y = np.array(ns_spill["psychometric"])
br_y = np.array(brown_spill["psychometric"])

# # natstor
# m_gpt_natstor, sd_gpt_natstor     = test_model_Ridge(gpt_filt_ns_spill, ns_y, 10, save_predictions=True, pred_file="gpt_natstor")
# m_small_natstor, sd_small_natstor = test_model_Ridge(small_filt_ns_spill, ns_y, 10, save_predictions=True, pred_file="small_natstor")
# m_med_natstor, sd_med_natstor     = test_model_Ridge(med_filt_ns_spill, ns_y, 10, save_predictions=True, pred_file="med_natstor")
# m_larg_natstor, sd_larg_natstor   = test_model_Ridge(large_filt_ns_spill, ns_y, 10, save_predictions=True, pred_file="large_natstor")
# m_xl_natstor, sd_xl_natstor       = test_model_Ridge(xl_filt_ns_spill, ns_y, 10, save_predictions=True, pred_file="xl_natstor")
# print(m_gpt_natstor, m_small_natstor, m_med_natstor, m_larg_natstor, m_xl_natstor)

# natstor
m_gpt_natstor, sd_gpt_natstor     = test_model_Ridge(gpt_filt_ns_spill, ns_y, 10, "natstor_gpt")
m_small_natstor, sd_small_natstor = test_model_Ridge(small_filt_ns_spill, ns_y, 10, "natstor_small")
m_med_natstor, sd_med_natstor     = test_model_Ridge(med_filt_ns_spill, ns_y, 10, "natstor_med")
m_larg_natstor, sd_larg_natstor   = test_model_Ridge(large_filt_ns_spill, ns_y, 10, "natstor_large")
m_xl_natstor, sd_xl_natstor       = test_model_Ridge(xl_filt_ns_spill, ns_y, 10, "natstor_xl")
print(m_gpt_natstor, m_small_natstor, m_med_natstor, m_larg_natstor, m_xl_natstor)


# brown
m_gpt_brown, sd_gpt_brown     = test_model_Ridge(gpt_filt_brown_spill, br_y, 10, "brown_gpt")
m_small_brown, sd_small_brown = test_model_Ridge(small_filt_brown_spill, br_y, 10, "brown_small")
m_med_brown, sd_med_brown     = test_model_Ridge(med_filt_brown_spill, br_y, 10, "brown_med")
m_larg_brown, sd_larg_brown   = test_model_Ridge(large_filt_brown_spill, br_y, 10, "brown_large")
m_xl_brown, sd_xl_brown       = test_model_Ridge(xl_filt_brown_spill, br_y, 10, "brown_xl")
print(m_gpt_brown, m_small_brown, m_med_brown, m_larg_brown, m_xl_brown)

#################
# BASELINE ONLY #
#################

# natstor, baseline only
m_gpt_natstor_b, sd_gpt_natstor_b     = test_model_Ridge(np.array(ns_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]]), ns_y, 10, 
                                                   "natstor_gpt_base")
m_small_natstor_b, sd_small_natstor_b = test_model_Ridge(np.array(ns_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2"]]), ns_y, 10, 
                                                   "natstor_small_base")
m_med_natstor_b, sd_med_natstor_b     = test_model_Ridge(np.array(ns_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2"]]), ns_y, 10, 
                                                   "natstor_med_base")
m_larg_natstor_b, sd_larg_natstor_b   = test_model_Ridge(np.array(ns_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2"]]), ns_y, 10, 
                                                   "natstor_large_base")
m_xl_natstor_b, sd_xl_natstor_b       = test_model_Ridge(np.array(ns_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2"]]), ns_y, 10, 
                                                   "natstor_xl_base")
print(m_gpt_natstor_b, m_small_natstor_b, m_med_natstor_b, m_larg_natstor_b, m_xl_natstor_b)

# decompose_sfl

decompose_sfl(np.array(ns_spill[["s_GPT", "logfreq", "length",
                              "s_GPT-1", "logfreq-1", "length-1",
                              "s_GPT-2", "logfreq-2", "length-2"]]), ns_y, "natstor_gpt_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(ns_spill[["s_GPT2_small", "logfreq", "length",
                              "s_GPT2_small-1", "logfreq-1", "length-1",
                              "s_GPT2_small-2", "logfreq-2", "length-2"]]), ns_y, "natstor_small_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(ns_spill[["s_GPT2_med", "logfreq", "length",
                              "s_GPT2_med-1", "logfreq-1", "length-1",
                              "s_GPT2_med-2", "logfreq-2", "length-2"]]), ns_y, "natstor_med_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(ns_spill[["s_GPT2_large", "logfreq", "length",
                              "s_GPT2_large-1", "logfreq-1", "length-1",
                              "s_GPT2_large-2", "logfreq-2", "length-2"]]), ns_y, "natstor_large_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(ns_spill[["s_GPT2_xl", "logfreq", "length",
                              "s_GPT2_xl-1", "logfreq-1", "length-1",
                              "s_GPT2_xl-2", "logfreq-2", "length-2"]]), ns_y, "natstor_xl_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])


# brown, baseline only
m_gpt_brown_b, sd_gpt_brown_b     = test_model_Ridge(np.array(brown_spill[["s_GPT", "logfreq", "length",
                                                                     "s_GPT-1", "logfreq-1", "length-1",
                                                                     "s_GPT-2", "logfreq-2", "length-2"]]), br_y, 10, 
                                                  "brown_gpt_base")
m_small_brown_b, sd_small_brown_b = test_model_Ridge(np.array(brown_spill[["s_GPT2_small", "logfreq", "length",
                                                                     "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                     "s_GPT2_small-2", "logfreq-2", "length-2"]]), br_y, 10, 
                                                  "brown_small_base")
m_med_brown_b, sd_med_brown_b     = test_model_Ridge(np.array(brown_spill[["s_GPT2_med", "logfreq", "length",
                                                                     "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                     "s_GPT2_med-2", "logfreq-2", "length-2"]]), br_y, 10, 
                                                  "brown_med_base")
m_larg_brown_b, sd_larg_brown_b   = test_model_Ridge(np.array(brown_spill[["s_GPT2_large", "logfreq", "length",
                                                                     "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                     "s_GPT2_large-2", "logfreq-2", "length-2"]]), br_y, 10, 
                                                  "brown_large_base")
m_xl_brown_b, sd_xl_brown_b       = test_model_Ridge(np.array(brown_spill[["s_GPT2_xl", "logfreq", "length",
                                                                     "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                     "s_GPT2_xl-2", "logfreq-2", "length-2"]]), br_y, 10, 
                                                  "brown_xl_base")
print(m_gpt_brown_b, m_small_brown_b, m_med_brown_b, m_larg_brown_b, m_xl_brown_b)

# decompose_sfl
decompose_sfl(np.array(brown_spill[["s_GPT", "logfreq", "length",
                                 "s_GPT-1", "logfreq-1", "length-1",
                                 "s_GPT-2", "logfreq-2", "length-2"]]), br_y, "brown_gpt_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(brown_spill[["s_GPT2_small", "logfreq", "length",
                                 "s_GPT2_small-1", "logfreq-1", "length-1",
                                 "s_GPT2_small-2", "logfreq-2", "length-2"]]), br_y, "brown_small_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(brown_spill[["s_GPT2_med", "logfreq", "length",
                                 "s_GPT2_med-1", "logfreq-1", "length-1",
                                 "s_GPT2_med-2", "logfreq-2", "length-2"]]), br_y, "brown_med_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(brown_spill[["s_GPT2_large", "logfreq", "length",
                                 "s_GPT2_large-1", "logfreq-1", "length-1",
                                 "s_GPT2_large-2", "logfreq-2", "length-2"]]), br_y, "brown_large_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])
decompose_sfl(np.array(brown_spill[["s_GPT2_xl", "logfreq", "length",
                                 "s_GPT2_xl-1", "logfreq-1", "length-1",
                                 "s_GPT2_xl-2", "logfreq-2", "length-2"]]), br_y, "brown_xl_base", idx_s = [0,3,6], idx_f = [1,4,7], idx_l = [2,5,8])

##########################
# BASELINE + ACTIVATIONS #
##########################

gpt_all_natstor        = np.concatenate((np.array(ns_spill[["s_GPT", "logfreq", "length",
                                                          "s_GPT-1", "logfreq-1", "length-1",
                                                          "s_GPT-2", "logfreq-2", "length-2"]]), gpt_filt_ns_spill), axis=1)
gpt2_small_all_natstor = np.concatenate((np.array(ns_spill[["s_GPT2_small", "logfreq", "length",
                                                          "s_GPT2_small-1", "logfreq-1", "length-1",
                                                          "s_GPT2_small-2", "logfreq-2", "length-2"]]), small_filt_ns_spill), axis=1)
gpt2_med_all_natstor   = np.concatenate((np.array(ns_spill[["s_GPT2_med", "logfreq", "length",
                                                          "s_GPT2_med-1", "logfreq-1", "length-1",
                                                          "s_GPT2_med-2", "logfreq-2", "length-2"]]), med_filt_ns_spill), axis=1)
gpt2_large_all_natstor = np.concatenate((np.array(ns_spill[["s_GPT2_large", "logfreq", "length",
                                                          "s_GPT2_large-1", "logfreq-1", "length-1",
                                                          "s_GPT2_large-2", "logfreq-2", "length-2"]]), large_filt_ns_spill), axis=1)
gpt2_xl_all_natstor    = np.concatenate((np.array(ns_spill[["s_GPT2_xl", "logfreq", "length",
                                                          "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                          "s_GPT2_xl-2", "logfreq-2", "length-2"]]), xl_filt_ns_spill), axis=1)

gpt_all_brown        = np.concatenate((np.array(brown_spill[["s_GPT", "logfreq", "length",
                                                           "s_GPT-1", "logfreq-1", "length-1",
                                                           "s_GPT", "logfreq", "length"]]), gpt_filt_brown_spill), axis=1)
gpt2_small_all_brown = np.concatenate((np.array(brown_spill[["s_GPT2_small", "logfreq", "length",
                                                           "s_GPT2_small-1", "logfreq-1", "length-1",
                                                           "s_GPT2_small", "logfreq", "length"]]), small_filt_brown_spill), axis=1)
gpt2_med_all_brown   = np.concatenate((np.array(brown_spill[["s_GPT2_med", "logfreq", "length",
                                                           "s_GPT2_med-1", "logfreq-1", "length-1",
                                                           "s_GPT2_med", "logfreq", "length"]]), med_filt_brown_spill), axis=1)
gpt2_large_all_brown = np.concatenate((np.array(brown_spill[["s_GPT2_large", "logfreq", "length",
                                                           "s_GPT2_large-1", "logfreq-1", "length-1",
                                                           "s_GPT2_large", "logfreq", "length"]]), large_filt_brown_spill), axis=1)
gpt2_xl_all_brown    = np.concatenate((np.array(brown_spill[["s_GPT2_xl", "logfreq", "length",
                                                           "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                           "s_GPT2_xl", "logfreq", "length"]]), xl_filt_brown_spill), axis=1)

# all, natstor
m_gpt_natstor_all, sd_gpt_natstor_all     = test_model_Ridge(gpt_all_natstor, ns_y, 10, "natstor_gpt_all")
m_small_natstor_all, sd_small_natstor_all = test_model_Ridge(gpt2_small_all_natstor, ns_y, 10, "natstor_small_all")
m_med_natstor_all, sd_med_natstor_all     = test_model_Ridge(gpt2_med_all_natstor, ns_y, 10, "natstor_med_all")
m_larg_natstor_all, sd_larg_natstor_all   = test_model_Ridge(gpt2_large_all_natstor, ns_y, 10, "natstor_large_all")
m_xl_natstor_all, sd_xl_natstor_all       = test_model_Ridge(gpt2_xl_all_natstor, ns_y, 10, "natstor_xl_all")
print(m_gpt_natstor_all, m_small_natstor_all, m_med_natstor_all, m_larg_natstor_all, m_xl_natstor_all)


# all, brown
m_gpt_brown_all, sd_gpt_brown_all     = test_model_Ridge(gpt_all_brown, br_y, 10, "brown_gpt_all")
m_small_brown_all, sd_small_brown_all = test_model_Ridge(gpt2_small_all_brown, br_y, 10, "brown_small_all")
m_med_brown_all, sd_med_brown_all     = test_model_Ridge(gpt2_med_all_brown, br_y, 10, "brown_med_all")
m_larg_brown_all, sd_larg_brown_all   = test_model_Ridge(gpt2_large_all_brown, br_y, 10, "brown_large_all")
m_xl_brown_all, sd_xl_brown_all       = test_model_Ridge(gpt2_xl_all_brown, br_y, 10, "brown_xl_all")
print(m_gpt_brown_all, m_small_brown_all, m_med_brown_all, m_larg_brown_all, m_xl_brown_all)


df_results = pd.DataFrame({
    "base_natstor":[m_gpt_natstor_b, m_small_natstor_b, m_med_natstor_b, m_larg_natstor_b, m_xl_natstor_b],
    "base_natstor_sd": [sd_gpt_natstor_b, sd_small_natstor_b, sd_med_natstor_b, sd_larg_natstor_b, sd_xl_natstor_b],
    "base_brown":  [m_gpt_brown_b, m_small_brown_b, m_med_brown_b, m_larg_brown_b, m_xl_brown_b],
    "base_brown_sd":  [sd_gpt_brown_b, sd_small_brown_b, sd_med_brown_b, sd_larg_brown_b, sd_xl_brown_b],
    "act_natstor":[m_gpt_natstor, m_small_natstor, m_med_natstor, m_larg_natstor, m_xl_natstor],
    "act_natstor_sd": [sd_gpt_natstor, sd_small_natstor, sd_med_natstor, sd_larg_natstor, sd_xl_natstor],
    "act_brown":  [m_gpt_brown, m_small_brown, m_med_brown, m_larg_brown, m_xl_brown],
    "act_brown_sd":  [sd_gpt_brown, sd_small_brown, sd_med_brown, sd_larg_brown, sd_xl_brown],
    "all_natstor":[m_gpt_natstor_all, m_small_natstor_all, m_med_natstor_all, m_larg_natstor_all, m_xl_natstor_all],
    "all_natstor_sd":[sd_gpt_natstor_all, sd_small_natstor_all, sd_med_natstor_all, sd_larg_natstor_all, sd_xl_natstor_all],
    "all_brown":[m_gpt_brown_all, m_small_brown_all, m_med_brown_all, m_larg_brown_all, m_xl_brown_all],
    "all_brown_sd":[sd_gpt_brown_all, sd_small_brown_all, sd_med_brown_all, sd_larg_brown_all, sd_xl_brown_all]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_results.to_csv("results_natstor_brown_spill.csv")
df_results = pd.read_csv("results_natstor_brown_spill.csv")

base_plot(df_results,"act_brown", "base_brown", "all_brown")
base_plot(df_results,"act_natstor", "base_natstor", "all_natstor")

#####################
# stat significance #
#####################

thebase_natstor = ["natstor_gpt_base", "natstor_small_base", "natstor_med_base", "natstor_large_base", "natstor_xl_base"]
theemb_natstor  = ["natstor_gpt", "natstor_small", "natstor_med", "natstor_large", "natstor_xl"]
all_natstor     = ["natstor_gpt_all", "natstor_small_all", "natstor_med_all", "natstor_large_all", "natstor_xl_all"]

p_natstor = []
for base, emb, all_ in zip(thebase_natstor, theemb_natstor, all_natstor):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_natstor.append([p_base_emb, p_all_base, p_all_emb])
p_natstor = pd.DataFrame(p_natstor, columns=["base_emb", "all_base", "all_emb"])

#############################################

thebase_brown = ["brown_gpt_base", "brown_small_base", "brown_med_base", "brown_large_base", "brown_xl_base"]
theemb_brown  = ["brown_gpt", "brown_small", "brown_med", "brown_large", "brown_xl"]
all_brown     = ["brown_gpt_all", "brown_small_all", "brown_med_all", "brown_large_all", "brown_xl_all"]

p_brown = []
for base, emb, all_ in zip(thebase_brown, theemb_brown, all_brown):
    p_base_emb = significance_testing(base, emb)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_brown.append([p_base_emb, p_all_base, p_all_emb])
p_brown = pd.DataFrame(p_brown, columns=["base_emb", "all_base", "all_emb"])


for df, dfname in zip([p_natstor, p_brown], ["natstor", "brown"]):
    for df_column in df.columns:
        df_results["p_"+dfname+"_"+df_column] = list(df[df_column])
        



#################################
# FOLLOW-UP: SEMANTIC VARIABLES #
#################################

from utils import plot_sem, get_sem, plot_weights_sem

brown_words  = [t for w in brown_sent for t in w[1:]]
natst_words  = [t for w in natst_sent for t in w[1:]]

sem_brown = get_sem(brown_words)
# Missing affective = 74.73
# Missing perceptual = 21.59
# dump(sem_brown, "SEM/sem_brown.joblib") 
sem_brown = load("SEM/sem_brown.joblib")

sem_natst = get_sem(natst_words)
# Missing affective = 73.04
# Missing perceptual = 15.72
#dump(sem_natst, "SEM/sem_natst.joblib")
sem_natst = load("SEM/sem_natst.joblib")

sem_brown_spill   = filter_embeddings_spill(brown_spr, sem_brown)
sem_natst_spill   = filter_embeddings_spill(ns_spr, sem_natst)

# embedding
gpt_sem_brown   = np.concatenate((sem_brown_spill, gpt_filt_brown_spill), axis=1)
small_sem_brown = np.concatenate((sem_brown_spill, small_filt_brown_spill), axis=1)
med_sem_brown   = np.concatenate((sem_brown_spill, med_filt_brown_spill), axis=1)
large_sem_brown = np.concatenate((sem_brown_spill, large_filt_brown_spill), axis=1)
xl_sem_brown    = np.concatenate((sem_brown_spill, xl_filt_brown_spill), axis=1)

gpt_sem_natst     = np.concatenate((sem_natst_spill, gpt_filt_ns_spill), axis=1)
small_sem_natst   = np.concatenate((sem_natst_spill, small_filt_ns_spill), axis=1)
med_sem_natst     = np.concatenate((sem_natst_spill, med_filt_ns_spill), axis=1)
large_sem_natst   = np.concatenate((sem_natst_spill, large_filt_ns_spill), axis=1)
xl_sem_natst      = np.concatenate((sem_natst_spill, xl_filt_ns_spill), axis=1)

# baseline
gpt_base_sem_brown = np.concatenate((sem_brown_spill, np.array(brown_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]])), axis=1)
small_base_sem_brown = np.concatenate((sem_brown_spill, np.array(brown_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2",]])), axis=1)
med_base_sem_brown = np.concatenate((sem_brown_spill, np.array(brown_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2",]])), axis=1)
large_base_sem_brown = np.concatenate((sem_brown_spill, np.array(brown_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2",]])), axis=1)
xl_base_sem_brown = np.concatenate((sem_brown_spill, np.array(brown_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2",]])), axis=1)

gpt_base_sem_natst = np.concatenate((sem_natst_spill, np.array(ns_spill[["s_GPT", "logfreq", "length",
                                                                      "s_GPT-1", "logfreq-1", "length-1",
                                                                      "s_GPT-2", "logfreq-2", "length-2",]])), axis=1)
small_base_sem_natst = np.concatenate((sem_natst_spill, np.array(ns_spill[["s_GPT2_small", "logfreq", "length",
                                                                      "s_GPT2_small-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_small-2", "logfreq-2", "length-2",]])), axis=1)
med_base_sem_natst = np.concatenate((sem_natst_spill, np.array(ns_spill[["s_GPT2_med", "logfreq", "length",
                                                                      "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_med-2", "logfreq-2", "length-2",]])), axis=1)
large_base_sem_natst = np.concatenate((sem_natst_spill, np.array(ns_spill[["s_GPT2_large", "logfreq", "length",
                                                                      "s_GPT2_large-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_large-2", "logfreq-2", "length-2",]])), axis=1)
xl_base_sem_natst = np.concatenate((sem_natst_spill, np.array(ns_spill[["s_GPT2_xl", "logfreq", "length",
                                                                      "s_GPT2_xl-1", "logfreq-1", "length-1",
                                                                      "s_GPT2_xl-2", "logfreq-2", "length-2",]])), axis=1)

# all
gpt_all_sem_brown   = np.concatenate((gpt_base_sem_brown, gpt_filt_brown_spill), axis=1)
small_all_sem_brown = np.concatenate((small_base_sem_brown, small_filt_brown_spill), axis=1)
med_all_sem_brown   = np.concatenate((med_base_sem_brown, med_filt_brown_spill), axis=1)
large_all_sem_brown = np.concatenate((large_base_sem_brown, large_filt_brown_spill), axis=1)
xl_all_sem_brown    = np.concatenate((xl_base_sem_brown, xl_filt_brown_spill), axis=1)

gpt_all_sem_natst   = np.concatenate((gpt_base_sem_natst, gpt_filt_ns_spill), axis=1)
small_all_sem_natst = np.concatenate((small_base_sem_natst, small_filt_ns_spill), axis=1)
med_all_sem_natst   = np.concatenate((med_base_sem_natst, med_filt_ns_spill), axis=1)
large_all_sem_natst = np.concatenate((large_base_sem_natst, large_filt_ns_spill), axis=1)
xl_all_sem_natst    = np.concatenate((xl_base_sem_natst, xl_filt_ns_spill), axis=1)

#################
# MODEL TESTING #
#################

m_gpt_natstor, sd_gpt_natstor     = test_model_Ridge(gpt_sem_natst, ns_y, 10, "natstor_gpt_sem")
m_small_natstor, sd_small_natstor = test_model_Ridge(small_sem_natst, ns_y, 10, "natstor_small_sem")
m_med_natstor, sd_med_natstor     = test_model_Ridge(med_sem_natst, ns_y, 10, "natstor_med_sem")
m_larg_natstor, sd_larg_natstor   = test_model_Ridge(large_sem_natst, ns_y, 10, "natstor_large_sem")
m_xl_natstor, sd_xl_natstor       = test_model_Ridge(xl_sem_natst, ns_y, 10, "natstor_xl_sem") #########################
print(m_gpt_natstor, m_small_natstor, m_med_natstor, m_larg_natstor, m_xl_natstor)


# brown
m_gpt_brown, sd_gpt_brown     = test_model_Ridge(gpt_sem_brown, br_y, 10, "brown_gpt_sem")
m_small_brown, sd_small_brown = test_model_Ridge(small_sem_brown, br_y, 10, "brown_small_sem")
m_med_brown, sd_med_brown     = test_model_Ridge(med_sem_brown, br_y, 10, "brown_med_sem")
m_larg_brown, sd_larg_brown   = test_model_Ridge(large_sem_brown, br_y, 10, "brown_large_sem")
m_xl_brown, sd_xl_brown       = test_model_Ridge(xl_sem_brown, br_y, 10, "brown_xl_sem")
print(m_gpt_brown, m_small_brown, m_med_brown, m_larg_brown, m_xl_brown)

#################
# BASELINE ONLY #
#################

# natstor, baseline only
m_gpt_natstor_b, sd_gpt_natstor_b     = test_model_Ridge(gpt_base_sem_natst, ns_y, 10, "natstor_gpt_base_sem")
m_small_natstor_b, sd_small_natstor_b = test_model_Ridge(small_base_sem_natst, ns_y, 10, "natstor_small_base_sem")
m_med_natstor_b, sd_med_natstor_b     = test_model_Ridge(med_base_sem_natst, ns_y, 10, "natstor_med_base_sem")
m_larg_natstor_b, sd_larg_natstor_b   = test_model_Ridge(large_base_sem_natst, ns_y, 10, "natstor_large_base_sem")
m_xl_natstor_b, sd_xl_natstor_b       = test_model_Ridge(xl_base_sem_natst, ns_y, 10, "natstor_xl_base_sem")
print(m_gpt_natstor_b, m_small_natstor_b, m_med_natstor_b, m_larg_natstor_b, m_xl_natstor_b)

# brown, baseline only
m_gpt_brown_b, sd_gpt_brown_b     = test_model_Ridge(gpt_base_sem_brown, br_y, 10, "brown_gpt_base_sem")
m_small_brown_b, sd_small_brown_b = test_model_Ridge(small_base_sem_brown, br_y, 10, "brown_small_base_sem")
m_med_brown_b, sd_med_brown_b     = test_model_Ridge(med_base_sem_brown, br_y, 10, "brown_med_base_sem")
m_larg_brown_b, sd_larg_brown_b   = test_model_Ridge(large_base_sem_brown, br_y, 10, "brown_large_base_sem")
m_xl_brown_b, sd_xl_brown_b       = test_model_Ridge(xl_base_sem_brown, br_y, 10, "brown_xl_base_sem")
print(m_gpt_brown_b, m_small_brown_b, m_med_brown_b, m_larg_brown_b, m_xl_brown_b)

##########################
# BASELINE + ACTIVATIONS #
##########################

# all, natstor
m_gpt_natstor_all, sd_gpt_natstor_all     = test_model_Ridge(gpt_all_sem_natst, ns_y, 10, "natstor_gpt_all_sem")
m_small_natstor_all, sd_small_natstor_all = test_model_Ridge(small_all_sem_natst, ns_y, 10, "natstor_small_all_sem")
m_med_natstor_all, sd_med_natstor_all     = test_model_Ridge(med_all_sem_natst, ns_y, 10, "natstor_med_all_sem")
m_larg_natstor_all, sd_larg_natstor_all   = test_model_Ridge(large_all_sem_natst, ns_y, 10, "natstor_large_all_sem")
m_xl_natstor_all, sd_xl_natstor_all       = test_model_Ridge(xl_all_sem_natst, ns_y, 10, "natstor_xl_all_sem")
print(m_gpt_natstor_all, m_small_natstor_all, m_med_natstor_all, m_larg_natstor_all, m_xl_natstor_all)


# all, brown
m_gpt_brown_all, sd_gpt_brown_all     = test_model_Ridge(gpt_all_sem_brown, br_y, 10, "brown_gpt_all_sem")
m_small_brown_all, sd_small_brown_all = test_model_Ridge(small_all_sem_brown, br_y, 10, "brown_small_all_sem")
m_med_brown_all, sd_med_brown_all     = test_model_Ridge(med_all_sem_brown, br_y, 10, "brown_med_all_sem")
m_larg_brown_all, sd_larg_brown_all   = test_model_Ridge(large_all_sem_brown, br_y, 10, "brown_large_all_sem")
m_xl_brown_all, sd_xl_brown_all       = test_model_Ridge(xl_all_sem_brown, br_y, 10, "brown_xl_all_sem")
print(m_gpt_brown_all, m_small_brown_all, m_med_brown_all, m_larg_brown_all, m_xl_brown_all)


df_results_sem = pd.DataFrame({
    "base_natstor_sem":[m_gpt_natstor_b, m_small_natstor_b, m_med_natstor_b, m_larg_natstor_b, m_xl_natstor_b],
    "base_natstor_sem_sd": [sd_gpt_natstor_b, sd_small_natstor_b, sd_med_natstor_b, sd_larg_natstor_b, sd_xl_natstor_b],
    "base_brown_sem":  [m_gpt_brown_b, m_small_brown_b, m_med_brown_b, m_larg_brown_b, m_xl_brown_b],
    "base_brown_sem_sd":  [sd_gpt_brown_b, sd_small_brown_b, sd_med_brown_b, sd_larg_brown_b, sd_xl_brown_b],
    "act_natstor_sem":[m_gpt_natstor, m_small_natstor, m_med_natstor, m_larg_natstor, m_xl_natstor],
    "act_natstor_sem_sd": [sd_gpt_natstor, sd_small_natstor, sd_med_natstor, sd_larg_natstor, sd_xl_natstor],
    "act_brown_sem":  [m_gpt_brown, m_small_brown, m_med_brown, m_larg_brown, m_xl_brown],
    "act_brown_sem_sd":  [sd_gpt_brown, sd_small_brown, sd_med_brown, sd_larg_brown, sd_xl_brown],
    "all_natstor_sem":[m_gpt_natstor_all, m_small_natstor_all, m_med_natstor_all, m_larg_natstor_all, m_xl_natstor_all],
    "all_natstor_sem_sd":[sd_gpt_natstor_all, sd_small_natstor_all, sd_med_natstor_all, sd_larg_natstor_all, sd_xl_natstor_all],
    "all_brown_sem":[m_gpt_brown_all, m_small_brown_all, m_med_brown_all, m_larg_brown_all, m_xl_brown_all],
    "all_brown_sem_sd":[sd_gpt_brown_all, sd_small_brown_all, sd_med_brown_all, sd_larg_brown_all, sd_xl_brown_all]
    })
df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_results_sem.to_csv("results_natstor_brown_spill_sem.csv")

df_results = pd.read_csv("results_natstor_brown_spill.csv")
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
merged = pd.concat([df_results, df_results_sem], axis=1)
#merged.to_csv("SEM/merged_natstor_brown.csv")

plot_sem(merged, "act_natstor", "base_natstor", "all_natstor", "act_natstor_sem", "base_natstor_sem", "all_natstor_sem", title="NatStor", subtract=True)

plot_sem(merged, "act_brown", "base_brown", "all_brown", "act_brown_sem", "base_brown_sem", "all_brown_sem", title="brown", subtract=True)


########################
# for last probing exp #
########################

probe(med_filt_ns_spill, ns_y, 10, "natstor_med_probing")
probe(med_filt_brown_spill, br_y, 10, "brown_med_probing")


# from joblib import load
# pca = load('probing_LMs/pca.pkl')

# med_filt_brown_pca = pca.transform(med_filt_brown)
# med_filt_natstor_pca = pca.transform(med_filt_natstor)

# med_filt_brown_spill_pca   = filter_embeddings_spill(brown_spr, med_filt_brown_pca)
# med_filt_ns_spill_pca   = filter_embeddings_spill(ns_spr, med_filt_natstor_pca)

# m_med_natstor, sd_med_natstor     = test_model_Ridge(med_filt_ns_spill_pca, ns_y, 10, "natstor_med_probing_pca")
# print(m_med_natstor)
# m_med_brown, sd_med_brown     = test_model_Ridge(med_filt_brown_spill_pca, br_y, 10, "brown_med_probing_pca")
# print(m_med_brown)


##########################
# end of weight analysis #
##########################
# save med_filt, y
dataname = "natstor_spr"
tosave = [med_filt_ns_spill, ns_y]
# with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
dataname = "brown"
tosave = [med_filt_brown_spill, br_y]
# with open(f"probing_LMs/XYdata/{dataname}", 'wb') as handle:  # [removed: probing_LMs sub-study]
#     pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)