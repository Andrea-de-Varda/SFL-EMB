import numpy as np
import pandas as pd
from os import chdir, path
from scipy.stats import zscore
from tqdm import tqdm
import pickle
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from scipy.special import softmax
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost

dirName = "/home/dev/Documents/PhD/MIT/project/fMRI"
chdir(dirName)

parentdir = path.dirname(dirName)
sys.path.insert(0, parentdir) 

from utils import test_model_Ridge, base_plot, significance_testing, decompose_sfl, probe

SKIP_WORDS=20; END_WORDS=5176

words = np.load('fMRI/words_fmri.npy') #                               (5176,)
# s_wor = np.load('fMRI/stimuli_words.npy') #                          (5176,)
time = np.load('fMRI/time_fmri.npy') # [0,2,4,6,8,10 ... 2700]         (1351,)
runs = np.load('fMRI/runs_fmri.npy') # [1,1,1 ... 2,2,2 ... 4,4,4]     (1351,)
time_words = np.load('fMRI/time_words_fmri.npy') # [20,20.5,21 ..., 2692,2692.5,2693]  (5176,)
time_words = time_words[SKIP_WORDS:END_WORDS]    #                     (5156,)

part_f = np.load("fMRI/data_subject_F.npy") #                          (1211, 27905)
part_h = np.load("fMRI/data_subject_H.npy")
part_i = np.load("fMRI/data_subject_I.npy")
part_j = np.load("fMRI/data_subject_J.npy")
part_k = np.load("fMRI/data_subject_K.npy")
part_l = np.load("fMRI/data_subject_L.npy")
part_m = np.load("fMRI/data_subject_M.npy")
part_n = np.load("fMRI/data_subject_N.npy")

# The first 20TRs have been removed, as well as the last 15TR (tot = 35). 1211+(35*4)

words_id = np.zeros([len(time_words)]) # 5156
# w=find what TR each word belongs to; then I'll need to aggregate representations
for i in range(len(time_words)):
    words_id[i] = np.where(time_words[i]> time)[0][-1]
# I will need to aggregate SFL and EMB representations accordingly (see [1])

sentences = [[w for w in l.split(" _#_ ")+["+"] if w] for l in " _#_ ".join(list(words)).split("+")]
sentences = [" ".join(s) for s in sentences[:-1]]
sum([len(s.split()) for s in sentences]) # 5176, ok

#####################################################
# grouping sentences into passages (longer context) #
#####################################################

def divide_list(lst, sublist_length):
    sublists = [lst[i:i+sublist_length] for i in range(0, len(lst), sublist_length)]
    return [" ".join(lst) for lst in sublists]

def clean_string(string):
    # substituting out-of-vocabulary words
    str1 = re.sub("…", "...", string)
    str2 = re.sub("—", "-", str1)
    str3 = re.sub("‘", "'", str2)
    return str3

sentences = divide_list(sentences, 5) # context of 5 sentences!
sum([len(s.split()) for s in sentences]) # 5176, ok

# with open("sentences.txt", 'w') as fp:
#     for item in sentences:
#         item = clean_string(item)
#         fp.write("%s\n" % item)
#     print('Done')

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
if path.isfile('HP_emb/activations_gpt.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        "sentences.txt",
        'HP_emb/activations_gpt.json',
        aggregation="average"
    )

################
# GPT-2 FAMILY #
################

# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile('HP_emb/activations_gpt2_small.json') == False:
    transformers_extractor.extract_representations('gpt2',
        "sentences.txt",
        'HP_emb/activations_gpt2_small.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile('HP_emb/activations_gpt2_medium.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        "sentences.txt",
        'HP_emb/activations_gpt2_medium.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile('HP_emb/activations_gpt2_large.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        "sentences.txt",
        'HP_emb/activations_gpt2_large.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile('HP_emb/activations_gpt2_xl.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        "sentences.txt",
        'HP_emb/activations_gpt2_xl.json',
        aggregation="average"
    )


act_gpt, num_layers_gpt = data_loader.load_activations('HP_emb/activations_gpt.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations('HP_emb/activations_gpt2_small.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations('HP_emb/activations_gpt2_medium.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations('HP_emb/activations_gpt2_large.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations('HP_emb/activations_gpt2_xl.json', 1600, is_brnn=False)

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act(act_list, layer, dim):
    out = []
    for i in range(len(act_list)):
        act_sent = act_list[i]
        out.extend([get_layer(v, layer, dim) for v in act_sent])
    return np.array(out)

gpt_filt   = get_act(act_gpt, 12, 768)
small_filt = get_act(act_gpt2_small, 12, 768)
med_filt   = get_act(act_gpt2_medium, 24, 1024)
large_filt = get_act(act_gpt2_large, 36, 1280)
xl_filt    = get_act(act_gpt2_xl, 48, 1600)

# [1]

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def delay_one(mat, d):
        # delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
        # delays a matrix by a set of delays d.
        # a row t in the returned matrix has the concatenated:
        # row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat      # (1351,40)

def embed_words(embeddings):
    ids = words_id.astype(int)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[20:][ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    tmp = delay_mat(emb_words, np.arange(1,5)) #  (1351, 3072) --> spillover
    # remove the edges of each run
    #tmp = np.vstack([zscore(tmp[runs==i][20:-15]) for i in range(1,5)]) # (1211, 40) ==> IMPORTANT how 1351 --> 1211
    tmp = np.vstack([tmp[runs==i][20:-15] for i in range(1,5)]) # (1211, 40) ==> IMPORTANT how 1351 --> 1211
    #tmp = np.nan_to_num(tmp) # changed this from Toneva - makes more sense to imputate
    tmp = imputate_na(tmp)
    return tmp

gpt_X   = embed_words(gpt_filt)
small_X = embed_words(small_filt)
med_X   = embed_words(med_filt)
large_X = embed_words(large_filt)
xl_X    = embed_words(xl_filt)

#######
# SFL #
#######

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

sent = [re.sub("\n", "", s).split() for s in open("sentences.txt").readlines()] 

freq = [get_f(word) for s in sent for word in s]
length = [len(word) for s in sent for word in s]

# GPT
model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "</w>")] 
# adding np.nan to keep track of beginning-of-sentence tokens

# GPT2 SMALL (124M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2")
s_small = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 medium
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-medium")
s_med = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 large
model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-large")
s_large = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# GPT2 xl
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# with open('SFL/gpt', 'wb') as handle:
#     pickle.dump(s_gpt, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('SFL/small', 'wb') as handle:
#     pickle.dump(s_small, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('SFL/med', 'wb') as handle:
#     pickle.dump(s_med, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('SFL/large', 'wb') as handle:
#     pickle.dump(s_large, handle, protocol=pickle.HIGHEST_PROTOCOL) 
# with open('SFL/xl', 'wb') as handle:
#     pickle.dump(s_xl, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('SFL/gpt', 'rb') as handle:
    s_gpt = pickle.load(handle)
with open('SFL/small', 'rb') as handle:
    s_small = pickle.load(handle)
with open('SFL/med', 'rb') as handle:
    s_med = pickle.load(handle)
with open('SFL/large', 'rb') as handle:
    s_large = pickle.load(handle)
with open('SFL/xl', 'rb') as handle:
    s_xl = pickle.load(handle)
    
#########
    
sfl_gpt = np.stack([np.array(s_gpt), np.array(freq), np.array(length)], axis=1)
filt = ~np.isnan(sfl_gpt.mean(axis=1))
#sfl_gpt_filt = sfl_gpt[filt]
sfl_gpt_X = embed_words(sfl_gpt)

sfl_small = np.stack([np.array(s_small), np.array(freq), np.array(length)], axis=1)
sfl_small_X = embed_words(sfl_small)

sfl_med = np.stack([np.array(s_med), np.array(freq), np.array(length)], axis=1)
sfl_med_X = embed_words(sfl_med)

sfl_large = np.stack([np.array(s_large), np.array(freq), np.array(length)], axis=1)
sfl_large_X = embed_words(sfl_large)

sfl_xl = np.stack([np.array(s_xl), np.array(freq), np.array(length)], axis=1)
sfl_xl_X = embed_words(sfl_xl)

#############################
# predicting BRAIN ACTIVITY #
#############################

# Important: differently from previous analyses, using Ridge regression (consistent with previous literature,
# also works better)
# IMPORTANT --> try with different penalty values

#from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

roi = np.load('fMRI/HP_subj_roi_inds.npy', allow_pickle=True)
roi = roi.tolist()

parts = [part_f, part_h, part_i, part_j, part_k, part_l, part_m, part_n]
partn = ["F", "H", "I", "J", "K", "L", "M", "N"]
roin =  ["all", "PostTemp", "AntTemp", "AngularG", "IFG", "MFG", "IFGorb", "pCingulate"]

######################################################
# averaging activity in each ROI across participants #
######################################################

brain_data_avg = {}
for the_roi in roin:
    out = []
    for part_act, part_name in zip(parts, partn):
        part = part_act[:,roi[part_name][the_roi]].mean(axis=1)
        out.append(part)
    out = np.array(out).mean(axis=0)
    brain_data_avg[the_roi] = out
    
results = {}
for roi in roin:
    print("\n\n\nProcessing", roi)
    # baseline
    base_gpt, base_gpt_sd = test_model_Ridge(sfl_gpt_X, brain_data_avg[roi], 10, "hp_"+roi+"_gpt_base")
    base_small, base_small_sd = test_model_Ridge(sfl_small_X, brain_data_avg[roi], 10, "hp_"+roi+"_small_base")
    base_med, base_med_sd = test_model_Ridge(sfl_med_X, brain_data_avg[roi], 10, "hp_"+roi+"_med_base")
    base_large, base_large_sd = test_model_Ridge(sfl_large_X, brain_data_avg[roi], 10, "hp_"+roi+"_large_base")
    base_xl, base_xl_sd = test_model_Ridge(sfl_xl_X, brain_data_avg[roi], 10, "hp_"+roi+"_xl_base")
    print(base_gpt, base_small, base_med, base_large, base_xl)
    
    # act
    gpt, gpt_sd = test_model_Ridge(gpt_X, brain_data_avg[roi], 10, "hp_"+roi+"_gpt")
    small, small_sd = test_model_Ridge(small_X, brain_data_avg[roi], 10, "hp_"+roi+"_small")
    med, med_sd = test_model_Ridge(med_X, brain_data_avg[roi], 10, "hp_"+roi+"_med")
    large, large_sd = test_model_Ridge(large_X, brain_data_avg[roi], 10, "hp_"+roi+"_large")
    xl, xl_sd = test_model_Ridge(xl_X, brain_data_avg[roi], 10, "hp_"+roi+"_xl")
    print(gpt, small, med, large, xl)
    
    # all
    gpt_a, gpt_a_sd = test_model_Ridge(np.concatenate([gpt_X, sfl_gpt_X], axis=1), brain_data_avg[roi], 10, "hp_"+roi+"_gpt_all")
    small_a, small_a_sd = test_model_Ridge(np.concatenate([small_X, sfl_small_X], axis=1), brain_data_avg[roi], 10, "hp_"+roi+"_small_all")
    med_a, med_a_sd = test_model_Ridge(np.concatenate([med_X, sfl_med_X], axis=1), brain_data_avg[roi], 10, "hp_"+roi+"_med_all")
    large_a, large_a_sd = test_model_Ridge(np.concatenate([large_X, sfl_large_X], axis=1), brain_data_avg[roi], 10, "hp_"+roi+"_large_all")
    xl_a, xl_a_sd = test_model_Ridge(np.concatenate([xl_X, sfl_xl_X], axis=1), brain_data_avg[roi], 10, "hp_"+roi+"_xl_all")
    print(gpt_a, small_a, med_a, large_a, xl_a)
    
    df_results = pd.DataFrame({
        "base":[base_gpt, base_small, base_med, base_large, base_xl],
        "base_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
        "act":[gpt, small, med, large, xl],
        "act_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
        "all":[gpt_a, small_a, med_a, large_a, xl_a],
        "all_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
        })
    df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
    df_results.to_csv("results/"+roi+"_results.csv", index=False)
    results[roi] = df_results
    base_plot(df_results, "act", "base", "all", title=roi, ylim1=-0.25, ylim2=0.9)
    
# decompose_sfl
for roi in roin:
    print("\n\n\nProcessing", roi)
    # baseline
    decompose_sfl(sfl_gpt_X, brain_data_avg[roi], "hp_"+roi+"_gpt_base", idx_s = [0,3,6,9], idx_f = [1,4,7,10], idx_l = [2,5,8,11])
    decompose_sfl(sfl_small_X, brain_data_avg[roi], "hp_"+roi+"_small_base", idx_s = [0,3,6,9], idx_f = [1,4,7,10], idx_l = [2,5,8,11])
    decompose_sfl(sfl_med_X, brain_data_avg[roi], "hp_"+roi+"_med_base", idx_s = [0,3,6,9], idx_f = [1,4,7,10], idx_l = [2,5,8,11])
    decompose_sfl(sfl_large_X, brain_data_avg[roi], "hp_"+roi+"_large_base", idx_s = [0,3,6,9], idx_f = [1,4,7,10], idx_l = [2,5,8,11])
    decompose_sfl(sfl_xl_X, brain_data_avg[roi], "hp_"+roi+"_xl_base", idx_s = [0,3,6,9], idx_f = [1,4,7,10], idx_l = [2,5,8,11])

sfl_gpt_X[0]


df_results = pd.read_csv("results/all_results.csv")

thebase = ["hp_all_gpt_base", "hp_all_small_base", "hp_all_med_base", "hp_all_large_base", "hp_all_xl_base"]
theemb  = ["hp_all_gpt", "hp_all_small", "hp_all_med", "hp_all_large", "hp_all_xl"]
theall     = ["hp_all_gpt_all", "hp_all_small_all", "hp_all_med_all", "hp_all_large_all", "hp_all_xl_all"]

p_all = []
for base, emb, all_ in zip(thebase, theemb, theall):
    p_base_emb = significance_testing(emb, base)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_all.append([p_base_emb, p_all_base, p_all_emb])
p_all = pd.DataFrame(p_all, columns=["base_emb", "all_base", "all_emb"])

for df_column in p_all.columns:
    df_results["p_"+df_column] = list(p_all[df_column])
df_results.to_csv("results/all_results.csv")

#######
# SEM #
#######

from utils import plot_sem, get_sem, plot_weights_sem

wordlist = [w for s in sent for w in s]

sem = get_sem(wordlist)
# Missing affective = 81.09
# Missing perceptual = 30.8

sem0 = np.array([list(vec[:14]) for vec in sem])
np.std(sem0, axis=0)
# array([0.52271287, 0.51415222, 0.51965928, 0.55279225, 0.51729822, 0.49560875, 0.53796002, 0.49807445, 0.47240124, 0.51157359, 0.49651252, 0.50787201, 0.55548842, 0.51607958])

sem = embed_words(sem)

# baseline
gpt_base_sem   = np.concatenate((sem, sfl_gpt_X), axis=1)
small_base_sem = np.concatenate((sem, sfl_small_X), axis=1)
med_base_sem   = np.concatenate((sem, sfl_med_X), axis=1)
large_base_sem = np.concatenate((sem, sfl_large_X), axis=1)
xl_base_sem    = np.concatenate((sem, sfl_xl_X), axis=1)

# embedding
gpt_sem   = np.concatenate((sem, gpt_X), axis=1)
small_sem = np.concatenate((sem, small_X), axis=1)
med_sem   = np.concatenate((sem, med_X), axis=1)
large_sem = np.concatenate((sem, large_X), axis=1)
xl_sem    = np.concatenate((sem, xl_X), axis=1)

# all
gpt_all_sem   = np.concatenate((sem, np.concatenate([gpt_X, sfl_gpt_X], axis=1)), axis=1)
small_all_sem = np.concatenate((sem, np.concatenate([small_X, sfl_small_X], axis=1)), axis=1)
med_all_sem   = np.concatenate((sem, np.concatenate([med_X, sfl_med_X], axis=1)), axis=1)
large_all_sem = np.concatenate((sem, np.concatenate([large_X, sfl_large_X], axis=1)), axis=1)
xl_all_sem    = np.concatenate((sem, np.concatenate([xl_X, sfl_xl_X], axis=1)), axis=1)

for roi in roin:
    print("\n\n\nProcessing", roi)
    # baseline
    base_gpt, base_gpt_sd = test_model_Ridge(gpt_base_sem, brain_data_avg[roi], 10, "hp_"+roi+"_gpt_base_sem")
    base_small, base_small_sd = test_model_Ridge(small_base_sem, brain_data_avg[roi], 10, "hp_"+roi+"_small_base_sem")
    base_med, base_med_sd = test_model_Ridge(med_base_sem, brain_data_avg[roi], 10, "hp_"+roi+"_med_base_sem")
    base_large, base_large_sd = test_model_Ridge(large_base_sem, brain_data_avg[roi], 10, "hp_"+roi+"_large_base_sem")
    base_xl, base_xl_sd = test_model_Ridge(xl_base_sem, brain_data_avg[roi], 10, "hp_"+roi+"_xl_base_sem")
    print(base_gpt, base_small, base_med, base_large, base_xl)
    
    # act
    gpt, gpt_sd = test_model_Ridge(gpt_sem, brain_data_avg[roi], 10, "hp_"+roi+"_gpt_sem")
    small, small_sd = test_model_Ridge(small_sem, brain_data_avg[roi], 10, "hp_"+roi+"_small_sem")
    med, med_sd = test_model_Ridge(med_sem, brain_data_avg[roi], 10, "hp_"+roi+"_med_sem")
    large, large_sd = test_model_Ridge(large_sem, brain_data_avg[roi], 10, "hp_"+roi+"_large_sem")
    xl, xl_sd = test_model_Ridge(xl_sem, brain_data_avg[roi], 10, "hp_"+roi+"_xl_sem")
    print(gpt, small, med, large, xl)
    
    # all
    gpt_a, gpt_a_sd = test_model_Ridge(gpt_all_sem, brain_data_avg[roi], 10, "hp_"+roi+"_gpt_all_sem")
    small_a, small_a_sd = test_model_Ridge(small_all_sem, brain_data_avg[roi], 10, "hp_"+roi+"_small_all_sem")
    med_a, med_a_sd = test_model_Ridge(med_all_sem, brain_data_avg[roi], 10, "hp_"+roi+"_med_all_sem")
    large_a, large_a_sd = test_model_Ridge(large_all_sem, brain_data_avg[roi], 10, "hp_"+roi+"_large_all_sem")
    xl_a, xl_a_sd = test_model_Ridge(xl_all_sem, brain_data_avg[roi], 10, "hp_"+roi+"_xl_all_sem")
    print(gpt_a, small_a, med_a, large_a, xl_a)
    
    df_results = pd.DataFrame({
        "base_sem":[base_gpt, base_small, base_med, base_large, base_xl],
        "base_sem_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
        "act_sem":[gpt, small, med, large, xl],
        "act_sem_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
        "all_sem":[gpt_a, small_a, med_a, large_a, xl_a],
        "all_sem_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
        })
    df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
    
    # significance testing
    thebase = ["hp_"+roi+"_gpt_base_sem","hp_"+roi+"_small_base_sem","hp_"+roi+"_med_base_sem","hp_"+roi+"_large_base_sem","hp_"+roi+"_xl_base_sem"]
    theemb  = ["hp_"+roi+"_gpt_sem","hp_"+roi+"_small_sem","hp_"+roi+"_med_sem","hp_"+roi+"_large_sem","hp_"+roi+"_xl_sem"]
    theall  = ["hp_"+roi+"_gpt_all_sem","hp_"+roi+"_small_all_sem","hp_"+roi+"_med_all_sem","hp_"+roi+"_large_all_sem","hp_"+roi+"_xl_all_sem"]

    p = []
    for base, emb, all_ in zip(thebase, theemb, theall):
        p_base_emb = significance_testing(emb, base)
        p_all_base = significance_testing(all_, base)
        p_all_emb  = significance_testing(all_, emb)
        p.append([p_base_emb, p_all_base, p_all_emb])
    p = pd.DataFrame(p, columns=["base_emb_sem", "all_base_sem", "all_emb_sem"])

    for df_column in p.columns:
        df_results["p_"+df_column] = list(p[df_column])
    
    df_old = pd.read_csv("results/"+roi+"_results.csv")
    df_old.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
    merged = pd.concat([df_old, df_results], axis=1)
    merged.to_csv("results/"+roi+"_results_sem.csv", index=False)
    plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", title=roi, subtract=True)


########################
# for last probing exp #
########################

probe(med_X, brain_data_avg["all"], 10, "hp_all_med_probing")
# print(med)

# from joblib import load
# pca = load('/home/dev/Documents/PhD/MIT/project/probing_LMs/pca.pkl')
# med_X_pca   = embed_words(pca.transform(med_filt))

# # first-pass, activation only
# m_med, sd_med = test_model_Ridge(med_X_pca, brain_data_avg["all"], 10, "hp_all_med_probing_pca")


##########################
# end of weight analysis #
##########################
# save med_filt, y
dataname = "hp"
tosave = [med_X, brain_data_avg["all"]]
with open(f"../probing_LMs/XYdata/{dataname}", 'wb') as handle:
    pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)