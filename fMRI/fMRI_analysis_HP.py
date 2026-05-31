import numpy as np
import pandas as pd
from os import chdir, path
from scipy.stats import zscore
from tqdm import tqdm
import pickle
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from scipy.special import softmax
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost

chdir("/home/dev/Documents/PhD/MIT/project/fMRI")

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

from sklearn.linear_model import LinearRegression, ElasticNet, MultiTaskElasticNetCV, Ridge, MultiTaskLassoCV, RidgeCV
#from glmnet import ElasticNet
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

roi = np.load('fMRI/HP_subj_roi_inds.npy', allow_pickle=True)
roi = roi.tolist()

parts = [part_f, part_h, part_i, part_j, part_k, part_l, part_m, part_n]
partn = ["F", "H", "I", "J", "K", "L", "M", "N"]

def test_model(X, y, n, partName, the_roi, average=False, give_coefs=False):
    if average:
        y = y[:,roi[partName][the_roi]].mean(axis=1)
    else:
        y = y[:,roi[partName][the_roi]]
    kf = KFold(n_splits=n)
    out_reg = []
    out_coefs = []
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        reg = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000), cv = 5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        coefs = reg.coef_
        out_reg.append([y_test, y_pred])
        out_coefs.append(coefs)
    if give_coefs:
        return out_reg, np.mean(out_coefs, axis=0) # average across folds
    else:
        return out_reg

def corr_res(res, average=False):
    out_outer = []
    for fold in res:
        fold_test, fold_pred = fold
        if average:
            out_outer.append(pearsonr(fold_test, fold_pred)[0])
        else:
            out_inner = []
            for target, pred in zip(fold_test, fold_pred):
                out_inner.append(pearsonr(target, pred)[0])
            thecorr = np.mean(out_inner)
            out_outer.append(thecorr)
    return np.mean(out_outer)# , np.std(out) # -- sd will be calculated across participants

# gpt_res = test_model(gpt_X, part_i, 5, "I", "IFG", average=True)
# gpt_res_s = test_model(sfl_gpt_X, part_i, 5, "I", "IFG")
# gpt_res_all = test_model(np.concatenate([sfl_gpt_X, gpt_X], axis=1), part_i, 5, "I", "IFG")
# corr_res(gpt_res_s)
# corr_res(gpt_res)
# corr_res(gpt_res_all)

def test_model_avg(X, the_roi, average=False, give_coefs=False):
    to_out = []
    out_coefs = []
    for part, part_name in zip(parts, partn):
        if give_coefs:
            part_res, coefs = test_model(X, part, 5, part_name, the_roi, average=average, give_coefs=True)
            score = corr_res(part_res, average=average)
            to_out.append(score)
            out_coefs.append(coefs)
        else:
            part_res = test_model(X, part, 5, part_name, the_roi, average=average)
            score = corr_res(part_res, average=average)
            to_out.append(score)
    if give_coefs:
        return np.mean(to_out), np.std(to_out), np.mean(out_coefs, axis=0), np.std(out_coefs, axis=0) # note that coefs had been averaged over folds before; now averaged over participants; dispersion (SD) calculated across participants
    else:
        return np.mean(to_out), np.std(to_out)

gpt_m, gpt_sd         = test_model_avg(gpt_X, "all")
base_gpt, base_gpt_sd = test_model_avg(sfl_gpt_X, "all")
all_gpt, all_gpt_sd   = test_model_avg(np.concatenate([gpt_X, sfl_gpt_X], axis=1), "all")
print(gpt_m, base_gpt, all_gpt)

small_m, small_sd         = test_model_avg(small_X, "all")
base_small, base_small_sd = test_model_avg(sfl_small_X, "all")
all_small, all_small_sd   = test_model_avg(np.concatenate([small_X, sfl_small_X], axis=1), "all")
print(small_m, base_small, all_small)

med_m, med_sd         = test_model_avg(med_X, "all")
base_med, base_med_sd = test_model_avg(sfl_med_X, "all")
all_med, all_med_sd   = test_model_avg(np.concatenate([med_X, sfl_med_X], axis=1), "all")
print(med_m, base_med, all_med)

large_m, large_sd         = test_model_avg(large_X, "all")
base_large, base_large_sd = test_model_avg(sfl_large_X, "all")
all_large, all_large_sd   = test_model_avg(np.concatenate([large_X, sfl_large_X], axis=1), "all")
print(large_m, base_large, all_large)

xl_m, xl_sd         = test_model_avg(xl_X, "all")
base_xl, base_xl_sd = test_model_avg(sfl_xl_X, "all")
all_xl, all_xl_sd   = test_model_avg(np.concatenate([xl_X, sfl_xl_X], axis=1), "all")
print(xl_m, base_xl, all_xl)

df_results_allroi = pd.DataFrame({
    "base":  [base_gpt, base_small, base_med, base_large, base_xl],
    "base_sd":  [base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
    "act":  [gpt_m, small_m, med_m, large_m, xl_m],
    "act_sd":  [gpt_sd, small_sd, med_sd, large_sd, xl_sd],
    "all":[all_gpt, all_small, all_med, all_large, all_xl],
    "all_sd":[all_gpt_sd, all_small_sd, all_med_sd, all_large_sd, all_xl_sd]
    })
df_results_allroi.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#df_results_allroi.to_csv("results/allroi.csv")


# all language network

plt.rcParams.update({
    "text.usetex": False})
fig = plt.figure(figsize = (5/1.2, 3.5/1.2), dpi=300)
plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
plt.minorticks_on()
plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
plt.bar([1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], 
        [df_results_allroi.loc["GPT", "act"], df_results_allroi.loc["GPT", "base"], df_results_allroi.loc["GPT", "all"],
         df_results_allroi.loc["GPT-2$_{124M}$", "act"], df_results_allroi.loc["GPT-2$_{124M}$", "base"], df_results_allroi.loc["GPT-2$_{124M}$", "all"],
         df_results_allroi.loc["GPT-2$_{355M}$", "act"], df_results_allroi.loc["GPT-2$_{355M}$", "base"], df_results_allroi.loc["GPT-2$_{355M}$", "all"],
         df_results_allroi.loc["GPT-2$_{774M}$", "act"], df_results_allroi.loc["GPT-2$_{774M}$", "base"], df_results_allroi.loc["GPT-2$_{774M}$", "all"],
         df_results_allroi.loc["GPT-2$_{1.5B}$", "act"], df_results_allroi.loc["GPT-2$_{1.5B}$", "base"], df_results_allroi.loc["GPT-2$_{1.5B}$", "all"]],
        color = ['lightblue', 'lightgreen', "indianred"]*5,width=0.9,
        yerr=[df_results_allroi.loc["GPT", "act_sd"], df_results_allroi.loc["GPT", "base_sd"], df_results_allroi.loc["GPT", "all_sd"],
         df_results_allroi.loc["GPT-2$_{124M}$", "act_sd"], df_results_allroi.loc["GPT-2$_{124M}$", "base_sd"], df_results_allroi.loc["GPT-2$_{124M}$", "all_sd"],
         df_results_allroi.loc["GPT-2$_{355M}$", "act_sd"], df_results_allroi.loc["GPT-2$_{355M}$", "base_sd"], df_results_allroi.loc["GPT-2$_{355M}$", "all_sd"],
         df_results_allroi.loc["GPT-2$_{774M}$", "act_sd"], df_results_allroi.loc["GPT-2$_{774M}$", "base_sd"], df_results_allroi.loc["GPT-2$_{774M}$", "all_sd"],
         df_results_allroi.loc["GPT-2$_{1.5B}$", "act_sd"], df_results_allroi.loc["GPT-2$_{1.5B}$", "base_sd"], df_results_allroi.loc["GPT-2$_{1.5B}$", "all_sd"]],capsize=2,
        tick_label=["", "GPT", "", "", "GPT-2$_{124M}$", "", "", "GPT-2$_{355M}$","", "",  "GPT-2$_{774M}$", "", "", "GPT-2$_{1.5B}$", ""]
        )
plt.ylabel("r")
plt.xticks(fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
plt.ylim(0, 0.2)
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
plt.show()

###############################################
# average activation in language network ROIs #
###############################################

out_univ = []
out_coefs = {}
for the_roi in ["all", "PostTemp", "AntTemp", "AngularG", "IFG", "MFG", "IFGorb", "pCingulate"]:
    gpt_m, gpt_sd         = test_model_avg(gpt_X, the_roi, average=True)
    base_gpt, base_gpt_sd, coefs_gpt, coefs_sd_gpt = test_model_avg(sfl_gpt_X, the_roi, average=True, give_coefs = True)
    all_gpt, all_gpt_sd   = test_model_avg(np.concatenate([gpt_X, sfl_gpt_X], axis=1), the_roi, average=True)
    print(the_roi, "GPT", gpt_m, base_gpt, all_gpt)
    
    small_m, small_sd         = test_model_avg(small_X, the_roi, average=True)
    base_small, base_small_sd, coefs_small, coefs_sd_small = test_model_avg(sfl_small_X, the_roi, average=True, give_coefs = True)
    all_small, all_small_sd   = test_model_avg(np.concatenate([small_X, sfl_small_X], axis=1), the_roi, average=True)
    print(the_roi, "small", small_m, base_small, all_small)
    
    med_m, med_sd         = test_model_avg(med_X, the_roi, average=True)
    base_med, base_med_sd, coefs_med, coefs_sd_med = test_model_avg(sfl_med_X, the_roi, average=True, give_coefs = True)
    all_med, all_med_sd   = test_model_avg(np.concatenate([med_X, sfl_med_X], axis=1), the_roi, average=True)
    print(the_roi, "med", med_m, base_med, all_med)
    
    large_m, large_sd         = test_model_avg(large_X, the_roi, average=True)
    base_large, base_large_sd, coefs_large, coefs_sd_large = test_model_avg(sfl_large_X, the_roi, average=True, give_coefs = True)
    all_large, all_large_sd   = test_model_avg(np.concatenate([large_X, sfl_large_X], axis=1), the_roi, average=True)
    print(the_roi, "large", large_m, base_large, all_large)
    
    xl_m, xl_sd         = test_model_avg(xl_X, the_roi, average=True)
    base_xl, base_xl_sd, coefs_xl, coefs_sd_xl = test_model_avg(sfl_xl_X, the_roi, average=True, give_coefs = True)
    all_xl, all_xl_sd   = test_model_avg(np.concatenate([xl_X, sfl_xl_X], axis=1), the_roi, average=True)
    print(the_roi, "XL", xl_m, base_xl, all_xl)
    
    out_univ.append([the_roi, gpt_m, base_gpt, all_gpt, small_m, base_small, all_small, med_m, base_med, all_med, large_m, base_large, all_large, xl_m, base_xl, all_xl, gpt_sd, base_gpt_sd, all_gpt_sd, small_sd, base_small_sd, all_small_sd, med_sd, base_med_sd, all_med_sd, large_sd, base_large_sd, all_large_sd, xl_sd, base_xl_sd, all_xl_sd])
    
    out_coefs[the_roi] = {"gpt": [coefs_gpt, coefs_sd_gpt], "small":[coefs_small, coefs_sd_small], "med": [coefs_med, coefs_sd_med], "large":[coefs_large, coefs_sd_large], "xl":[coefs_xl, coefs_sd_xl]}
    
results_univ = pd.DataFrame(out_univ, columns = ["the_roi", "gpt_m", "base_gpt", "all_gpt", "small_m", "base_small", "all_small", "med_m", "base_med", "all_med", "large_m", "base_large", "all_large", "xl_m", "base_xl", "all_xl", "gpt_sd", "base_gpt_sd", "all_gpt_sd", "small_sd", "base_small_sd", "all_small_sd", "med_sd", "base_med_sd", "all_med_sd", "large_sd", "base_large_sd", "all_large_sd", "xl_sd", "base_xl_sd", "all_xl_sd"])
results_univ.index = results_univ["the_roi"]

# results_univ.to_csv("results/fMRI_univariate_linear.csv")

with open('results/coefs_linear_fMRI', 'wb') as handle:
    pickle.dump(out_coefs, handle, protocol=pickle.HIGHEST_PROTOCOL)
