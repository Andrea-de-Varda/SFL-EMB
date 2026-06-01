from os import chdir, path
import sys
import numpy as np
import re
import scipy.io as io
import pandas as pd
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pickle

datadir = "/media/drive2/Datasets/Pereira/data/"
dirName = "/home/dev/Documents/PhD/MIT/project/pereira"
chdir(dirName)

parentdir = path.dirname(dirName)
sys.path.insert(0, parentdir) 

from utils import get_surprisal_tokens, loadmat1, test_model_Ridge, test_model_GAM, base_plot, significance_testing, decompose_sfl

##############################
# EXTRACTING REPRESENTATIONS #
##############################

# https://huggingface.co/transformers/v2.2.0/pretrained_models.html

import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

# 12-block, 768-hidden, 12-heads
if path.isfile(dirName+'/activations_gpt.json') == False:
    transformers_extractor.extract_representations('openai-gpt',
        datadir+"stimuli_384sentences.txt",
        dirName+'/activations_gpt.json',
        aggregation="average"
    )

# 12-block, 768-hidden, 12-heads, 117M parameters.
if path.isfile(dirName+'/activations_gpt2_small.json') == False:
    transformers_extractor.extract_representations('gpt2',
        datadir+"stimuli_384sentences.txt",
        dirName+'/activations_gpt2_small.json',
        aggregation="average"
    )
    
# 24-block, 1024-hidden, 16-heads, 345M parameters.
if path.isfile(dirName+'/activations_gpt2_medium.json') == False:
    transformers_extractor.extract_representations('gpt2-medium',
        datadir+"stimuli_384sentences.txt",
        dirName+'/activations_gpt2_medium.json',
        aggregation="average"
    )

# 36-block, 1280-hidden, 20-heads, 774M parameters.
if path.isfile(dirName+'/activations_gpt2_large.json') == False:
    transformers_extractor.extract_representations('gpt2-large',
        datadir+"stimuli_384sentences.txt",
        dirName+'/activations_gpt2_large.json',
        aggregation="average"
    )

# 48-block, 1600-hidden, 25-heads, 1558M parameters.
if path.isfile(dirName+'/activations_gpt2_xl.json') == False:
    transformers_extractor.extract_representations('gpt2-xl',
        datadir+"stimuli_384sentences.txt",
        dirName+'/activations_gpt2_xl.json',
        aggregation="average"
    )
    

# loading activations
act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl.json', 1600, is_brnn=False)

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act(act_list, layer, dim):
    out = []
    for act_sent in act_list:
        act_sent = act_sent[1:].mean(axis=0) # mean-pooling for sentence-level representations
        out.append(get_layer(act_sent, layer, dim))
    return np.array(out)

gpt_filt   = get_act(act_gpt, 12, 768)
small_filt = get_act(act_gpt2_small, 12, 768)
med_filt   = get_act(act_gpt2_medium, 24, 1024)
large_filt = get_act(act_gpt2_large, 36, 1280)
xl_filt    = get_act(act_gpt2_xl, 48, 1600)

##########################################################

sent = [re.sub("\n", "", s).split() for s in open(datadir+"stimuli_384sentences.txt").readlines()]

# GPT
gpt = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
tok_gpt = AutoTokenizer.from_pretrained("openai-gpt")
s_gpt = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "</w>", toker=tok_gpt, model=gpt)]

# GPT2 SMALL (124M parameters)
small = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
tok_small = AutoTokenizer.from_pretrained("gpt2")
s_small = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "Ġ", toker=tok_small, model=small, cased=True)]

# GPT2 medium
med = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
tok_med = AutoTokenizer.from_pretrained("gpt2-medium")
s_med = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "Ġ", toker=tok_med, model=med, cased=True)]

# GPT2 large
large = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
tok_large = AutoTokenizer.from_pretrained("gpt2-large")
s_large = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "Ġ", toker=tok_large, model=large, cased=True)]

# GPT2 xl
xl = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
tok_xl = AutoTokenizer.from_pretrained("gpt2-xl")
s_xl = [s for sent in tqdm(sent) for s in get_surprisal_tokens(sent, sep = "Ġ", toker = tok_xl, model=xl, cased=True)]

################################

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

freq = [get_f(word) for s in sent for word in s[1:]]
length = [len(word) for s in sent for word in s[1:]]
sentnum = [idx for idx, s in enumerate(sent) for word in s[1:]]

df = pd.DataFrame({"s_GPT":s_gpt,
                   "s_small":s_small,
                   "s_med":s_med,
                   "s_large":s_large,
                   "s_xl":s_xl,
                   "freq":freq,
                   "len":length,
                   "sentnum":sentnum})
# df.to_csv("SFL.csv")
df = pd.read_csv("SFL.csv")

################################
# note: 
subject_data = loadmat1(datadir+"M09/data_384sentences.mat")
subject_data["meta"]["atlases"] # language regions, multiple demands, etc; [1] is language-conservative
# subject_data["meta"]["rois"][1] # rois associated with language 
# subject_data["meta"]["roiMultimask"][2]

def load_brain_data_Pereira(path):
    subject_data = loadmat1(datadir+path+"/examples_384sentences.mat")
    subject_images = subject_data["examples"]
    
    subject_areas = loadmat1(datadir+path+"/data_384sentences.mat")
    lang_rois = subject_areas["meta"]["rois"][1]
    roi_masks = subject_areas["meta"]["roiMultimask"][1]
    
    # filtering per ROI
    dict_roi = {}
    for roi_name, roi_idx in zip(lang_rois, list(range(1, max(roi_masks)+1))):
        subj_roi = subject_images[:,roi_masks == roi_idx]
        dict_roi[roi_name] = subj_roi
    
    dict_roi["all"] = subject_images[:,roi_masks != 0]
    
    return dict_roi

participants = ["M02", "M04", "M07", "M08", "M09", "M14", "M15", "P01"]
brain_data = [load_brain_data_Pereira(p) for p in participants]
rois = ['LMidPostTemp', 'LPostTemp', 'LMidAntTemp', 'LIFG', 'LAntTemp', 'LIFGorb', 'LAngG', 'LMFG', 'all']

brain_data_avg = {}
for roi in rois:
    print(roi, "done")
    tmp = []
    for part in brain_data:
        act = part[roi].mean(axis=1)
        tmp.append(act)
    brain_data_avg[roi] = np.mean(tmp, axis=0)
    
#############################################

def pool_sfl(model, func):
    out = []
    for sentnum in range(384):
        sent_aggregate = list(func(df[df.sentnum == sentnum][[model, "len", "freq"]]))
        out.append(sent_aggregate)
    pooled = np.array(out)
    return pooled

baseline_gpt   = pool_sfl("s_GPT", np.mean)
baseline_small = pool_sfl("s_small", np.mean)
baseline_med   = pool_sfl("s_med", np.mean)
baseline_large = pool_sfl("s_large", np.mean)
baseline_xl    = pool_sfl("s_xl", np.mean)

# all
gpt_all   = np.concatenate((baseline_gpt, gpt_filt), axis=1)
small_all = np.concatenate((baseline_small, small_filt), axis=1)
med_all   = np.concatenate((baseline_med, med_filt), axis=1)
large_all = np.concatenate((baseline_large, large_filt), axis=1)
xl_all    = np.concatenate((baseline_xl, xl_filt), axis=1)

#############################################

results = {}
for roi in rois:
    print("\n\n\nProcessing", roi)
    # baseline
    base_gpt, base_gpt_sd = test_model_Ridge(baseline_gpt, brain_data_avg[roi], 10, "pereira_"+roi+"_gpt_base")
    base_small, base_small_sd = test_model_Ridge(baseline_small, brain_data_avg[roi], 10, "pereira_"+roi+"_small_base")
    base_med, base_med_sd = test_model_Ridge(baseline_med, brain_data_avg[roi], 10, "pereira_"+roi+"_med_base")
    base_large, base_large_sd = test_model_Ridge(baseline_large, brain_data_avg[roi], 10, "pereira_"+roi+"_large_base")
    base_xl, base_xl_sd = test_model_Ridge(baseline_xl, brain_data_avg[roi], 10, "pereira_"+roi+"_xl_base")
    
    # act
    gpt, gpt_sd = test_model_Ridge(gpt_filt, brain_data_avg[roi], 10, "pereira_"+roi+"_gpt")
    small, small_sd = test_model_Ridge(small_filt, brain_data_avg[roi], 10, "pereira_"+roi+"_small")
    med, med_sd = test_model_Ridge(med_filt, brain_data_avg[roi], 10, "pereira_"+roi+"_med")
    large, large_sd = test_model_Ridge(large_filt, brain_data_avg[roi], 10, "pereira_"+roi+"_large")
    xl, xl_sd = test_model_Ridge(xl_filt, brain_data_avg[roi], 10, "pereira_"+roi+"_xl")
    
    # all
    gpt_a, gpt_a_sd = test_model_Ridge(gpt_all, brain_data_avg[roi], 10, "pereira_"+roi+"_gpt_all")
    small_a, small_a_sd = test_model_Ridge(small_all, brain_data_avg[roi], 10, "pereira_"+roi+"_small_all")
    med_a, med_a_sd = test_model_Ridge(med_all, brain_data_avg[roi], 10, "pereira_"+roi+"_med_all")
    large_a, large_a_sd = test_model_Ridge(large_all, brain_data_avg[roi], 10, "pereira_"+roi+"_large_all")
    xl_a, xl_a_sd = test_model_Ridge(xl_all, brain_data_avg[roi], 10, "pereira_"+roi+"_xl_all")
    
    df_results = pd.DataFrame({
        "base":[base_gpt, base_small, base_med, base_large, base_xl],
        "base_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
        "act":[gpt, small, med, large, xl],
        "act_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
        "all":[gpt_a, small_a, med_a, large_a, xl_a],
        "all_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
        })
    df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
    
    # significance testing
    thebase = ["pereira_"+roi+"_gpt_base", "pereira_"+roi+"_small_base", "pereira_"+roi+"_med_base", "pereira_"+roi+"_large_base", "pereira_"+roi+"_xl_base"]
    theemb  = ["pereira_"+roi+"_gpt", "pereira_"+roi+"_small", "pereira_"+roi+"_med", "pereira_"+roi+"_large", "pereira_"+roi+"_xl"]
    theall  = ["pereira_"+roi+"_gpt_all", "pereira_"+roi+"_small_all", "pereira_"+roi+"_med_all", "pereira_"+roi+"_large_all", "pereira_"+roi+"_xl_all"]

    p = []
    for base, emb, all_ in zip(thebase, theemb, theall):
        p_base_emb = significance_testing(emb, base)
        p_all_base = significance_testing(all_, base)
        p_all_emb  = significance_testing(all_, emb)
        p.append([p_base_emb, p_all_base, p_all_emb])
    p = pd.DataFrame(p, columns=["base_emb", "all_base", "all_emb"])

    for df_column in p.columns:
        df_results["p_"+df_column] = list(p[df_column])
    
    
    df_results.to_csv("results/"+roi+"_results.csv", index=False)
    results[roi] = df_results
    if roi == "all":
        base_plot(df_results, "act", "base", "all", title=roi, ylim1=-0.25, ylim2=0.9)
    else:
        base_plot(df_results, "act", "base", "all", title=roi[1:], ylim1=-0.25, ylim2=0.9)

# done later, relative contribution of sfl
for roi in rois:
    print("\n\n\nProcessing", roi)
    # baseline
    decompose_sfl(baseline_gpt, brain_data_avg[roi], "pereira_"+roi+"_gpt_base", idx_s = 0, idx_f = 2, idx_l = 1)
    decompose_sfl(baseline_small, brain_data_avg[roi], "pereira_"+roi+"_small_base", idx_s = 0, idx_f = 2, idx_l = 1)
    decompose_sfl(baseline_med, brain_data_avg[roi], "pereira_"+roi+"_med_base", idx_s = 0, idx_f = 2, idx_l = 1)
    decompose_sfl(baseline_large, brain_data_avg[roi], "pereira_"+roi+"_large_base", idx_s = 0, idx_f = 2, idx_l = 1)
    decompose_sfl(baseline_xl, brain_data_avg[roi], "pereira_"+roi+"_xl_base", idx_s = 0, idx_f = 2, idx_l = 1)
    

# non-linear effects 

# results_gam = {}
# for roi in rois:
#     print("\n\n\nProcessing", roi)
#     # baseline
#     base_gpt, base_gpt_sd = test_model_GAM(baseline_gpt, brain_data_avg[roi], 5, "SFL")
#     base_small, base_small_sd = test_model_GAM(baseline_small, brain_data_avg[roi], 5, "SFL")
#     base_med, base_med_sd = test_model_GAM(baseline_med, brain_data_avg[roi], 5, "SFL")
#     base_large, base_large_sd = test_model_GAM(baseline_large, brain_data_avg[roi], 5, "SFL")
#     base_xl, base_xl_sd = test_model_GAM(baseline_xl, brain_data_avg[roi], 5, "SFL")
    
#     # act
#     gpt, gpt_sd = test_model_Ridge(gpt_filt, brain_data_avg[roi], 5)
#     small, small_sd = test_model_Ridge(small_filt, brain_data_avg[roi], 5)
#     med, med_sd = test_model_Ridge(med_filt, brain_data_avg[roi], 5)
#     large, large_sd = test_model_Ridge(large_filt, brain_data_avg[roi], 5)
#     xl, xl_sd = test_model_Ridge(xl_filt, brain_data_avg[roi], 5)
    
#     # all
#     gpt_a, gpt_a_sd = test_model_Ridge(gpt_all, brain_data_avg[roi], 5)
#     small_a, small_a_sd = test_model_Ridge(small_all, brain_data_avg[roi], 5)
#     med_a, med_a_sd = test_model_Ridge(med_all, brain_data_avg[roi], 5)
#     large_a, large_a_sd = test_model_Ridge(large_all, brain_data_avg[roi], 5)
#     xl_a, xl_a_sd = test_model_Ridge(xl_all, brain_data_avg[roi], 5)
    
#     df_results = pd.DataFrame({
#         "base":[base_gpt, base_small, base_med, base_large, base_xl],
#         "base_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
#         "act":[gpt, small, med, large, xl],
#         "act_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
#         "all":[gpt_a, small_a, med_a, large_a, xl_a],
#         "all_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
#         })
#     df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#     df_results.to_csv("results/"+roi+"_results.csv", index=False)
#     results_gam[roi] = df_results
#     if roi == "all":
#         base_plot(df_results, "act", "base", "all", title=roi, ylim1=-0.25, ylim2=0.9)
#     else:
#         base_plot(df_results, "act", "base", "all", title=roi[1:], ylim1=-0.25, ylim2=0.9)
# def test_model_aggregated(model, roi, pool):
#     agg = []
#     for part_data in brain_data:
#         y = part_data[roi].mean(axis=1)
#         X = pool_sfl(model, pool)
#         r, _ = test_model_Ridge(X, y, 5, model)
#         agg.append(r)
#     return np.mean(agg), np.std(agg)


############################################################################################################
# showing which sentences are predicted to elicit the strongest/weakest responses according to SFL and EMB #
############################################################################################################

from sklearn.linear_model import RidgeCV
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

reg_emb = RidgeCV(alphas=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000), cv = 5)
reg_emb.fit(xl_filt, brain_data_avg["all"])
y_pred_emb = reg_emb.predict(xl_filt)
r, _ = pearsonr(brain_data_avg["all"], y_pred_emb)

reg_sfl = RidgeCV(alphas=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000), cv = 5)
reg_sfl.fit(baseline_xl, brain_data_avg["all"])
y_pred_sfl = reg_sfl.predict(baseline_xl)
r, _ = pearsonr(brain_data_avg["all"], y_pred_sfl)

df_pred = pd.DataFrame(zip([" ".join(s) for s in sent], zscore(y_pred_emb), zscore(y_pred_sfl), zscore(brain_data_avg["all"])), columns = ["sent", "emb", "sfl", "brain"])
df_pred = df_pred.sort_values(by="emb", ascending=False)


# [removed: external semantic-norm files (Warriner affective, Lynott sensorimotor)
#  are not shipped with the repo. The Valence/Arousal/Max-percept.-strength columns
#  they produced are read from the precomputed interpret_brain_results_Pereira.csv below.]
df_pred = pd.read_csv("interpret_brain_results_Pereira.csv")

filt = ~np.isnan(list(df_pred["mean_ar"]))


df_pred["res_emb"] = (df_pred["emb"]-df_pred["brain"])**2
df_pred["res_sfl"] = (df_pred["sfl"]-df_pred["brain"])**2
df_pred["res_diff"] = df_pred["res_emb"] - df_pred["res_sfl"]
df_qualit = df_pred.sort_values(by="res_diff", ascending=True)[["sent", "res_diff", "emb", "sfl", "brain"]]


df_filt = df_pred[filt]

pearsonr(df_filt["Valence"], df_filt["res_diff"])

pearsonr(df_filt["Arousal"], df_filt["emb"])
pearsonr(df_filt["Arousal"], df_filt["brain"])
pearsonr(df_filt["Arousal"], df_filt["sfl"])

pearsonr(df_filt["Valence"], df_filt["emb"])
pearsonr(df_filt["Valence"], df_filt["brain"])
pearsonr(df_filt["Valence"], df_filt["sfl"])

pearsonr(df_filt["Max percept. strength"], df_filt["emb"])
pearsonr(df_filt["Max percept. strength"], df_filt["brain"])
pearsonr(df_filt["Max percept. strength"], df_filt["sfl"])


df_filt["res_emb"]


# figure size in inches
plt.figure(dpi=300)
#sns.set_context("paper", font_scale=1.5)
g = sns.JointGrid(data=df_filt, x="Arousal", y="emb",  height=5)#, xlim=(0, 1), ylim=(0, 7))#, height=5)
g.set_axis_labels("Arousal", "EMB", fontsize=16)
g1 = g.plot_joint(sns.scatterplot, hue=df_filt["brain"], palette=sns.color_palette("icefire", as_cmap=True))#[min([1000, x]) for x in glasgow["obj_count"]]) # icefire_r
g.plot_marginals(sns.histplot, kde=False, bins=50)
g.plot_joint(sns.regplot, scatter=False, color="midnightblue")
#g.ax_joint.text(0.15, 6.9, 'r = 0.43', fontstyle='italic')
# %sns.move_legend(g.ax_joint, "upper left", title='log$_{10}$ N')
plt.tight_layout()


#################################
# FOLLOW-UP: SEMANTIC VARIABLES #
#################################

from utils import plot_sem, get_sem, plot_weights_sem

sem = get_sem(sent)
# Missing affective = 72.23
# Missing perceptual = 29.42

# get_sem([w for s in sent for w in s])


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
small_all_sem = np.concatenate((sem, small_all), axis=1)
med_all_sem   = np.concatenate((sem, med_all), axis=1)
large_all_sem = np.concatenate((sem, large_all), axis=1)
xl_all_sem    = np.concatenate((sem, xl_all), axis=1)

results = {}
for roi in rois:
    print("\n\n\nProcessing", roi)
    # baseline
    base_gpt, base_gpt_sd = test_model_Ridge(gpt_base_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_gpt_base_sem")
    base_small, base_small_sd = test_model_Ridge(small_base_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_small_base_sem")
    base_med, base_med_sd = test_model_Ridge(med_base_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_med_base_sem")
    base_large, base_large_sd = test_model_Ridge(large_base_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_large_base_sem")
    base_xl, base_xl_sd = test_model_Ridge(xl_base_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_xl_base_sem")
    
    # act
    gpt, gpt_sd = test_model_Ridge(gpt_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_gpt_sem")
    small, small_sd = test_model_Ridge(small_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_small_sem")
    med, med_sd = test_model_Ridge(med_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_med_sem")
    large, large_sd = test_model_Ridge(large_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_large_sem")
    xl, xl_sd = test_model_Ridge(xl_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_xl_sem")
    
    # all
    gpt_a, gpt_a_sd = test_model_Ridge(gpt_all_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_gpt_all_sem")
    small_a, small_a_sd = test_model_Ridge(small_all_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_small_all_sem")
    med_a, med_a_sd = test_model_Ridge(med_all_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_med_all_sem")
    large_a, large_a_sd = test_model_Ridge(large_all_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_large_all_sem")
    xl_a, xl_a_sd = test_model_Ridge(xl_all_sem, brain_data_avg[roi], 10, "pereira_"+roi+"_xl_all_sem")
    
    df_results = pd.DataFrame({
        "base":[base_gpt, base_small, base_med, base_large, base_xl],
        "base_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
        "act":[gpt, small, med, large, xl],
        "act_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
        "all":[gpt_a, small_a, med_a, large_a, xl_a],
        "all_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
        })
    df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
    
    # significance testing
    thebase = ["pereira_"+roi+"_gpt_base", "pereira_"+roi+"_small_base", "pereira_"+roi+"_med_base", "pereira_"+roi+"_large_base", "pereira_"+roi+"_xl_base"]
    theemb  = ["pereira_"+roi+"_gpt", "pereira_"+roi+"_small", "pereira_"+roi+"_med", "pereira_"+roi+"_large", "pereira_"+roi+"_xl"]
    theall  = ["pereira_"+roi+"_gpt_all", "pereira_"+roi+"_small_all", "pereira_"+roi+"_med_all", "pereira_"+roi+"_large_all", "pereira_"+roi+"_xl_all"]

    p = []
    for base, emb, all_ in zip(thebase, theemb, theall):
        p_base_emb = significance_testing(emb, base)
        p_all_base = significance_testing(all_, base)
        p_all_emb  = significance_testing(all_, emb)
        p.append([p_base_emb, p_all_base, p_all_emb])
    p = pd.DataFrame(p, columns=["base_emb", "all_base", "all_emb"])

    for df_column in p.columns:
        df_results["p_"+df_column] = list(p[df_column])
    
    results[roi] = df_results
    
    df_old = pd.read_csv("results/"+roi+"_results.csv")
    df_old.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
    
    df_old["base_sem"] = df_results["base"]-df_old["base"]
    df_old["act_sem"] = df_results["act"]-df_old["act"]
    df_old["all_sem"] = df_results["all"]-df_old["all"]
    
    df_old["base_sem_sd"] = df_results["base_sd"]
    df_old["act_sem_sd"] = df_results["act_sd"]
    df_old["all_sem_sd"] = df_results["all_sd"]
    
    df_old.to_csv("results/"+roi+"_results_sem.csv", index=False)
    
    if roi == "all":
        #base_plot(df_results, "act", "base", "all", title=roi, ylim1=-0.25, ylim2=0.9)
        plot_sem(df_old, "act", "base", "all", "act_sem", "base_sem", "all_sem", title=roi)
    else:
        plot_sem(df_old, "act", "base", "all", "act_sem", "base_sem", "all_sem", title=roi[1:])

plot_weights_sem("pereira_all_gpt_base_sem", "all", "weights")




# for weight analysis
from scipy.stats import zscore

med, med_sd = test_model_Ridge(zscore(med_filt), zscore(brain_data_avg["all"]), 10, "pereira_"+roi+"_med_probing")
small, small_sd = test_model_Ridge(zscore(small_filt), zscore(brain_data_avg["all"]), 10, "pereira_"+roi+"_small_probing")
print(small)

from joblib import load
pca = load('/home/dev/Documents/PhD/MIT/project/probing_LMs/pca.pkl')
med_filt_pca = pca.transform(med_filt)

# first-pass, activation only
m_med_fpass, sd_med_fpass = test_model_Ridge(med_filt_pca, brain_data_avg["all"], 10, "pereira_all_med_probing_pca")

##########################
# end of weight analysis #
##########################
# save med_filt, y
dataname = "pereira"
tosave = [med_filt, brain_data_avg["all"]]
with open(f"../probing_LMs/XYdata/{dataname}", 'wb') as handle:
    pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)

