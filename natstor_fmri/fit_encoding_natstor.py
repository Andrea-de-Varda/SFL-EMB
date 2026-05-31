import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr
from math import sqrt, ceil
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import warnings
import pickle
import sys

path = "/home/dev/Documents/PhD/MIT/project/natstor_fmri"
chdir(path)

parent_dir = os.path.abspath(os.path.join(path, '..'))
sys.path.append(parent_dir)

from utils import get_surprisal_tokens, base_plot, significance_testing, decompose_sfl, probe

warnings.filterwarnings("ignore", message="Mean of empty slice")

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

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

def embed_words(embeddings, words_id, total_duration):
    ids = words_id.astype(int)
    time = np.arange(0, total_duration, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = delay_mat(np.array(emb_words), np.arange(1,5))
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, embeddings):
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df.drop(df.index[0]) # removing first row cause no surprisal for the first word
    df = df.reset_index()
    total_duration =  len(d[story_n_dict[lang]]) * 2 # ceil(df["end"].max())
    time = np.arange(0, total_duration, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id, total_duration)
    return embedded_words

def cross_story_encoding(langs, model_prefix, d, prefix = "", overwrite = True, condition = "emb", normalize = False, sem = False):
    if os.path.isfile(f"results/cross_story_{prefix}{model_prefix}") and overwrite == False:
        print(f"Encoding for {model_prefix} already done")
    else:
        modelname = re.sub("gpt2_", "", model_prefix) # for consistent saving
        if condition == "emb":
            embs_ = load(f"{model_prefix}_{langs[0]}")
            n = max(embs_.keys()) # get last layer
            fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n][1:]) for lang in langs] # skip first since no surprisal for first word
            saveto = f"natstorfmri_{modelname}"
            if normalize:
                fmri_data = [zscore(embeddings) for embeddings in fmri_data]
                saveto = f"natstorfmri_{modelname}_probing"
        elif condition == "sfl":
            fmri_data = [preproc_align(lang, load(f"sfl_{model_prefix}_{lang}")) for lang in langs]
            saveto = f"natstorfmri_{modelname}_base"
        elif condition == "all":
            embs_ = load(f"{model_prefix}_{langs[0]}")
            n = max(embs_.keys()) # get last layer
            embs = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n][1:]) for lang in langs]
            sfl = [preproc_align(lang, load(f"sfl_{model_prefix}_{lang}")) for lang in langs]
            fmri_data = []
            for emb_, sfl_ in zip(embs, sfl):
                all_ = np.concatenate([emb_, sfl_], axis=1)
                fmri_data.append(all_)
            saveto = f"natstorfmri_{modelname}_all"
        if sem:
            sem_data = [preproc_align(lang, load(f"sem_{lang}")) for lang in langs]
            fmri_data = [np.concatenate([sem_, data_], axis=1) for sem_, data_ in zip(sem_data, fmri_data)]
            saveto += "_sem"
        ############################
        out_reg = []
        out_coefs = []
        out_predictions = []
        for i in tqdm(range(len(langs))):
            X_data = fmri_data[:i] + fmri_data[i+1:] # exclude lang_i
            X_train = np.concatenate(X_data)
            y_names = langs[:i] + langs[i+1:]
            y_train = np.concatenate([d[story_n_dict[name]] for name in y_names])
            X_test = fmri_data[i]#.reshape(1, -1)
            y_test = d[story_n_dict[langs[i]]]
            if normalize:
                y_test = zscore(y_test)
                y_train = zscore(y_train)
            # fitting
            reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            r, p = pearsonr(y_test, y_pred)
            out_predictions.append([y_test, y_pred])
            coefs = reg.coef_#; print(coefs)
            out_coefs.append(coefs)
            out_reg.append(r)
            print(r, p)
        ############################
        coefs_avg = np.mean(out_coefs, axis=0)
        coefs_std = np.std(out_coefs, axis=0)
        if save:
            with open("results/"+saveto+"_coefs_avg", 'wb') as handle:
                pickle.dump(coefs_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("results/"+saveto+"_coefs", 'wb') as handle:
                pickle.dump(out_coefs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("results/ì"+saveto+"_coefs_std", 'wb') as handle:
                pickle.dump(coefs_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("results/"+saveto+"_predictions", "wb") as handle:
                pickle.dump(out_reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return np.mean(out_reg), np.std(out_reg)


###############################################################################

# load fMRI data
with open("response/d_natstor", 'rb') as handle:
    d = pickle.load(handle)
    
stories = ["1", "2", "3", "4", "5", "6", "7", "9", "10"]

story_n_dict = {"1" : "boar",
                "2" : "aqua",
                "3" : "matchstickseller",
                "4" : "kingofbirds", 
                "5" : "elvis", 
                "6" : "mrsticky",
                "7" : "highschool",
                "10" : "tree",
                "9" : "tulips"}

##################
# fitting models #
##################

# EMB
gpt, gpt_sd = cross_story_encoding(stories, "gpt", d)
small, small_sd = cross_story_encoding(stories, "gpt2_small", d)
med, med_sd = cross_story_encoding(stories, "gpt2_med", d)
large, large_sd = cross_story_encoding(stories, "gpt2_large", d)
xl, xl_sd = cross_story_encoding(stories, "gpt2_xl", d)

# for last probing exp
med_, med_sd_ = cross_story_encoding(stories, "gpt2_med", d, normalize = True)


# SFL
base_gpt, base_gpt_sd = cross_story_encoding(stories, "gpt", d, condition="sfl")
base_small, base_small_sd = cross_story_encoding(stories, "gpt2_small", d, condition="sfl")
base_med, base_med_sd = cross_story_encoding(stories, "gpt2_med", d, condition="sfl")
base_large, base_large_sd = cross_story_encoding(stories, "gpt2_large", d, condition="sfl")
base_xl, base_xl_sd = cross_story_encoding(stories, "gpt2_xl", d, condition="sfl")

# ALL
gpt_a, gpt_a_sd = cross_story_encoding(stories, "gpt", d, condition="all")
small_a, small_a_sd = cross_story_encoding(stories, "gpt2_small", d, condition="all")
med_a, med_a_sd = cross_story_encoding(stories, "gpt2_med", d, condition="all")
large_a, large_a_sd = cross_story_encoding(stories, "gpt2_large", d, condition="all")
xl_a, xl_a_sd = cross_story_encoding(stories, "gpt2_xl", d, condition="all")

df_results = pd.DataFrame({
    "base":[base_gpt, base_small, base_med, base_large, base_xl],
    "base_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
    "act":[gpt, small, med, large, xl],
    "act_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
    "all":[gpt_a, small_a, med_a, large_a, xl_a],
    "all_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
    })
df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

# singificance testing 
thebase = ["natstorfmri_gpt_base", "natstorfmri_small_base", "natstorfmri_med_base", "natstorfmri_large_base", "natstorfmri_xl_base"]
theemb  = ["natstorfmri_gpt", "natstorfmri_small", "natstorfmri_med", "natstorfmri_large", "natstorfmri_xl"]
theall     = ["natstorfmri_gpt_all", "natstorfmri_small_all", "natstorfmri_med_all", "natstorfmri_large_all", "natstorfmri_xl_all"]

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

base_plot(df_results, "act", "base", "all", title="NatStor$_{fMRI}$", ylim1=0, ylim2=1, cv_size = 9)

#######
# SEM #
#######

# EMB
gpt, gpt_sd = cross_story_encoding(stories, "gpt", d, sem = True)
small, small_sd = cross_story_encoding(stories, "gpt2_small", d, sem = True)
med, med_sd = cross_story_encoding(stories, "gpt2_med", d, sem = True)
large, large_sd = cross_story_encoding(stories, "gpt2_large", d, sem = True)
xl, xl_sd = cross_story_encoding(stories, "gpt2_xl", d, sem = True)

# SFL
base_gpt, base_gpt_sd = cross_story_encoding(stories, "gpt", d, condition="sfl", sem = True)
base_small, base_small_sd = cross_story_encoding(stories, "gpt2_small", d, condition="sfl", sem = True)
base_med, base_med_sd = cross_story_encoding(stories, "gpt2_med", d, condition="sfl", sem = True)
base_large, base_large_sd = cross_story_encoding(stories, "gpt2_large", d, condition="sfl", sem = True)
base_xl, base_xl_sd = cross_story_encoding(stories, "gpt2_xl", d, condition="sfl", sem = True)

# ALL
gpt_a, gpt_a_sd = cross_story_encoding(stories, "gpt", d, condition="all", sem = True)
small_a, small_a_sd = cross_story_encoding(stories, "gpt2_small", d, condition="all", sem = True)
med_a, med_a_sd = cross_story_encoding(stories, "gpt2_med", d, condition="all", sem = True)
large_a, large_a_sd = cross_story_encoding(stories, "gpt2_large", d, condition="all", sem = True)
xl_a, xl_a_sd = cross_story_encoding(stories, "gpt2_xl", d, condition="all", sem = True)

df_results_sem = pd.DataFrame({
    "base_sem":[base_gpt, base_small, base_med, base_large, base_xl],
    "base_sem_sd":[base_gpt_sd, base_small_sd, base_med_sd, base_large_sd, base_xl_sd],
    "act_sem":[gpt, small, med, large, xl],
    "act_sem_sd":[gpt_sd, small_sd, med_sd, large_sd, xl_sd],
    "all_sem":[gpt_a, small_a, med_a, large_a, xl_a],
    "all_sem_sd":[gpt_a_sd, small_a_sd, med_a_sd, large_a_sd, xl_a_sd]
    })
df_results_sem.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]

# singificance testing 
thebase = ["natstorfmri_gpt_base_sem", "natstorfmri_small_base_sem", "natstorfmri_med_base_sem", "natstorfmri_large_base_sem", "natstorfmri_xl_base_sem"]
theemb  = ["natstorfmri_gpt_sem", "natstorfmri_small_sem", "natstorfmri_med_sem", "natstorfmri_large_sem", "natstorfmri_xl_sem"]
theall     = ["natstorfmri_gpt_all_sem", "natstorfmri_small_all_sem", "natstorfmri_med_all_sem", "natstorfmri_large_all_sem", "natstorfmri_xl_all_sem"]

p_all_sem = []
for base, emb, all_ in zip(thebase, theemb, theall):
    p_base_emb = significance_testing(emb, base)
    p_all_base = significance_testing(all_, base)
    p_all_emb  = significance_testing(all_, emb)
    p_all_sem.append([p_base_emb, p_all_base, p_all_emb])
p_all_sem = pd.DataFrame(p_all_sem, columns=["base_emb_sem", "all_base_sem", "all_emb_sem"])

for df_column in p_all_sem.columns:
    df_results_sem["p_"+df_column] = list(p_all_sem[df_column])

merged = pd.concat([df_results, df_results_sem], axis=1)
merged.to_csv("results/all_results_sem.csv", index=False)
plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", title="NatStor$_{fMRI}$", subtract=True, cv_size = 9)

#################
# decompose SFL #
#################

def decompose_sfl_natstor(langs, model_prefix, d, idx_s = [0,3,6,9], idx_f = [1,4,7,10], idx_l = [2,5,8,11]):
    modelname = re.sub("gpt2_", "", model_prefix) # for consistent saving
    fmri_data = [preproc_align(lang, load(f"sfl_{model_prefix}_{lang}")) for lang in langs]
    saveto = f"natstorfmri_{modelname}"

    ############################
    names = ["S", "F", "L", "SF",   "FL",   "SL",   "SFL"]
    out = {name : [] for name in names}
    # desired = {"S" : [m, sd], ...}
    for i in tqdm(range(len(langs))):
        X_data = fmri_data[:i] + fmri_data[i+1:] # exclude lang_i
        X_train = np.concatenate(X_data)
        y_names = langs[:i] + langs[i+1:]
        y_train = np.concatenate([d[story_n_dict[name]] for name in y_names])
        X_test = fmri_data[i]#.reshape(1, -1)
        y_test = d[story_n_dict[langs[i]]]
        
        s_train = X_train[:, idx_s]
        f_train = X_train[:, idx_f]
        l_train = X_train[:, idx_l]
        
        s_test = X_test[:, idx_s]
        f_test = X_test[:, idx_f]
        l_test = X_test[:, idx_l]
        
        combs_train = [(s_train), (f_train), (l_train), (s_train, f_train), (f_train, l_train), (s_train, l_train), (s_train, f_train, l_train)]
        combs_test = [(s_test), (f_test), (l_test), (s_test, f_test), (f_test, l_test), (s_test, l_test), (s_test, f_test, l_test)]
        
        for c_train, c_test, n in zip(combs_train, combs_test, names):
            if n in ["S", "F", "L"]:
                the_x = c_train
                the_x_test = c_test
                reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
                reg.fit(the_x, y_train)
                y_pred = reg.predict(the_x_test)
                r, p = pearsonr(y_test, y_pred)
                out[n].append(r)
            else:
                the_x = np.hstack(c_train)
                the_x_test = np.hstack(c_test)
                reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
                reg.fit(the_x, y_train)
                y_pred = reg.predict(the_x_test)
                r, p = pearsonr(y_test, y_pred)
                out[n].append(r)
    for k, v in out.items():
        out[k] = [np.mean(out[k]), np.std(out[k])]
        
    df = pd.DataFrame(out)
    df.index = ["m", "sd"]
    print("\n", df)
    df.to_csv("results/"+saveto+"_sfl_decomposed.csv")

decompose_sfl_natstor(stories, "gpt", d)
decompose_sfl_natstor(stories, "gpt2_small", d)
decompose_sfl_natstor(stories, "gpt2_med", d)
decompose_sfl_natstor(stories, "gpt2_large", d)
decompose_sfl_natstor(stories, "gpt2_xl", d)

##########################
# end of weight analysis #
##########################
# save med_filt, y

all_data = np.concatenate([preproc_align(lang, load(f"gpt2_med_{lang}")[24][1:]) for lang in stories])
all_y = np.concatenate([d[story_n_dict[name]] for name in stories])

probe(all_data, all_y, 10, "natstorfmri_med_probing")

dataname = "natstor_fmri"
tosave = [all_data, all_y]
with open(f"../probing_LMs/XYdata/{dataname}", 'wb') as handle:
    pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)