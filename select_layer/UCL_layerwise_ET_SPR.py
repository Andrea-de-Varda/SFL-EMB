import neurox
import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from glmnet import ElasticNet
import scipy.io
from researchpy import corr_pair
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from scipy.special import softmax
import re
import matplotlib.patches as mpatches
import neurox.data.loader as data_loader
from sklearn.model_selection import KFold

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge, significance_testing, base_plot, plot_sem, get_sem, plot_weights_sem, decompose_sfl

df_all = pd.read_csv("UCL_reading_all.csv")
df_noinit = df_all.dropna(subset=["s_GPT2_small"]) # remove sentence-starting

filtr = []
for index, row in df_noinit.iterrows():
    if "." in row.word:
        filtr.append(False)
    elif "," in row.word:
        filtr.append(False)
    else:
        filtr.append(True)
filtr = np.array(filtr)

fp   = np.array(df_noinit["RTfirstpass"])
ff   = np.array(df_noinit["RTfirstfix"])
tf   = np.array(df_noinit["RTrightbound"])
n400 = np.array(df_noinit["N400"])[filtr]
p600 = np.array(df_noinit["P600"])[filtr]
spr  = np.array(df_noinit["RT"])

# get activations
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium.json', 1024, is_brnn=False)
available = set(df_all.sent_nr-1)

#################################################
# need to account for spillover effects in SPR! #
#################################################

out = []
masks = []
for sent in set(df_noinit.sent_nr):
    temp = df_noinit[df_noinit.sent_nr == sent]
    idxmax = max(temp.word_pos)+1
    for idx in range(4, idxmax):
        twobef = temp[temp.word_pos == idx-2]
        onebef = temp[temp.word_pos == idx-1]
        actual = temp[temp.word_pos == idx]
        a = twobef.values.tolist()[0]+onebef.values.tolist()[0]+actual.values.tolist()[0]
        out.append(a)
        masks.append(True)
cols = [colname + "-2" for colname in df_noinit.columns.tolist()] + [colname + "-1" for colname in df_noinit.columns.tolist()] + df_noinit.columns.tolist()
df_spill = pd.DataFrame(out, columns = cols)

spr_spill = np.array(df_spill["RT"])

# filtering embeddings
count = 0
to_filt = []
for sent in set(df_noinit.sent_nr):
    temp = df_noinit[df_noinit.sent_nr == sent]
    idxmax = max(temp.word_pos)+1
    for idx in range(2, idxmax):
        if idx > 3:
            to_filt.append([count, count-1, count-2])
        count += 1
        
#########################      
# ITERATING OVER LAYERS #
#########################

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act_eye(act_list, layer, dim):
    out = []
    for i in available:
        act_sent = act_list[i][1:]
        out.extend([get_layer(v, layer, dim) for v in act_sent])
    return np.array(out)

def spill_embeddings(emb, filtlist):
    filtered = []
    for idx1, idx2, idx3 in filtlist:
        filtered.append(np.concatenate([emb[idx1], emb[idx2], emb[idx3]]))
    return np.array(filtered)


layerwise_results_et = []; layerwise_results_spr = []
for layer in range(num_layers_gpt2_medium):
    print(f"Processing layer {layer}...")
    med_filt   = get_act_eye(act_gpt2_medium, layer, 1024)
    med_filt_f   = med_filt[filtr]
    med_filt_spill   = spill_embeddings(med_filt, to_filt)

    m_med_fpass, sd_med_fpass     = test_model_Ridge(med_filt, fp, 10, f"UCL_fpass_med_layer{layer}")
    m_med_spr, sd_med_spr     = test_model_Ridge(med_filt_spill, spr_spill, 10, f"UCL_spr_med_layer{layer}")
    print(f"EMB - fp: {m_med_fpass}, spr: {m_med_spr}")
    
    m_med_fpass_b, sd_med_fpass_b     = test_model_Ridge(np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), fp, 10, f"UCL_fpass_med_base_layer{layer}")
    m_med_spr_b, sd_med_spr_b     = test_model_Ridge(np.array(df_spill[["s_GPT2_med", "logfreq", "length",
                                                                  "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                  "s_GPT2_med-2", "logfreq-2", "length-2"]]), spr_spill, 10, f"UCL_spr_med_base_layer{layer}")
    print(f"SFL - fp: {m_med_fpass_b}, spr: {m_med_spr_b}")
    
    gpt2_med_all   = np.concatenate((np.array(df_noinit[["s_GPT2_med", "logfreq", "length"]]), med_filt), axis=1)
    
    
    # first-pass
    m_med_fpass_all, sd_med_fpass_all     = test_model_Ridge(gpt2_med_all, fp, 10, f"UCL_fpass_med_all_layer{layer}")
    
    # SPR, spill
    gpt2_med_all_spill   = np.concatenate((np.array(df_spill[["s_GPT2_med", "logfreq", "length",
                                                                  "s_GPT2_med-1", "logfreq-1", "length-1",
                                                                  "s_GPT2_med-2", "logfreq-2", "length-2"]]), med_filt_spill), axis=1)
    m_med_spr_all, sd_med_spr_all     = test_model_Ridge(gpt2_med_all_spill, spr_spill, 10, f"UCL_spr_med_all_layer{layer}")
    print(f"ALL - fp: {m_med_fpass_all}, spr: {m_med_spr_all}")
    
    p_base_emb_et = significance_testing(f"UCL_fpass_med_base_layer{layer}", f"UCL_fpass_med_layer{layer}")
    p_all_base_et = significance_testing(f"UCL_fpass_med_all_layer{layer}", f"UCL_fpass_med_base_layer{layer}")
    p_all_emb_et  = significance_testing(f"UCL_fpass_med_all_layer{layer}", f"UCL_fpass_med_layer{layer}")
    
    p_base_emb_spr = significance_testing(f"UCL_spr_med_base_layer{layer}", f"UCL_spr_med_layer{layer}")
    p_all_base_spr = significance_testing(f"UCL_spr_med_all_layer{layer}", f"UCL_spr_med_base_layer{layer}")
    p_all_emb_spr  = significance_testing(f"UCL_spr_med_all_layer{layer}", f"UCL_spr_med_layer{layer}")
    
    layerwise_results_et.append([m_med_fpass, sd_med_fpass, m_med_fpass_b, sd_med_fpass_b, m_med_fpass_all, sd_med_fpass_all, p_base_emb_et, p_all_base_et, p_all_emb_et])
    layerwise_results_spr.append([m_med_spr, sd_med_spr, m_med_spr_b, sd_med_spr_b, m_med_spr_all, sd_med_spr_all, p_base_emb_spr, p_all_base_spr, p_all_emb_spr])

layerwise_results_et  = pd.DataFrame(layerwise_results_et, columns = ["EMB", "EMB_sd", "SFL", "SFL_sd", "ALL", "ALL_sd", "P_base_emb", "P_all_base", "P_all_emb"])
layerwise_results_spr = pd.DataFrame(layerwise_results_spr, columns = ["EMB", "EMB_sd", "SFL", "SFL_sd", "ALL", "ALL_sd", "P_base_emb", "P_all_base", "P_all_emb"])

layerwise_results_et.to_csv("select_layer/layerwise_UCL_fpass_med.csv")
layerwise_results_spr.to_csv("select_layer/layerwise_UCL_spr_med.csv")

############
# plotting #
############

layerwise_results_et = pd.read_csv("select_layer/layerwise_UCL_fpass_med.csv")
layerwise_results_spr = pd.read_csv("select_layer/layerwise_UCL_spr_med.csv")

print(f"Best layer ET: {layerwise_results_et['EMB'].idxmax()}")   # 3
print(f"Best layer SPR: {layerwise_results_spr['EMB'].idxmax()}") # 24

def plot_layerwise(df, title = "", plot_sfl = False):
    fig, ax = plt.subplots(figsize=(8*.4, 6*.4), dpi=300)
    x = df.index

    # EMB line and confidence intervals
    ax.plot(x, df['EMB'], label='EMB', color='blue')
    emb_sig = df['P_base_emb'] < 0.05
    ax.fill_between(x, df['EMB'] - df['EMB_sd'] / np.sqrt(10), df['EMB'] + df['EMB_sd'] / np.sqrt(10),
                    color='blue', alpha=0.1)
    for i in range(len(x)):
        if emb_sig[i]:
            ax.plot(x[i], df['EMB'][i], color='blue', alpha=1.0)
        else:
            ax.plot(x[i], df['EMB'][i], color='blue', alpha=0.2)

    ax.plot(x, df['ALL'], label='ALL', color='green')
    all_sig = df['P_all_base'] < 0.05
    ax.fill_between(x, df['ALL'] - df['ALL_sd'] / np.sqrt(10), df['ALL'] + df['ALL_sd'] / np.sqrt(10),
                    color='green', alpha=0.1)
    for i in range(len(x)):
        if all_sig[i]:
            ax.plot(x[i], df['ALL'][i], color='green', alpha=1.0)
        else:
            ax.plot(x[i], df['ALL'][i], color='green', alpha=0.2)

    # horizontal line for SFL
    if plot_sfl:
        ax.axhline(y=df['SFL'].iloc[0], color='red', linestyle='--', label='SFL', alpha=0.5)
        ax.fill_between(x, df['SFL'].iloc[0] - df['SFL_sd'].iloc[0]/np.sqrt(10), 
                        df['SFL'].iloc[0] + df['SFL_sd'].iloc[0]/np.sqrt(10), 
                        color='red', alpha=0.2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('R')
    ax.set_title(title)
    ax.set_xlim(min(x), max(x))
    ax.legend()
    plt.tight_layout()
    plt.show()


plot_layerwise(layerwise_results_et, title = "UCL$_{ET}$")
plot_layerwise(layerwise_results_spr, title = "UCL$_{SPR}$")
