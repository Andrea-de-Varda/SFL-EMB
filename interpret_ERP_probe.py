import neurox
import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from glmnet import ElasticNet
import scipy.io
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.ticker as ticker
from researchpy import corr_pair
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from scipy.special import softmax
import matplotlib.patches as mpatches

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT)
chdir(dirName)

df_all = pd.read_csv("UCL_reading_all.csv")

import neurox.data.loader as data_loader

act_gpt, num_layers_gpt = data_loader.load_activations(dirName+'/activations_gpt.json', 768, is_brnn=False)
act_gpt2_small, num_layers_gpt2_small = data_loader.load_activations(dirName+'/activations_gpt2_small.json', 768, is_brnn=False)
act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(dirName+'/activations_gpt2_medium.json', 1024, is_brnn=False)
act_gpt2_large, num_layers_gpt2_large = data_loader.load_activations(dirName+'/activations_gpt2_large.json', 1280, is_brnn=False)
act_gpt2_xl, num_layers_gpt2_xl = data_loader.load_activations(dirName+'/activations_gpt2_xl.json', 1600, is_brnn=False)

available = set(df_all.sent_nr-1) 

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

gpt_filt   = get_act_eye(act_gpt, 12, 768)
small_filt = get_act_eye(act_gpt2_small, 12, 768)
med_filt   = get_act_eye(act_gpt2_medium, 24, 1024)
large_filt = get_act_eye(act_gpt2_large, 36, 1280)
xl_filt    = get_act_eye(act_gpt2_xl, 48, 1600)
#t_xl_filt  = get_act_eye(act_transf_xl, 18, 1024)

df_noinit = df_all.dropna(subset=["s_GPT2_small"]) # remove sentence-starting

n400 = np.array(df_noinit["N400"])
p600 = np.array(df_noinit["P600"])

##################################
# linear regression, elastic net #
##################################

from utils import test_model_Ridge_nofolds, simple_barplot

# n400, activation only
m_gpt_n400, pred_gpt_n4 = test_model_Ridge_nofolds(gpt_filt, n400, return_predictions = True)
m_small_n400, pred_small_n4 = test_model_Ridge_nofolds(small_filt, n400, return_predictions = True)
m_med_n400, pred_med_n4   = test_model_Ridge_nofolds(med_filt, n400, return_predictions = True)
m_larg_n400, pred_large_n4  = test_model_Ridge_nofolds(large_filt, n400, return_predictions = True)
m_xl_n400, pred_xl_n4    = test_model_Ridge_nofolds(xl_filt, n400, return_predictions = True)

# p600, activations only
m_gpt_p600, pred_gpt_p6     = test_model_Ridge_nofolds(gpt_filt, p600, return_predictions = True)
m_small_p600, pred_small_p6   = test_model_Ridge_nofolds(small_filt, p600, return_predictions = True)
m_med_p600, pred_med_p6     = test_model_Ridge_nofolds(med_filt, p600, return_predictions = True)
m_larg_p600, pred_large_p6    = test_model_Ridge_nofolds(large_filt, p600, return_predictions = True)
m_xl_p600, pred_xl_p6      = test_model_Ridge_nofolds(xl_filt, p600, return_predictions = True)

plt.figure(dpi=300, figsize=(4*0.85, 3*0.85))
plt.plot(list(range(len(m_gpt_n400))), sorted(abs(m_gpt_n400)))
plt.plot(list(range(len(m_small_n400))), sorted(abs(m_small_n400)))
plt.plot(list(range(len(m_med_n400))), sorted(abs(m_med_n400)))
plt.plot(list(range(len(m_larg_n400))), sorted(abs(m_larg_n400)))
plt.plot(list(range(len(m_xl_n400))), sorted(abs(m_xl_n400)))
plt.xlabel("Unit (ranked)")
plt.ylabel("Weight")
plt.ylim(0, .11)
plt.title("UCL, N400")

plt.figure(dpi=300, figsize=(4*0.85, 3*0.85))
plt.plot(list(range(len(m_gpt_p600))), sorted(abs(m_gpt_p600)))
plt.plot(list(range(len(m_small_p600))), sorted(abs(m_small_p600)))
plt.plot(list(range(len(m_med_p600))), sorted(abs(m_med_p600)))
plt.plot(list(range(len(m_larg_p600))), sorted(abs(m_larg_p600)))
plt.plot(list(range(len(m_xl_p600))), sorted(abs(m_xl_p600)))
plt.xlabel("Unit (ranked)")
plt.ylabel("Weight")
plt.ylim(0, .11)
plt.title("UCL, p600")

###############################

out_corr_preds = []
for array1, array2 in zip([pred_gpt_n4, pred_small_n4, pred_med_n4, pred_large_n4, pred_xl_n4], [pred_gpt_p6, pred_gpt_p6, pred_gpt_p6, pred_gpt_p6, pred_gpt_p6]):
    out_corr_preds.append(pearsonr(array1, array2)[0])
    
out_corr_weights = []
for array1, array2 in zip([m_gpt_p600, m_small_p600, m_med_p600, m_larg_p600, m_xl_p600], [m_gpt_n400, m_small_n400, m_med_n400, m_larg_n400, m_xl_n400]):
    out_corr_weights.append(pearsonr(array1, array2)[0])

simple_barplot(out_corr_weights, 0, 0.55, "Correlation Ridge weights (N400, p600)")
simple_barplot(out_corr_preds, 0, 0.55, "Correlation Ridge predictions (N400, p600)")

#############################################
# see if models distinguish between the two #
#############################################

corr_df = []
for array1, array2 in zip([pred_gpt_n4, pred_small_n4, pred_med_n4, pred_large_n4, pred_xl_n4], [pred_gpt_p6, pred_gpt_p6, pred_gpt_p6, pred_gpt_p6, pred_gpt_p6]):
    probe_corr = pearsonr(array1, array2)[0]
    n4_p6 = pearsonr(array1, p600)[0]
    p6_n4 = pearsonr(array2, n400)[0]
    n4_n4 = pearsonr(array1, n400)[0]
    p6_p6 = pearsonr(array2, p600)[0]
    corr_df.append([probe_corr, n4_n4, n4_p6, p6_p6, p6_n4])
corr_df = pd.DataFrame(corr_df, columns = ["pc", "n4", "n4p6", "p6", "p6n4"])

ylabel  = ["GPT","GPT-2$_{124M}$","GPT-2$_{355M}$","GPT-2$_{774M}$","GPT-2$_{1.5B}$"]
plt.rcParams.update({'font.size': 13})
color1 = '#9b90c2'  # Pastel purple
color2 = '#87d3f8'  # Pastel blue
color3 = '#f8b195'  # Pastel peach
color4 = '#FAC8CD' 
color5 = "lavender"
fig1 = plt.figure(dpi = 300, figsize=(8, 6))
ax1 = SubplotHost(fig1, 111)
fig1.add_subplot(ax1)
x = [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23,25,26,27,28,29]
y = list(corr_df.iloc[0, :])+list(corr_df.iloc[1, :])+list(corr_df.iloc[2, :])+list(corr_df.iloc[3, :])+list(corr_df.iloc[4, :])
# First X-axis
ax1.bar(x, y, color = [color1, color2, color3, color4, color5]*5,width=0.9, zorder=4)
ax1.set_xticks(x)
ax1.set_xticklabels([""]*25, fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
ax1.yaxis.set_label_text("r")
ax1.set_xlim(0, 30)
ax1.axhline(y=0, color='black', linestyle='-', zorder=6, linewidth=1)
# Second X-axis
ax2 = ax1.twiny()
offset = 0, -25 # Position of the second axis
new_axisline = ax2.get_grid_helper().new_fixed_axis
ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax2.axis["right"].minor_ticks.set_ticksize(0)
ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator([0.1, 0.3, 0.5, 0.7, 0.9]))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(ylabel))
ax2.axis["bottom"].minor_ticks.set_ticksize(0)
ax1.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=3)
#plt.minorticks_on()
ax1.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=2)
xlabels = ["$r(\widehat{N400}, \widehat{P600})$", "$r(\widehat{N400}, N400)$", "$r(\widehat{N400}, P600)$", "$r(\widehat{P600}, P600)$", "$r(\widehat{P600}, N400)$"]
a  = mpatches.Patch(color=color1, label=xlabels[0])
b  = mpatches.Patch(color=color2, label=xlabels[1])
c = mpatches.Patch(color=color3, label=xlabels[2])
d = mpatches.Patch(color=color4, label=xlabels[3])
e = mpatches.Patch(color=color5, label=xlabels[4])
plt.legend(handles=[a, b, c, d, e], bbox_to_anchor=(1.03, 1.01))
plt.title("Correlation between model predictions and ERP amplitudes")
plt.show()


for pred, name in zip([pred_gpt_n4, pred_small_n4, pred_med_n4, pred_large_n4, pred_xl_n4], ["gpt", "small", "med", "large", "xl"]):
    df_noinit["n4_pred_"+name] = pred

for pred, name in zip([pred_gpt_p6, pred_small_p6, pred_med_p6, pred_large_p6, pred_xl_p6], ["gpt", "small", "med", "large", "xl"]):
    df_noinit["p6_pred_"+name] = pred
    
list(df_noinit[["P600", "p6_pred_small", "word"]].sort_values(by="p6_pred_small")["word"][:10])
list(df_noinit[["P600", "p6_pred_small", "word"]].sort_values(by="p6_pred_small")["word"][-10:])

list(df_noinit[["N400", "n4_pred_small", "word"]].sort_values(by="n4_pred_small")["word"][:10])
list(df_noinit[["N400", "n4_pred_small", "word"]].sort_values(by="n4_pred_small")["word"][-10:])


df_noinit["f*l"] = df_noinit["logfreq"]*df_noinit["length"]
pearsonr(df_noinit["f*l"], df_noinit["n4_pred_large"])

plt.scatter(df_noinit["punct"], df_noinit["n4_pred_small"])



















act_974_sents = [arr[974] for arr in large_filt]
pearsonr(df_noinit["P600"], act_974_sents) # 0.5183848943828538, 2.552325239284303e-119
pearsonr(df_noinit["logfreq"], act_974_sents) # -0.3279875845654431
pearsonr(df_noinit["length"], act_974_sents) # 0.3712862105363316
pearsonr(df_noinit["s_GPT2_large"], act_974_sents) # 0.5304140701597945
pearsonr(df_noinit["s_GPT2_large"], df_noinit["P600"]) # 0.30586639799386084

# it is a psychometric neuron, partially correlated with surprisal

df_noinit["top_p600"] = act_974_sents
print(df_noinit.sort_values(by="top_p600", ascending=True)["word"][:30])
print(df_noinit.sort_values(by="top_p600", ascending=True)["word"][-30:])

import statsmodels.formula.api as smf
mod = smf.ols(formula='P600 ~ s_GPT2_large + top_p600', data=df_noinit)
res = mod.fit()
print(res.summary())




