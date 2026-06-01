from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier, Ridge, RidgeCV
import warnings
import pickle
import pandas as pd
from os import chdir
import numpy as np
from scipy.special import softmax
import torch.nn.functional as F
import torch
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from scipy.stats import zscore
import random
import re
import time
from itertools import combinations
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.patches as mpatches
from math import sqrt

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT)
chdir(dirName)

from utils import test_model_Ridge

def load_weights_corr(name, thedir="", split =False, n = 4):
    with open(thedir+'results/'+name+'_coefs', 'rb') as handle:
        weights = pickle.load(handle)
    if split:
        weights = [np.mean(np.array_split(w, 4), axis=0) for w in weights]
    out_corr = []
    for c1, c2 in combinations(weights, 2):
        r, _ = pearsonr(c1, c2)
        out_corr.append(r)
    print(np.mean(out_corr))
    return np.mean(out_corr), np.std(out_corr)/sqrt(len(out_corr))

hp = load_weights_corr("hp_all_med", "fMRI/", split=True, n=4)
pereira = load_weights_corr("pereira_all_med", "pereira/")
michaelov = load_weights_corr("michaelov_med")
ucl_fpass = load_weights_corr("UCL_fpass_med")
ucl_n400 = load_weights_corr("UCL_n400_med")
ucl_p600 = load_weights_corr("UCL_p600_med")
ucl_spr = load_weights_corr("UCL_spr_med", split = True, n=3)
meco = load_weights_corr("meco_med")
provo = load_weights_corr("provo_med")
zuco = load_weights_corr("zuco_med", "ZuCo/")
maze = load_weights_corr("maze_med")
natstor = load_weights_corr("natstor_med", split = True, n=3)
brown = load_weights_corr("brown_med", split = True, n=3)

heights = [provo[0], meco[0], zuco[0], ucl_fpass[0], ucl_spr[0], brown[0], natstor[0], maze[0], ucl_n400[0], ucl_p600[0], michaelov[0], hp[0], pereira[0]]
yerr = [provo[1], meco[1], zuco[1], ucl_fpass[1], ucl_spr[1], brown[1], natstor[1], maze[1], ucl_n400[1], ucl_p600[1], michaelov[1], hp[1], pereira[1]]

c1 = '#9b90c2'  # Pastel purple
c2 = '#87d3f8'  # Pastel blue
c3 = '#f8b195'  # Pastel peach

plt.figure(figsize = (6, 3), dpi = 300)
x = [0,1,2,3,5,6,7,9,11,12,13,15,16]
y = heights
# First X-axis
plt.bar(x, y, color = [c2, c2,c2,c2,c2,c2,c2,c2,c3,c3,c3,c3,c3],yerr=yerr,capsize=2,width=0.9, zorder=4, tick_label = ["Provo", "Meco", "ZuCo", "UCL$_{ET}$", "UCL$_{SPR}$", "Brown", "NatStor$_{SPR}$", "NatStor$_{Maze}$", "UCL$_{N400}$", "UCL$_{P600}$", "Michaelov", "Harry Potter", "Pereira"])
#ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
plt.xticks(fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
plt.xlim(-1, 17)
plt.ylim(.5, .99)
plt.ylabel("r")
plt.title("Correlation of the EMB-based regression weights across the 10 folds")
plt.show()



def load_weights(name, thedir=""):
    with open(thedir+'results/'+name+'_coefs', 'rb') as handle:
        weights = pickle.load(handle)
        weights = np.mean(weights, axis=0)
    return weights

hp = load_weights("hp_all_med", "fMRI/")
hp_split = np.mean(np.array_split(hp, 4), axis=0) # averaging for spillover
#hp_split = np.array_split(hp, 4)[-1]
pereira = load_weights("pereira_all_med", "pereira/")
michaelov = load_weights("michaelov_med")
ucl_fpass = load_weights("UCL_fpass_med")
ucl_n400 = load_weights("UCL_n400_med")
ucl_p600 = load_weights("UCL_p600_med")
ucl_spr = load_weights("UCL_spr_med")
ucl_spr = np.mean(np.array_split(ucl_spr, 3), axis=0) # averaging for spillover
meco = load_weights("meco_med")
provo = load_weights("provo_med")
zuco = load_weights("zuco_med", "ZuCo/")
maze = load_weights("maze_med")
natstor = load_weights("natstor_med")
natstor = np.mean(np.array_split(natstor, 3), axis=0) # averaging for spillover
brown = load_weights("brown_med")
brown = np.mean(np.array_split(brown, 3), axis=0) # averaging for spillover

datasets = [provo, meco, zuco, ucl_fpass, ucl_spr, brown, natstor, maze, ucl_n400, ucl_p600, michaelov, hp_split, pereira]
names = ["Provo", "Meco-en", "ZuCo", "UCL$_{ET}$", "UCL$_{SPR}$", "Brown", "NatStor, SPR", "NatStor, maze", "UCL$_{N400}$", "UCL$_{P600}$", "Michaelov", "Harry Potter", "Pereira"]

# Compute pairwise correlation and p-values between all arrays
corr = np.zeros((len(datasets), len(datasets)))
p_values = np.zeros((len(datasets), len(datasets)))

for i, data_i in enumerate(datasets):
    for j, data_j in enumerate(datasets):
        if i != j:  
            corr[i, j], p_values[i, j] = pearsonr(data_i.flatten(), data_j.flatten())
        else:
            corr[i, j] = 1  
            p_values[i, j] = 0  # No p-value for self-correlation

# Create a mask for non-significant correlations
mask_non_sig = p_values > 0.01  
#mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
corr[mask] = np.nan

df = pd.DataFrame(corr, columns=names, index=names)
#np.fill_diagonal(df.values, np.nan)
#df = df.iloc[1:, :-1]
plt.figure(figsize=(7, 7), dpi=150)
sns.heatmap(df, annot=True, fmt=".2f", cmap='jet', vmin=-.55, vmax=.55, square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 10})
#plt.title('Correlation between probe weights', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.annotate('Behav.', xy=(-0.42, 0.692), xytext=(-0.57, 0.692),
            fontsize=19, ha='center', va='center', xycoords='axes fraction', 
            bbox=dict(boxstyle='square', fc='0.8'),
            arrowprops=dict(arrowstyle='-[, widthB=4.9, lengthB=.5', lw=2.0))
plt.annotate('Neural', xy=(-.42, 0.199), xytext=(-.57, 0.199),
            fontsize=19, ha='center', va='center', xycoords='axes fraction', 
            bbox=dict(boxstyle='square', fc='0.8'),
            arrowprops=dict(arrowstyle='-[, widthB=2.9, lengthB=.5', lw=2.0))
plt.show()


####################### 
# individual ROIs, HP #
#######################

def load_weights(name, thedir=""):
    with open(thedir+'results/'+name+'_coefs_avg', 'rb') as handle:
        weights = pickle.load(handle)
    return weights

names = ["all", "PostTemp", "AntTemp", "IFG","IFGorb", "MFG", "AngularG",  "pCingulate"]
datasets = [np.mean(np.array_split(load_weights("hp_"+roi+"_med", "fMRI/"), 4), axis=0) for roi in names]

# Compute pairwise correlation and p-values between all arrays
corr = np.zeros((len(datasets), len(datasets)))
p_values = np.zeros((len(datasets), len(datasets)))

for i, data_i in enumerate(datasets):
    for j, data_j in enumerate(datasets):
        if i != j:  
            corr[i, j], p_values[i, j] = pearsonr(data_i.flatten(), data_j.flatten())
        else:
            corr[i, j] = 1  
            p_values[i, j] = 0  # No p-value for self-correlation

# Create a mask for non-significant correlations
mask_non_sig = p_values > 0.01  
#mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
corr[mask] = np.nan

df = pd.DataFrame(corr, columns=names, index=names)
#np.fill_diagonal(df.values, np.nan)
#df = df.iloc[1:, :-1]
plt.figure(figsize=(7, 7), dpi=150)
sns.heatmap(df, annot=True, fmt=".2f", cmap='jet', vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 10})
#plt.title('Correlation between probe weights', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.show()

###############
### Pereira ###
###############

names = ['all', 'LMidPostTemp', 'LPostTemp', 'LIFGorb', 'LMidAntTemp',
       'LAntTemp', 'LMFG', 'LAngG', 'LIFG']
datasets = [load_weights("pereira_"+roi+"_med", "pereira/") for roi in names]

names = ["all", "PostTemp", "AntTemp", "IFG","IFGorb", "MFG", "AngularG",  "pCingulate"]
datasets = [np.mean(np.array_split(load_weights("hp_"+roi+"_med", "fMRI/"), 4), axis=0) for roi in names]


# Compute pairwise correlation and p-values between all arrays
corr = np.zeros((len(datasets), len(datasets)))
p_values = np.zeros((len(datasets), len(datasets)))

for i, data_i in enumerate(datasets):
    for j, data_j in enumerate(datasets):
        if i != j:  
            corr[i, j], p_values[i, j] = pearsonr(data_i.flatten(), data_j.flatten())
        else:
            corr[i, j] = 1  
            p_values[i, j] = 0  # No p-value for self-correlation

# Create a mask for non-significant correlations
mask_non_sig = p_values > 0.01  
#mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
corr[mask] = np.nan

df = pd.DataFrame(corr, columns=names, index=names)
#np.fill_diagonal(df.values, np.nan)
#df = df.iloc[1:, :-1]
plt.figure(figsize=(7, 7), dpi=150)
sns.heatmap(df, annot=True, fmt=".2f", cmap='jet', vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 10})
#plt.title('Correlation between probe weights', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.show()


df.mean().sort_values(ascending=False).index



# Within ROI, across study
names = ['all', 'LMidPostTemp', 'LPostTemp', 'LIFGorb', 'LMidAntTemp','LAntTemp', 'LMFG', 'LAngG', 'LIFG']
per = {roi[1:] : load_weights("pereira_"+roi+"_med", "pereira/") for roi in names}

names = ["all", "PostTemp", "AntTemp", "IFG","IFGorb", "MFG", "AngularG",  "pCingulate"]
hp = {roi : np.mean(np.array_split(load_weights("hp_"+roi+"_med", "fMRI/"), 4), axis=0) for roi in names}

pearsonr(per["IFGorb"], hp["IFGorb"])
