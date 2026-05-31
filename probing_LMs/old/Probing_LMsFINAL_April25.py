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
import matplotlib.patheffects as path_effects
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from scipy.stats import zscore, combine_pvalues
import random
import re
import time
from statsmodels.stats.multitest import multipletests
from itertools import combinations

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge


def load_weights(name, thedir=""):
    with open(thedir+'results/'+name+'_probing_coefs_avg', 'rb') as handle: # IMPORTANT: remove "_probing" for no norm
        weights = pickle.load(handle)
    return weights

def load_embeddings(name):
    with open('probing_LMs/embeddings/'+name, 'rb') as handle:
        embs = pickle.load(handle)
    return embs

pereira = load_weights("pereira_1_med", "pereira_1/")
control = load_weights("control_med", "control_fmri/")
michaelov = load_weights("michaelov_med")
ucl_fpass = load_weights("UCL_fpass_med")
ucl_n400 = load_weights("UCL_n400_med")
ucl_p600 = load_weights("UCL_p600_med")
meco = load_weights("meco_med")
provo = load_weights("provo_med")
zuco = load_weights("zuco_med", "ZuCo/")
maze = load_weights("maze_med")

# those with spillover
ucl_spr = load_weights("UCL_spr_med")
hp = load_weights("hp_all_med", "fMRI/")
natstor_fmri = load_weights("natstorfmri_med", "natstor_fmri/")
natstor = load_weights("natstor_med")
brown = load_weights("brown_med")

# check correlations in weights for spillover data
def corr_all(lst):
    corrs = []
    for w1, w2 in combinations(lst, 2):
        r, _ = pearsonr(w1, w2)
        corrs.append(r)
    print(np.mean(corrs))
    print(corrs)
      
corr_all(np.array_split(ucl_spr, 3))
corr_all(np.array_split(hp, 4))
corr_all(np.array_split(natstor_fmri, 4))
corr_all(np.array_split(natstor, 3))
corr_all(np.array_split(brown, 3))

# ucl_spr = np.array_split(ucl_spr, 3)[-1] 
# hp_split = np.array_split(hp, 4)[-1]
# natstor_fmri_split = np.array_split(natstor_fmri, 4)[-1]
# natstor = np.array_split(natstor, 3)[-1]
# brown = np.array_split(brown, 3)[-1] 

# average
ucl_spr = np.mean(np.array_split(ucl_spr, 3), axis=0)
hp_split = np.mean(np.array_split(hp, 4), axis=0)
natstor_fmri_split = np.mean(np.array_split(natstor_fmri, 4), axis=0)
natstor = np.mean(np.array_split(natstor, 3), axis=0)
brown = np.mean(np.array_split(brown, 3), axis=0)

##########################################################################

datasets = [provo, meco, zuco, ucl_fpass, ucl_spr, brown, natstor, maze, ucl_n400, #ucl_p600, michaelov, 
            hp_split, natstor_fmri_split, pereira, control]
names = ["Provo", "Meco-en", "ZuCo", "UCL$_{ET}$", "UCL$_{SPR}$", "Brown", "NatStor$_{SPR}$", "NatStor$_{Maze}$", "UCL$_{N400}$", #"UCL$_{P600}$", "Michaelov", 
         "Harry Potter", "NatStor$_{fMRI}$", "Pereira", "Control"]

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
mask_non_sig = p_values > 0.1
#mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
corr[mask] = np.nan

df = pd.DataFrame(corr, columns=names, index=names)
#np.fill_diagonal(df.values, np.nan)
#df = df.iloc[1:, :-1]
plt.figure(figsize=(7, 7), dpi=150)
sns.heatmap(df, annot=True, fmt=".2f", cmap='jet', vmin=-.55, vmax=.55, square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 9})
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

#####################
# linguistic probes #
#####################

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium')

def get_sentence_embeddings(sentences, saveto):
    embeddings = []
    for sentence in tqdm(sentences):
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
        sentence_embedding = outputs[0].mean(dim=1)
        embeddings.append(sentence_embedding.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    with open("probing_LMs/embeddings/"+saveto, 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return embeddings

def classify(X, y, n, saveto=None, save = True, normalize=True):
    print("Ridge classifier")
    warnings.filterwarnings('ignore')
    kf = StratifiedKFold(n_splits=n)
    out_reg = []
    out_coefs = []
    out_predictions = []
    if normalize:
        X = zscore(X)
    for train_index, test_index in tqdm(kf.split(X, y), total=n):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        # Define RidgeClassifier with GridSearchCV to find the best alpha
        ridge = RidgeClassifier()
        parameters = {'alpha': (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)}
        reg = GridSearchCV(ridge, parameters, cv=5)
        
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        out_predictions.append([y_test, y_pred])
        coefs = reg.best_estimator_.coef_  #; print(coefs)
        out_coefs.append(coefs)
        out_reg.append(acc)
    coefs_avg = np.mean(out_coefs, axis=0)
    coefs_std = np.std(out_coefs, axis=0)
    if save:
        with open("probing_LMs/results/"+saveto+"_coefs_avg", 'wb') as handle:
            pickle.dump(coefs_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("probing_LMs/results/"+saveto+"_coefs_std", 'wb') as handle:
            pickle.dump(coefs_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Accuracy =", round(np.mean(out_reg), 2), "SD =", round(np.std(out_reg), 2))
    return coefs_avg

def probe(X, y, n, saveto=None, save = True, normalize=True):
    print("Ridge linear regression")
    kf = KFold(n_splits=n, shuffle=True, random_state=0)
    out_reg = []
    out_coefs = []
    out_predictions = []
    if normalize:
        X = zscore(X)
        y = zscore(y)
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        ridge = Ridge()
        parameters = {'alpha': (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)}
        reg = GridSearchCV(ridge, parameters, cv=5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r = pearsonr(y_test, y_pred)
        out_predictions.append([y_test, y_pred])
        coefs = reg.best_estimator_.coef_  #; print(coefs)
        out_coefs.append(coefs)
        out_reg.append(r)
    coefs_avg = np.mean(out_coefs, axis=0)
    coefs_std = np.std(out_coefs, axis=0)
    if save:
        with open("probing_LMs/results/"+saveto+"_coefs_avg", 'wb') as handle:
            pickle.dump(coefs_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("probing_LMs/results/"+saveto+"_coefs_std", 'wb') as handle:
            pickle.dump(coefs_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("R =", round(np.mean(out_reg), 2), "SD =", round(np.std(out_reg), 2))
    return coefs_avg

###############################################################################

###########
# SentLen #
###########

SE = "probing_LMs/SentEval/data/probing/"

sentlen = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"sentence_length.txt").readlines()])
sentlen.columns = ["train", "class", "sent"]

n = 1666 # 1200 total
classes = sentlen['class'].unique()
samples = [sentlen[sentlen['class'] == cls].sample(n) for cls in classes]
sentlen_sample = pd.concat(samples)
sentlen_sent = list(sentlen_sample["sent"])
sentlen_y = np.array([len(sent.split()) for sent in sentlen_sent])

embeddings = get_sentence_embeddings(sentlen_sent, "sentlen_10k")
embeddings = load_embeddings("sentlen_10k")

weights_sentlen = probe(embeddings, sentlen_y, 10, "sentlen_10k")
# R = 0.46 SD = 0.46

for data, name in zip(datasets, names):
    print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    
# looking at predictions
sentlen_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings).T)
    r, p = pearsonr(pred, sentlen_y)
    sentlen_scores.append([name, r, p])

##########
# BShift #
##########

bshift = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"bigram_shift.txt").readlines()])
bshift.columns = ["train", "class", "sent"]

n = 5000
classes = bshift['class'].unique()
samples = [bshift[bshift['class'] == cls].sample(n) for cls in classes]
bshift_sample = pd.concat(samples)
bshift_sent = list(bshift_sample["sent"])
bshift_y = np.array([{"O":0, "I":1}[s] for s in bshift_sample["class"]])

embeddings = get_sentence_embeddings(bshift_sent, "bshift_10k")
embeddings = load_embeddings("bshift_10k")
weights_bshift = classify(embeddings, bshift_y, 10, "bshift_10k")
# Accuracy = 0.81 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_bshift[0], data))
    
bshift_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings).T)
    r, p = pearsonr(pred, bshift_y)
    bshift_scores.append([name, r, p])

##############
# Tree depth #
##############

tdepth = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"tree_depth.txt").readlines()])
tdepth.columns = ["train", "class", "sent"]
tdepth["class"] = tdepth["class"].astype(int)

n = 1428 # 10,000 total
classes = tdepth['class'].unique()
samples = [tdepth[tdepth['class'] == cls].sample(n) for cls in classes]
tdepth_sample = pd.concat(samples)
tdepth_sent = list(tdepth_sample["sent"])
tdepth_y = np.array(tdepth_sample["class"])

embeddings = get_sentence_embeddings(tdepth_sent, "tdepth_10k")
embeddings = load_embeddings("tdepth_10k")

weights_tdepth = probe(embeddings, tdepth_y, 10, "tdepth_10k")
# R = 0.39 SD = 0.39

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_tdepth, data))
    
tdepth_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings).T)
    r, p = pearsonr(pred, tdepth_y)
    tdepth_scores.append([name, r, p])

########
# SOMO #
########

somo = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"odd_man_out.txt").readlines()])
somo.columns = ["train", "class", "sent"]

n = 5000
classes = somo['class'].unique()
samples = [somo[somo['class'] == cls].sample(n) for cls in classes]
somo_sample = pd.concat(samples)
somo_sent = list(somo_sample["sent"])
somo_y = np.array([{"C":0, "O":1}[s] for s in somo_sample["class"]])

embeddings = get_sentence_embeddings(somo_sent, "somo_10k")
embeddings = load_embeddings("somo_10k")

weights_somo = classify(embeddings, somo_y, 10, "somo_10k")
# Accuracy = 0.58 SD = 0.02

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_somo[0], data))
    
somo_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings).T)
    r, p = pearsonr(pred, somo_y)
    somo_scores.append([name, r, p])

##########
# ObjNUM #
##########

objn = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"obj_number.txt").readlines()])
objn.columns = ["train", "class", "sent"]

n = 5000
classes = objn['class'].unique()
samples = [objn[objn['class'] == cls].sample(n) for cls in classes]
objn_sample = pd.concat(samples)
objn_sent = list(objn_sample["sent"])
objn_y = np.array([{"NNS":0, "NN":1}[s] for s in objn_sample["class"]])

embeddings = get_sentence_embeddings(objn_sent, "objn_10k")
embeddings = load_embeddings("objn_10k")

weights_objn = classify(embeddings, objn_y, 10, "objn_10k")
# Accuracy = 0.83 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_objn[0], data))
    
objn_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings).T)
    r, p = pearsonr(pred, objn_y)
    objn_scores.append([name, r, p])

##########
# SubNUM #
##########

subjn = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"subj_number.txt").readlines()])
subjn.columns = ["train", "class", "sent"]

n = 5000
classes = subjn['class'].unique()
samples = [subjn[subjn['class'] == cls].sample(n) for cls in classes]
subjn_sample = pd.concat(samples)
subjn_sent = list(subjn_sample["sent"])
subjn_y = np.array([{"NNS":0, "NN":1}[s] for s in subjn_sample["class"]])

embeddings = get_sentence_embeddings(subjn_sent, "subjn_10k")
embeddings = load_embeddings("subjn_10k")

weights_subjn = classify(embeddings, subjn_y, 10, "subjn_10k")
# Accuracy = 0.9 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_subjn[0], data))
    
subjn_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings).T)
    r, p = pearsonr(pred, subjn_y)
    subjn_scores.append([name, r, p])

######### 
# Tense #
#########

tense = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"past_present.txt").readlines()])
tense.columns = ["train", "class", "sent"]

n = 5000
classes = tense['class'].unique()
samples = [tense[tense['class'] == cls].sample(n) for cls in classes]
tense_sample = pd.concat(samples)
tense_sent = list(tense_sample["sent"])
tense_y = np.array([{"PRES":0, "PAST":1}[s] for s in tense_sample["class"]])

embeddings = get_sentence_embeddings(tense_sent, "tense_10k")
embeddings = load_embeddings("tense_10k")

weights_tense = classify(embeddings, tense_y, 10, "tense_10k")
# Accuracy = 0.88 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_tense[0], data))
    
tense_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings).T)
    r, p = pearsonr(pred, tense_y)
    tense_scores.append([name, r, p])

    
###############################################################################

############
# CoordInv #
############

coord_inv = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE    +"coordination_inversion.txt").readlines()])
coord_inv.columns = ["train", "class", "sent"]

n = 5000 # 12000 total
classes = coord_inv['class'].unique()
samples = [coord_inv[coord_inv['class'] == cls].sample(n) for cls in classes]
coord_inv_sample = pd.concat(samples)
coord_inv_sent = list(coord_inv_sample["sent"])
coord_inv_y = np.array([{"O":0, "I":1}[s] for s in coord_inv_sample["class"]])

embeddings = get_sentence_embeddings(coord_inv_sent, "coord_inv_10k")
embeddings = load_embeddings("coord_inv_10k")

weights_coord_inv = classify(embeddings, coord_inv_y, 10, "coord_inv_10k")

# Accuracy = 0.65 SD = 0.02

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_coord_inv[0], data))
    
coord_inv_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings).T)
    r, p = pearsonr(pred, coord_inv_y)
    coord_inv_scores.append([name, r, p])

######################
# semantics, lexical #
######################


affective = pd.read_csv("/home/dev/Documents/Datasets/affective-norms_Warriner_et_al_2013.csv")
perceptual = pd.read_csv("/home/dev/Documents/Datasets/sensorimotor_Lynott-et-al_2020.csv")

def check_strings(word_list):
    return np.array([isinstance(word, str) for word in word_list])

w_a  = list(affective["Word"].str.lower())
w_p  = list(perceptual["Word"].str.lower())

a_filt = check_strings(w_a)
p_filt = check_strings(w_p)

w_a = list(np.array(w_a)[a_filt])

v = affective["V.Mean.Sum"][a_filt]
a = affective["A.Mean.Sum"][a_filt]
d = affective["D.Mean.Sum"][a_filt]

aud = perceptual["Auditory.mean"]
gus = perceptual["Gustatory.mean"]
hap = perceptual["Haptic.mean"]
nte = perceptual["Interoceptive.mean"]
olf = perceptual["Olfactory.mean"]
vis = perceptual["Visual.mean"]
max_p = perceptual["Max_strength.perceptual"]
exc_p = perceptual["Exclusivity.perceptual"]

foot = perceptual['Foot_leg.mean']
hand = perceptual['Hand_arm.mean']
head = perceptual['Head.mean']
mout = perceptual['Mouth.mean']
tors = perceptual['Torso.mean']
max_act = perceptual["Max_strength.action"]

a_embeddings = get_sentence_embeddings(w_a, "affective")
p_embeddings = get_sentence_embeddings(w_p, "perceptual")

a_embeddings = load_embeddings("affective")
p_embeddings = load_embeddings("perceptual")


weights_aud = probe(p_embeddings, aud, 10, "auditory")  # R = 0.32 SD = 0.32
weights_viz = probe(p_embeddings, vis, 10, "visual")    # R = 0.31 SD = 0.31
weights_gus = probe(p_embeddings, gus, 10, "gustatory") # R = 0.33 SD = 0.33
weights_olf = probe(p_embeddings, olf, 10, "olfactory") # R = 0.3 SD = 0.3
weights_hap = probe(p_embeddings, hap, 10, "haptic")    # R = 0.34 SD = 0.34
weights_max = probe(p_embeddings, max_p, 10, "max_p")   # R = 0.3 SD = 0.3
weights_nte = probe(p_embeddings, nte, 10, "nte")       # R = 0.34 SD = 0.34

weights_foot = probe(p_embeddings, foot, 10, "foot")
weights_hand = probe(p_embeddings, hand, 10, "hand")
weights_head = probe(p_embeddings, head, 10, "head")
weights_mout = probe(p_embeddings, mout, 10, "mout")
weights_tors = probe(p_embeddings, tors, 10, "tors")
weights_max_act = probe(p_embeddings, max_act, 10, "max_act")

weights_val = probe(a_embeddings, np.array(v), 10, "valence") # R = 0.36 SD = 0.36
weights_ara = probe(a_embeddings, np.array(a), 10, "arousal") # R = 0.29 SD = 0.29
weights_dom = probe(a_embeddings, np.array(d), 10, "dominance") # R = 0.32 SD = 0.32

for w in [weights_aud, weights_viz, weights_gus, weights_olf, weights_hap, weights_max]:
    print("\n\n")
    for data, name in zip(datasets, names):
        r, p = pearsonr(w, data)
        if p < .05:
            print(name, r)

for w in [weights_val, weights_ara, weights_dom]:
    print("\n\n")
    for data, name in zip(datasets, names):
        r, p = pearsonr(w, data)
        if p < .05:
            print(name, round(r, 2), round(p, 4))
            
aud_scores, viz_scores, gus_scores, olf_scores, hap_scores, max_scores, nte_scores, val_scores, ara_scores, dom_scores, foot_scores, hand_scores, head_scores, mout_scores, tors_scores, max_act_scores = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for data, name in tqdm(zip(datasets, names), total=len(names)):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred_val = np.dot(data, zscore(a_embeddings).T)
    r, p = pearsonr(pred_val, v)
    val_scores.append([name, r, p])
    
    pred_ara = np.dot(data, zscore(a_embeddings).T)
    r, p = pearsonr(pred_ara, a)
    ara_scores.append([name, r, p])
    
    pred_dom = np.dot(data, zscore(a_embeddings).T)
    r, p = pearsonr(pred_dom, d)
    dom_scores.append([name, r, p])
    
    pred_aud = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_aud, aud)
    aud_scores.append([name, r, p])
    
    pred_viz = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_viz, vis)
    viz_scores.append([name, r, p])
    
    pred_gus = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_gus, gus)
    gus_scores.append([name, r, p])
    
    pred_olf = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_olf, olf)
    olf_scores.append([name, r, p])
    
    pred_hap = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_hap, hap)
    hap_scores.append([name, r, p])
    
    pred_max = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_max, max_p)
    max_scores.append([name, r, p])
    
    pred_nte = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_nte, nte)
    nte_scores.append([name, r, p])
    
    pred_foot = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_foot, foot)
    foot_scores.append([name, r, p])
    
    pred_hand = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_hand, hand)
    hand_scores.append([name, r, p])
    
    pred_head = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_head, head)
    head_scores.append([name, r, p])
    
    pred_mout = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_mout, mout)
    mout_scores.append([name, r, p])
    
    pred_tors = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_tors, tors)
    tors_scores.append([name, r, p])
    
    pred_max_act = np.dot(data, zscore(p_embeddings).T)
    r, p = pearsonr(pred_max_act, max_act)
    max_act_scores.append([name, r, p])

####################################
# Socialness https://osf.io/2dqnj/ #
####################################

socialness = pd.read_csv("/home/dev/Documents/Datasets/socialness.csv")
social_w = [row.Word for index, row in socialness.iterrows()]
social_y = np.array([row.Mean for index, row in socialness.iterrows()])

embeddings = get_sentence_embeddings(social_w, "social")
embeddings = load_embeddings("social")

weights_social = probe(embeddings, social_y, 10, "social") # R = 0.38 SD = 0.38

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_social, data))
    
social_scores = []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings.T))
    r, p = pearsonr(pred, social_y)
    social_scores.append([name, r, p])

# ===================================================#
#                       #######                      #
#                       # SFL #                      #
#                       #######                      #
# ===================================================#

from utils import get_surprisal_tokens, tok_maker
import wikipedia
from tqdm import tqdm
import nltk
import spacy
import string
import re

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')

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

# length     
def get_l(word):
    return len(word)

def extract_sentences(num_sentences):
    sentences = []
    while len(sentences) < num_sentences:
        try:
            page = wikipedia.page(wikipedia.random())
            text = page.content
            new_sentences = nltk.sent_tokenize(text)
            if "==" in new_sentences:
                pass
            else:
                sentences.extend(new_sentences)
                print(round(len(sentences)/num_sentences, 4))
        except: 
            pass
    return sentences[:num_sentences]

def tokenize(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def get_word_embeddings(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    sentence_embedding = outputs[0]
    return sentence_embedding.cpu().numpy()[0][1:]

def get_embeddings_tokens(tokens, sep, cased=True):
    s = get_word_embeddings(" ".join(tokens))
    toks = tok_maker(tokens, sep, tokenizer, cased)
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
            out.append(np.mean(surp, axis=0))
    return out

def is_known(sentence):
    return bool(re.match("^[a-zA-Z0-9 \.\,\!\?\'\"]*$", sentence))

sentences = extract_sentences(3000)
sentences_clean = [tokenize(sent) for sent in tqdm(sentences)]
filtered_sentences = [sentence[0] for sentence in sentences_clean if is_known(sentence[0])]
sents = sorted(filtered_sentences, key=len, reverse=True)[:300]

model1 = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker1 = AutoTokenizer.from_pretrained("gpt2-medium")
s_med = np.array([s for sent in tqdm(sents) for s in get_surprisal_tokens(sent.split(), sep = "Ġ", toker=toker1, model=model1, cased=True)])
embeddings_med = np.array([list(s) for sent in tqdm(sents) for s in get_embeddings_tokens(sent.split(), sep = "Ġ")])
frequency = np.array([get_f(w) for sent in sents for w in sent.split()[1:]])
length = np.array([get_l(w) for sent in sents for w in sent.split()[1:]])

with open("probing_LMs/embeddings/frequency", 'wb') as handle:
    pickle.dump(frequency, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("probing_LMs/embeddings/surprisal", 'wb') as handle:
    pickle.dump(s_med, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("probing_LMs/embeddings/length", 'wb') as handle:
    pickle.dump(length, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("probing_LMs/embeddings/embeddings_med", 'wb') as handle:
    pickle.dump(embeddings_med, handle, protocol=pickle.HIGHEST_PROTOCOL)


embeddings_med = load_embeddings("embeddings_med")
frequency = load_embeddings("frequency")
s_med = load_embeddings("surprisal")
length = load_embeddings("length")

weights_surp = probe(embeddings_med, s_med, 10, "surprisal_10k") # R = 0.35 SD = 0.35
weights_freq = probe(embeddings_med, frequency, 10, "frequency_10k") # R = 0.35 SD = 0.35
weights_len  = probe(embeddings_med, length, 10, "length_10k") # R = 0.3 SD = 0.3

for w in [weights_surp, weights_freq, weights_len]:
    print("\n\n")
    for data, name in zip(datasets, names):
        r, p = pearsonr(w, data)
        print(name, round(r, 2), round(p, 4))
        
freq_scores, s_scores, len_scores = [], [], []
for data, name in zip(datasets, names):
    #print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))
    pred = np.dot(data, zscore(embeddings_med).T)
    r_freq, p_freq = pearsonr(pred, frequency)
    r_s, p_s = pearsonr(pred, s_med)
    r_len, p_len = pearsonr(pred, length)
    freq_scores.append([name, r_freq, p_freq])
    len_scores.append([name, r_len, p_len])
    s_scores.append([name, r_s, p_s])
    
############
# Plotting #
############

def load_weights(name, thedir=""):
    with open(thedir+'results/'+name+'_coefs_avg', 'rb') as handle:
        weights = pickle.load(handle)
    return weights

# Surface
weights_sentlen = load_weights("sentlen_10k", "probing_LMs/")
weights_surp = load_weights("surprisal_10k", "probing_LMs/")
weights_freq = load_weights("frequency_10k", "probing_LMs/")
weights_len = load_weights("length_10k", "probing_LMs/") 

# Syntax
weights_bshift = load_weights("bshift_10k", "probing_LMs/")
weights_tdepth = load_weights("tdepth_10k", "probing_LMs/")

# Semantics, words
weights_aud = load_weights("auditory", "probing_LMs/")
weights_viz = load_weights("visual", "probing_LMs/")
weights_gus = load_weights("gustatory", "probing_LMs/")
weights_olf = load_weights("olfactory", "probing_LMs/")
weights_hap = load_weights("haptic", "probing_LMs/")
weights_nte = load_weights("nte", "probing_LMs/")
weights_max = load_weights("max_p", "probing_LMs/")

weights_val = load_weights("valence", "probing_LMs/")
weights_ara = load_weights("arousal", "probing_LMs/")
weights_dom = load_weights("dominance", "probing_LMs/")

weights_social = load_weights("social", "probing_LMs/")

weights_foot = load_weights("foot", "probing_LMs/")
weights_hand = load_weights("hand", "probing_LMs/")
weights_head = load_weights("head", "probing_LMs/")
weights_mout = load_weights("mout", "probing_LMs/")
weights_tors = load_weights("tors", "probing_LMs/")
weights_max_act = load_weights("max_act", "probing_LMs/")

# Semantics, sentence
weights_somo = load_weights("somo_10k", "probing_LMs/")
weights_objn = load_weights("objn_10k", "probing_LMs/")
weights_subjn = load_weights("subjn_10k", "probing_LMs/")
weights_tense = load_weights("tense_10k", "probing_LMs/")
weights_coord_inv = load_weights("coord_inv_10k", "probing_LMs/")

weights_tasks = [weights_surp, weights_freq, weights_len, weights_sentlen, weights_tdepth, weights_bshift, weights_aud, weights_viz, weights_gus, weights_olf, weights_hap, weights_nte, weights_max, weights_foot, weights_hand, weights_head, weights_mout, weights_tors, weights_max_act, weights_val, weights_ara, weights_dom, weights_social, weights_somo, weights_objn, weights_subjn, weights_tense, weights_coord_inv]
weights_names =  ["Surprisal", "Frequency", "Length", "SentLen", "TDepth", "BShift", "Auditory", "Visual", "Gustatory", "Olfactory", "Haptic", "Enterocep.", "Max$_{perc.}$", "Foot", "Hand", "Head", "Mouth", "Torso", "Max$_{act.}$", "Valence", "Arousal", "Domin.", "Social", "SOMO", "ObjN", "SubjN", "Tense", "CoordInv"]


for w, name in zip(weights_tasks, weights_names):
    print("\n\n", name)
    for data, name in zip(datasets, names):
        try:
            r, p = pearsonr(w, data)
            if p < .05:
                print(name, r)
        except ValueError:
            r, p = pearsonr(w[0], data)
            if p < .05:
                print(name, r)

# Compute pairwise correlation and p-values between all arrays
corr = np.zeros((len(datasets), len(weights_tasks)))
p_values = np.zeros((len(datasets), len(weights_tasks)))

for i, data_i in enumerate(datasets):
    for j, data_j in enumerate(weights_tasks):
        corr[i, j], p_values[i, j] = abs(pearsonr(data_i.flatten(), data_j.flatten())[0]), pearsonr(data_i.flatten(), data_j.flatten())[1]

df_full = pd.DataFrame(corr, columns=weights_names, index=names)

# Create a mask for non-significant correlations
mask_non_sig = p_values > 0.01
mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
corr[mask] = np.nan

df = pd.DataFrame(corr, columns=weights_names, index=names)

plt.figure(figsize=(30*.7, 20*.7), dpi=150)
sns.heatmap(df, annot=True, fmt=".3f", cmap='Reds', vmin=df.min().min(), vmax=df.max().max(), square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 9})
plt.title('Weights correlation', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.xlabel("")
plt.show()

#####################
# prediction scores #
#####################

tasks_scores = [sentlen_scores, s_scores, freq_scores, len_scores, bshift_scores, tdepth_scores, aud_scores, viz_scores, gus_scores, olf_scores, hap_scores, nte_scores, max_scores, foot_scores, hand_scores, head_scores, mout_scores, tors_scores, max_act_scores, val_scores, ara_scores, dom_scores, social_scores, somo_scores, objn_scores, subjn_scores, tense_scores, coord_inv_scores]

d_task_scores = {name : score for name, score in zip(weights_names, tasks_scores)}

with open("probing_LMs/task_scores", 'wb') as handle:
    pickle.dump(d_task_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("probing_LMs/task_scores", 'rb') as handle:
    d_task_scores = pickle.load(handle)
    
tasks_scores = [d_task_scores[w_name] for w_name in weights_names]

all_task_scores = []
for task, taskname in zip(tasks_scores, weights_names):
    df_task = pd.DataFrame(task, columns = ["dataset", "r", "p"])
    rs = df_task["r"].tolist()
    all_task_scores.append(rs)
    
all_task_scores = pd.DataFrame(all_task_scores, columns = names, index = weights_names)

all_task_scores = all_task_scores.abs().T

plt.figure(figsize=(30*.7, 20*.7), dpi=150)
sns.heatmap(all_task_scores, annot=True, fmt=".3f", cmap='Blues', vmin=all_task_scores.min().min(), vmax=all_task_scores.max().max(), square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 9})
plt.title('Probe transfer results (cognitive to linguistic)', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.show()

#####################
# inverse direction # ### to continue
#####################

all_task_scores_inv = []
for dataname, dfname in zip(["provo", "meco", "zuco", "ucl_fpass", "ucl_spr", "brown", "natstor_spr", "maze", "ucl_n400", #"ucl_p600", "michaelov", 
                             "hp", "natstor_fmri", "pereira", "control"], names):
    print(dataname, "---", dfname)
    #print(dfname.upper())
    with open(f'probing_LMs/XYdata/{dataname}', 'rb') as handle:
        X_emb, y = pickle.load(handle)
        X_emb = zscore(X_emb)
        results_transfer_weights = []
        for weights, taskname in zip(weights_tasks, weights_names):
            weights = np.ravel(weights)
            try:
                pred = np.dot(weights, X_emb.T)
                r, p = pearsonr(pred, y)
                print(taskname, r, p)
                results_transfer_weights.append(r)
            except ValueError: # shape mismatch due to spillover ---- here, repeating the weigths for spill size!!
                spill_size = X_emb.shape[1] // weights.shape[0]
                print(f"{dataname} --- Shape mismatch! {spill_size} spillover")
                weights_spill = np.hstack([weights for _ in range(spill_size)])
                pred = np.dot(weights_spill, X_emb.T)
                r, p = pearsonr(pred, y)
                print(taskname, r, p)
                results_transfer_weights.append(r)
        all_task_scores_inv.append(results_transfer_weights)
        
all_task_scores_inv = pd.DataFrame(all_task_scores_inv, columns = weights_names, index = names).abs()

plt.figure(figsize=(30*.7, 20*.7), dpi=150)
sns.heatmap(all_task_scores_inv, annot=True, fmt=".3f", cmap='Blues', vmin=all_task_scores_inv.min().min(), vmax=all_task_scores_inv.max().max(), square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 10})
plt.title('Probe transfer results (linguistic to cognitive)', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.show()

###################################
# average over the two directions #
###################################

average = (all_task_scores + all_task_scores_inv / 2)

plt.figure(figsize=(30*.7, 20*.7), dpi=150)
sns.heatmap(average, annot=True, fmt=".3f", cmap='Reds', vmin=average.min().min(), vmax=average.max().max(), square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 9})
plt.title('Probe transfer results (average)', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.xlabel("")
plt.show()

############################
# STATISTICAL SIGNIFICANCE #
############################

def permutation_test(a1, a2, num_permutations):
    combined = np.concatenate([a1, a2])
    n1 = len(a1)
    observed_diff = np.mean(a1) - np.mean(a2)
    count = 0
    np.random.seed(0)
    for _ in range(num_permutations):
        permuted = np.random.permutation(combined)
        new_a1 = permuted[:n1]
        new_a2 = permuted[n1:]
        diff = np.mean(new_a1) - np.mean(new_a2)
        if abs(diff) >= abs(observed_diff):
            count += 1
    p_value = count / num_permutations
    return p_value

def analyze_columns(df, behavior_rows, brain_rows, num_permutations):
    results = []
    for col in df.columns:
        a1 = df.loc[behavior_rows, col].values
        a2 = df.loc[brain_rows, col].values
        p_value = permutation_test(a1, a2, num_permutations)
        observed_diff = np.mean(a1) - np.mean(a2)
        higher_group = "BEHAVIOR" if observed_diff > 0 else "BRAIN" if observed_diff < 0 else "Equal"
        results.append({
            "column": col,
            "p_value": p_value,
            "mean_BEHAVIOR": np.mean(a1),
            "mean_BRAIN": np.mean(a2),
            "higher_group": higher_group
        })
    return pd.DataFrame(results)

def analyze_columns_with_fdr(df, behavior_rows, brain_rows, num_permutations=10000, alpha=0.05):
    results_df = analyze_columns(df, behavior_rows, brain_rows, num_permutations=num_permutations)
    reject, pvals_corrected, _, _ = multipletests(results_df['p_value'], alpha=alpha, method='fdr_bh')
    results_df['p_value_FDR'] = pvals_corrected
    results_df['reject_FDR'] = reject
    return results_df

behavior = ['Provo', 'Meco-en', 'ZuCo', 'UCL$_{ET}$', 'UCL$_{SPR}$', 'Brown',
            'NatStor$_{SPR}$', 'NatStor$_{Maze}$']
brain = ['UCL$_{N400}$', 'Harry Potter', 'NatStor$_{fMRI}$', 'Pereira', 'Control']

results_df = analyze_columns_with_fdr(all_task_scores, behavior, brain)
results_df_average = analyze_columns_with_fdr(average, behavior, brain)
results_df_weights = analyze_columns_with_fdr(df_full, behavior, brain)
results_df_inv = analyze_columns_with_fdr(all_task_scores_inv, behavior, brain)

######################
# PLOT AS DENDROGRAM #
######################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

def create_custom_dendrogram_heatmap(
    data_df,
    row_hierarchy, 
    col_hierarchy, 
    figsize=(30, 20), 
    dpi=150,
    results_df=None,
    p_column = "p_value"
):
    """
    Creates a heatmap with custom (multi-branch) dendrograms on the left (rows)
    and bottom (columns). Key points:
      - Numeric annotations on the heatmap cells are reintroduced.
      - Standard tick labels are removed; leaf labels are drawn in the dendrogram.
      - The final (leaf-level) row labels are right-aligned.
      - The final (leaf-level) column labels are vertical.
      - If results_df is provided, a barplot showing the difference 
        (mean_BEHAVIOR - mean_BRAIN) is added on top, with error bars (standard error).
          - Bars are red if positive and significant (p < 0.05),
            blue if negative and significant, and gray otherwise.
          - A thick black horizontal line is drawn at 0.
        - The barplot’s y-axis is labeled r'$\Delta R$' with ticks at -0.1, 0.1, 0.2, 0.3, and 0.4.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Create main axes for the heatmap and dendrograms.
    ax_heatmap    = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax_row_dendro = fig.add_axes([0.1, 0.2, 0.1, 0.6])
    ax_col_dendro = fig.add_axes([0.2, 0.1, 0.6, 0.1])
    ax_cbar       = fig.add_axes([0.81, 0.2, 0.02, 0.6])

    # Build maps from row/col labels to integer positions.
    row_label_to_pos = {lab: i for i, lab in enumerate(data_df.index)}
    col_label_to_pos = {lab: i for i, lab in enumerate(data_df.columns)}

    # Recursively traverse hierarchy to build bracket intervals and label positions.
    def labeled_traverse(node, parent_label, label_to_pos_map, depth, connections, level_texts):
        positions = []
        if isinstance(node, dict):
            for sublabel, subnode in node.items():
                child_positions = labeled_traverse(
                    subnode,
                    sublabel,
                    label_to_pos_map,
                    depth+1,
                    connections,
                    level_texts
                )
                positions.extend(child_positions)
            if positions:
                mn, mx = min(positions), max(positions)
                connections.append((mn, mx, depth, parent_label))
                level_texts.append(((mn + mx)*0.5, depth, parent_label))
            return positions
        elif isinstance(node, list):
            leaf_positions = []
            for leaf_label in node:
                if leaf_label in label_to_pos_map:
                    leaf_positions.append(label_to_pos_map[leaf_label])
            if leaf_positions:
                mn, mx = min(leaf_positions), max(leaf_positions)
                connections.append((mn, mx, depth, parent_label))
                level_texts.append(((mn + mx)*0.5, depth, parent_label))
                for lf_label in node:
                    if lf_label in label_to_pos_map:
                        lf_pos = label_to_pos_map[lf_label]
                        connections.append((lf_pos, lf_pos, depth+1, lf_label))
                        level_texts.append((lf_pos, depth+1, lf_label))
            return leaf_positions
        return positions

    # Build row hierarchy.
    row_connections = []
    row_texts = []
    for top_label, subtree in row_hierarchy.items():
        labeled_traverse(subtree, top_label, row_label_to_pos, 0, row_connections, row_texts)
    max_row_level = max((c[2] for c in row_connections), default=0)

    # Build column hierarchy.
    col_connections = []
    col_texts = []
    for top_label, subtree in col_hierarchy.items():
        labeled_traverse(subtree, top_label, col_label_to_pos, 0, col_connections, col_texts)
    max_col_level = max((c[2] for c in col_connections), default=0)

    # Identify leaf-level connections for alignment.
    row_leaf_connections = set()
    for (st, en, dp, lab) in row_connections:
        if st == en:
            row_leaf_connections.add((st, dp, lab))
    col_leaf_connections = set()
    for (st, en, dp, lab) in col_connections:
        if st == en:
            col_leaf_connections.add((st, dp, lab))

    # 5) Plot heatmap with numeric annotations.
    sns.heatmap(
        data_df,
        ax=ax_heatmap,
        cmap='Reds',
        vmin=data_df.values.min(),
        vmax=data_df.values.max(),
        square=True,
        linewidths=6,
        cbar_ax=ax_cbar,
        cbar_kws={"shrink": 1.0, "extend": "both"}
    )
    # Remove standard tick labels.
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_cbar.tick_params(labelsize=30)

    # ----- New code: add barplot on top if results_df is provided -----
    if results_df is not None:
        # Helper function to extract leaves from a hierarchy.
        def get_leaves(hier):
            leaves = []
            if isinstance(hier, dict):
                for k, v in hier.items():
                    leaves.extend(get_leaves(v))
            elif isinstance(hier, list):
                leaves.extend(hier)
            return leaves

        behavior_leaves = get_leaves(row_hierarchy.get("Behavior", []))
        brain_leaves = get_leaves(row_hierarchy.get("Brain", []))
        
        cols = data_df.columns
        differences = []
        stderr = []
        colors = []
        for col in cols:
            row_sel = results_df[results_df['column'] == col]
            if not row_sel.empty:
                diff = row_sel['mean_BEHAVIOR'].values[0] - row_sel['mean_BRAIN'].values[0]
                p_val = row_sel[p_column].values[0]
            else:
                diff = 0
                p_val = 1.0
            differences.append(diff)
            a1 = data_df.loc[behavior_leaves, col].values
            a2 = data_df.loc[brain_leaves, col].values
            se = np.sqrt(np.var(a1, ddof=1)/len(a1) + np.var(a2, ddof=1)/len(a2))
            stderr.append(se)
            if p_val < 0.05:
                if diff > 0:
                    colors.append('red')
                elif diff < 0:
                    colors.append('blue')
                else:
                    colors.append('gray')
            else:
                colors.append('gray')
        heatmap_pos = ax_heatmap.get_position()
        bar_height = 0.3 * heatmap_pos.height
        ax_bar = fig.add_axes([heatmap_pos.x0, heatmap_pos.y0 + heatmap_pos.height, 
                               heatmap_pos.width, bar_height])
        x = np.arange(len(cols)) + 0.5  # center positions for each column
        ax_bar.bar(x, differences, width=0.8, color=colors, align='center', 
                   zorder=5, yerr=stderr, capsize=5, linewidth=3, error_kw={'linewidth': 3}, alpha = .8)
        ax_bar.axhline(0, color='black', linewidth=3, zorder = 7)
        ax_bar.set_xlim(ax_heatmap.get_xlim())
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.set_ylabel(r'$\Delta$R', fontsize=35)
        ax_bar.set_yticks([-0.1, 0.1, 0.3, 0.5])
        ax_bar.tick_params(axis='y', labelsize=30)
        ax_bar.set_xticks([])

    # ----------------------------------------------------------------
    # Row dendrogram
    #   - The top row in data_df appears on top (no y-axis inversion).
    # ----------------------------------------------------------------
    ax_row_dendro.set_ylim(*ax_heatmap.get_ylim())
    ax_row_dendro.axis('off')

    level_colors = ["lightgrey", "white", "white"]
    def get_level_color(d):
        if d < len(level_colors):
            return level_colors[d]
        return level_colors[-1]

    # Draw row brackets.
    bracket_space = 700
    for (start_idx, end_idx, depth, label) in row_connections:
        if depth == 2:
            continue
        y1 = start_idx + 0.5
        y2 = end_idx   + 0.5
        if y1 > y2:
            y1, y2 = y2, y1
        inv = max_row_level - depth
        x_right = 0.95
        x_left  = x_right - (inv+1)*(bracket_space/(max_row_level+1))
        ax_row_dendro.plot([x_left, x_right], [y1, y1], color='black', lw=3.5)
        ax_row_dendro.plot([x_left, x_right], [y2, y2], color='black', lw=3.5)
        ax_row_dendro.plot([x_left, x_left], [y1, y2], color='black', lw=3.5)

    # Draw row labels.
    for (mid_idx, depth, label) in row_texts:
        y_mid = mid_idx + 0.5
        inv   = max_row_level - depth
        x_text = 0.95 - (inv+1)*(bracket_space/(max_row_level+1)) - 0.01
        is_leaf = (round(mid_idx), depth, label) in row_leaf_connections
        if depth == 0:
            fsize = 26
            fw    = 'bold'
        else:
            fsize = 21
            fw    = 'bold'
        color_box = get_level_color(depth)
        bbox_props = dict(
            boxstyle="round,pad=0.3",
            fc=color_box,
            ec="black",
            lw=3,
            alpha=1.0
        )
        ha_style = 'right' if depth == 2 else 'center'
        add_x = 230 if depth == 2 else 0
        txt = ax_row_dendro.text(
            x_text+add_x, 
            y_mid,
            label,
            ha=ha_style,
            va='center',
            fontsize=fsize,
            fontweight=fw,
            bbox=bbox_props
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # ----------------------------------------------------------------
    # Column dendrogram
    #   - Largest depth near the heatmap, top-level further down.
    # ----------------------------------------------------------------
    ax_col_dendro.set_xlim(*ax_heatmap.get_xlim())
    ax_col_dendro.axis('off')

    col_bracket_space = 700
    for (start_idx, end_idx, depth, label) in col_connections:
        if depth == 2:
            continue
        x1 = start_idx + 0.5
        x2 = end_idx   + 0.5
        if x1 > x2:
            x1, x2 = x2, x1
        inv = max_col_level - depth
        y_top = 0.0
        y_bottom = y_top - (inv+1)*(col_bracket_space/(max_col_level+1))
        ax_col_dendro.plot([x1, x1], [y_bottom, y_top], color='black', lw=3.5)
        ax_col_dendro.plot([x2, x2], [y_bottom, y_top], color='black', lw=3.5)
        ax_col_dendro.plot([x1, x2], [y_bottom, y_bottom], color='black', lw=3.5)

    for (mid_idx, depth, label) in col_texts:
        is_leaf = depth >= 2
        if label == "CoordInv":
            is_leaf = True
            depth = 2
        x_mid = mid_idx + 0.5
        inv   = max_col_level - depth
        y_text = -(inv+1)*(col_bracket_space/(max_col_level+1)) - 0.01
        if label == "Discourse":
            x_mid = x_mid + 1.05
        if depth == 0:
            fsize = 26
            fw    = 'bold'
        else:
            fsize = 21
            fw    = 'bold'
        color_box = get_level_color(depth)
        bbox_props = dict(
            boxstyle="round,pad=0.3",
            fc=color_box,
            ec="black",
            lw=3,
            alpha=1.0
        )
        if is_leaf:
            rotation_angle = 90
            ha_style = 'center'
            va_style = 'top'
        else:
            rotation_angle = 0
            ha_style = 'center'
            va_style = 'center'
        add_y = 220 if is_leaf else 0
        txt = ax_col_dendro.text(
            x_mid,
            y_text+add_y,
            label,
            ha=ha_style,
            va=va_style,
            rotation=rotation_angle,
            fontsize=fsize,
            fontweight=fw,
            bbox=bbox_props
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # ----------------------------------------------------------------
    # Final layout adjustments.
    # ----------------------------------------------------------------
    fig.canvas.draw()
    heatmap_pos = ax_heatmap.get_position()
    ax_row_dendro.set_position([
        heatmap_pos.x0 - 0.2,
        heatmap_pos.y0,
        0.2,
        heatmap_pos.height
    ])
    ax_col_dendro.set_position([
        heatmap_pos.x0,
        heatmap_pos.y0 - 0.25,
        heatmap_pos.width,
        0.25
    ])
    ax_cbar.set_position([
        heatmap_pos.x0 + heatmap_pos.width + 0.01,
        heatmap_pos.y0,
        0.02,
        heatmap_pos.height
    ])

    return fig

row_hierarchy = {
    "Behavior": {
        "Eye-tracking": ["Provo", "Meco-en", "ZuCo", "UCL$_{ET}$"],
        "Self-paced R.": ["UCL$_{SPR}$", "Brown", "NatStor$_{SPR}$"],
        "Maze": ["NatStor$_{Maze}$"]
    },
    "Brain": {
        "EEG": ["UCL$_{N400}$"],
        "fMRI": ["Harry Potter", "NatStor$_{fMRI}$", "Pereira", "Control"]
    }
}
col_hierarchy = {
    "Form": {
    "Surface": ["SentLen", "Surprisal", "Frequency", "Length"],
    "Syntax": ["BShift", "TDepth"]},
    "Semantics": {
        "Sensory": [
            "Auditory", "Visual", "Gustatory", "Olfactory",
            "Haptic", "Enterocep.", "Max$_{perc.}$"
        ],
        "Motor": [
            "Foot", "Hand", "Head", "Mouth", "Torso", "Max$_{act.}$"
        ],
        "Affective": [
            "Valence", "Arousal", "Domin.", "Social"
        ],
        "Sentence": [
            "SOMO", "ObjN", "SubjN", "Tense"
        ]
    },
    "Discourse": ["CoordInv"]
}

fig = create_custom_dendrogram_heatmap(all_task_scores, row_hierarchy, col_hierarchy, results_df=results_df, p_column = "p_value_FDR")

fig = create_custom_dendrogram_heatmap(average, row_hierarchy, col_hierarchy, results_df=results_df_average, p_column = "p_value_FDR")

# test stats sig
form = ["SentLen", "Surprisal", "Frequency", "Length", "BShift", "TDepth"]
semantics = ["Auditory", "Visual", "Gustatory", "Olfactory", "Haptic", "Enterocep.", "Max$_{perc.}$", "Foot", "Hand", "Head", "Mouth", "Torso", "Max$_{act.}$", "Valence", "Arousal", "Domin.", "Social", "SOMO", "ObjN", "SubjN", "Tense"]

behavior_form_vals = np.abs(average.loc[behavior, form].values.flatten())
behavior_semantics_vals = np.abs(average.loc[behavior, semantics].values.flatten())

brain_form_vals = np.abs(average.loc[brain, form].values.flatten())
brain_semantics_vals = np.abs(average.loc[brain, semantics].values.flatten())

pval_behavior_form_vs_sem = permutation_test(behavior_form_vals, behavior_semantics_vals, num_permutations=10000)
pval_brain_sem_vs_form = permutation_test(brain_semantics_vals, brain_form_vals, num_permutations=10000)

pval_behavior_form_vs_brain = permutation_test(behavior_form_vals, brain_form_vals, num_permutations=10000)
pval_brain_sem_vs_behavior = permutation_test(brain_semantics_vals, behavior_semantics_vals, num_permutations=10000)


############################
# Transfer across datasets #
############################

cross_data_scores = []
cross_data_scores_p = []
for dataname, dfname in zip(["provo", "meco", "zuco", "ucl_fpass", "ucl_spr", "brown", "natstor_spr", "maze", "ucl_n400", #"michaelov", 
                             "hp", "natstor_fmri", "pereira_1", "control"], names):
    print(dfname.upper())
    with open(f'probing_LMs/XYdata/{dataname}', 'rb') as handle:
        X_emb, y = pickle.load(handle)
        X_emb = zscore(X_emb)
        results_transfer_weights = []
        results_transfer_weights_p = []
        for weights, taskname in zip(datasets, names):
            weights = np.ravel(weights)
            try:
                pred = np.dot(weights, zscore(X_emb.T))
                r, p = pearsonr(pred, y)
                print(taskname, r, p)
                results_transfer_weights.append(r)
                results_transfer_weights_p.append(p)
            except ValueError: # shape mismatch due to spillover
                spill_size = X_emb.shape[1] // weights.shape[0]
                weights_spill = np.hstack([weights for _ in range(spill_size)])
                pred = np.dot(weights_spill, zscore(X_emb.T))
                r, p = pearsonr(pred, y)
                print(taskname, r, p)
                results_transfer_weights.append(r)
                results_transfer_weights_p.append(p)
        cross_data_scores.append(results_transfer_weights)
        cross_data_scores_p.append(results_transfer_weights_p)
        
cross_data_scores = pd.DataFrame(cross_data_scores, columns = names, index = names)
cross_data_scores_p = pd.DataFrame(cross_data_scores_p, columns = names, index = names)
mask_non_sig = cross_data_scores_p.applymap(lambda x: np.nan if x > 0.01 else x)
cross_data_scores_masked = cross_data_scores.mask(mask_non_sig.isna())

plt.figure(figsize=(7, 7), dpi=150)
sns.heatmap(cross_data_scores_masked, annot=True, fmt=".2f", cmap='jet', vmin=-cross_data_scores_masked.max().max(), vmax=cross_data_scores_masked.max().max(), square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 9})
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

# test stats sig
bb = cross_data_scores.loc[behavior, behavior].values
bb_vals = np.abs(bb[~np.eye(len(bb), dtype=bool)])  # exclude diagonal

bx_vals = np.abs(cross_data_scores.loc[behavior, brain].values.flatten())
xb_vals = np.abs(cross_data_scores.loc[brain, behavior].values.flatten())
xx_vals = np.concatenate([bx_vals, xb_vals])

brbr = cross_data_scores.loc[brain, brain].values
brbr_vals = np.abs(brbr[~np.eye(len(brbr), dtype=bool)])  # exclude diagonal

# Compute p-values
pval_bb_vs_xx = permutation_test(bb_vals, xx_vals, num_permutations = 10000)
pval_brbr_vs_xx = permutation_test(brbr_vals, xx_vals, num_permutations=10000)

###############################################
# check if dimensions are actually correlated #
###############################################

out = []
for dataname, dfname in zip(["provo", "meco", "zuco", "ucl_fpass", "ucl_spr", "brown", "natstor_spr", "maze", "ucl_n400", "ucl_p600", "michaelov", "hp", "natstor_fmri", "pereira", "control"], names):
    print(dfname.upper())
    with open(f'probing_LMs/XYdata/{dataname}', 'rb') as handle:
        X_emb, y = pickle.load(handle)
        X_emb = zscore(X_emb)
        if X_emb.shape[1] == 1024:
            out.append(X_emb)
        else:
            n = X_emb.shape[1] // 1024
            X_emb = np.array_split(X_emb, n)[n-1]
out = np.vstack(out)  
out = pd.DataFrame(out)

correlation_matrix = out.corr()

mask = np.ones(correlation_matrix.shape, dtype=bool)
np.fill_diagonal(mask, False)
print("mean corr =", np.mean(correlation_matrix.abs().values[mask]))
print("std corr =", np.std(correlation_matrix.abs().values[mask]))

print("corrs higher than .2 =", sum(correlation_matrix.abs().values[mask] > .2) / len(correlation_matrix.abs().values[mask]))
