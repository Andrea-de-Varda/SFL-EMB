# In general, I use avg representations

###########
# Surface #
###########

# https://github.com/facebookresearch/SentEval/tree/main/data/probing

# SentLen
    # The first task is to predict the length of sentences in terms of number of words. Group sentences into 6 equal-width bins by length, and treat SentLen as a 6-way classification task
    
# Word Content (skip)
    # tests whether it is possible to recover information about the original words in the sentence from its embedding. We picked 1000 mid-frequency words from the source corpus vocabulary (the words with ranks between 2k and 3k when sorted by frequency), and sampled equal numbers of sentences that contain one and only one of these words. The task is to tell which of the 1k words a sentence contains (1k-way classification).
    
###########
# Lexical #
###########

# Linear probes predicting S, F, and L
    
##########
# Syntax #
##########

# Bigram Shift 
    # The bigram shift (BShift) task tests whether an encoder is sensitive to legal word orders. In this binary classification problem, models must distinguish intact sentences sampled from the corpus from sentences where we inverted two random adjacent words (“What you are doing out there?”).
    
# Tree depth 
    # The tree depth (TreeDepth) task checks whether an encoder infers the hierarchical structure of sentences, and in particular whether it can group sentences by the depth of the longest path from root to any leaf. Since tree depth is naturally correlated with sentence length, we de-correlate these variables through a structured sampling procedure. In the resulting data set, tree depth values range from 5 to 12, and the task is to categorize sentences into the class corresponding to their depth (8 classes). As an example, the following is a long (22 tokens) but shallow (max depth: 5) sentence: “[1 [2 But right now, for the time being, my past, my fears, and my thoughts [3 were [4 my [5business]]].]]” (the outermost brackets correspond to the ROOT and S nodes in the parse).

# Top constituent  (skip)
    # In the top constituent task (TopConst), sentences must be classified in terms of the sequence of top constituents immediately below the sentence (S) node. An encoder that successfully addresses this challenge is not only capturing latent syntactic structures, but clustering them by constituent types. TopConst was introduced by Shi et al. (2016). Following them, we frame it as a 20-way classification problem: 19 classes for the most frequent top constructions, and one for all other constructions. As an example, “[Then] [very dark gray letters on a black screen] [appeared] [.]” has top constituent sequence: “ADVP NP VP .”.
    
# Consider adding SuperPOSTagging (function vs content work)
    
######################
# Semantics, lexical #
######################

# Predict affective and perceptual information

#######################
# Semantics, sentence #
#######################

# Semantic tasks (Tense, SubjNum, and ObjNum) from SentEval (Conneau et al., 2018) https://github.com/facebookresearch/SentEval/tree/main/data/probing

# Tense task
    #  asks for the tense of the main-clause verb (VBP/VBZ forms are labeled as present, VBD as past). No target form occurs across the train/dev/test split, so that classifiers cannot rely on specific words
    
# Subject number
    # The subject number (SubjNum) task focuses on the number of the subject of the main clause (number in English is more often explicitly marked on nouns than verbs).
    
# Object number 
    # (ObjNum) tests for the number of the direct object of the main clause (again, avoiding lexical overlap). To solve the previous tasks correctly, an encoder must not only capture tense and number, but also extract structural information (about the main clause and its arguments)
    
# SOMO
    # In the semantic odd man out (SOMO) task, we modified sentences by replacing a random noun or verb o with another noun or verb r. To make the task more challenging, the bigrams formed by the replacement with the previous and following words in the sentence have frequencies that are comparable (on a log-scale) with those of the original bigrams. 

###############################################################
# Discourse: https://aclanthology.org/2021.naacl-main.301.pdf #
###############################################################

# Next sentence prediction: 
        # Similar to the nextsentence prediction (NSP) objective in BERT pretraining, but here we frame it as a 4-way classification task, with one positive and 3 negative candidates for the next sentence. The preceding context takes the form of between 2 and 8 sentences, but the candidates are always single sentences.
        
# Sentence ordering:
    # classify if sentences have been scrambled
    
# Cloze story test:
    # Given a 4-sentence story context, pick the best ending from two possible options (Mostafazadeh et al., 2016; Sharma et al., 2018).
    
# CoordInv (IMPORTANT: pragmatic, but from SentEval task)
    # The coordination inversion (CoordInv) benchmark contains sentences made of two coordinate clauses. In half of the sentences, we inverted the order of the clauses. The task is to tell whether a sentence is intact or modified. Sentences are balanced in terms of clause length, and no sentence appears in both original and inverted versions. As an example, original “They might be only memories, but I can still feel each one” becomes: “I can still feel each one, but they might be only memories.” Often, addressing CoordInv requires an understanding of broad discourse and pragmatic factors.
    
    
# Multiclass ridge https://stackoverflow.com/questions/61222686/scikit-learns-ridge-clasifiers-working-for-multi-class-not-clear
    
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



dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import test_model_Ridge


def load_weights(name, thedir=""):
    with open(thedir+'results/'+name+'_coefs_avg', 'rb') as handle:
        weights = pickle.load(handle)
    return weights

hp = load_weights("hp_all_med_probing", "fMRI/")
hp_split = np.mean(np.array_split(hp, 4), axis=0) # averaging for spillover
#hp_split = np.array_split(hp, 4)[-1]
pereira = load_weights("pereira_all_med_probing", "pereira/")
michaelov = load_weights("michaelov_med_probing")
ucl_fpass = load_weights("UCL_fpass_med_probing")
ucl_n400 = load_weights("UCL_n400_med_probing")
ucl_p600 = load_weights("UCL_p600_med_probing")
ucl_spr = load_weights("UCL_spr_med_probing")
ucl_spr = np.mean(np.array_split(ucl_spr, 3), axis=0) # averaging for spillover
meco = load_weights("meco_med_probing")
provo = load_weights("provo_med_probing")
zuco = load_weights("zuco_med_probing", "ZuCo/")
maze = load_weights("maze_med_probing")
natstor = load_weights("natstor_med_probing")
natstor = np.mean(np.array_split(natstor, 3), axis=0) # averaging for spillover
brown = load_weights("brown_med_probing")
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
mask_non_sig = p_values > 0.05  
#mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
corr[mask] = np.nan

df = pd.DataFrame(corr, columns=names, index=names)
#np.fill_diagonal(df.values, np.nan)
#df = df.iloc[1:, :-1]
plt.figure(figsize=(10, 10), dpi=150)
sns.heatmap(df, annot=True, fmt=".2f", cmap='jet', vmin=-.55, vmax=.55, square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 10})
plt.title('Correlation between probe weights', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.annotate('Behav.', xy=(-0.29, 0.692), xytext=(-0.44, 0.692),
            fontsize=19, ha='center', va='center', xycoords='axes fraction', 
            bbox=dict(boxstyle='square', fc='0.8'),
            arrowprops=dict(arrowstyle='-[, widthB=7, lengthB=.5', lw=2.0))
plt.annotate('Neural', xy=(-.29, 0.199), xytext=(-.44, 0.199),
            fontsize=19, ha='center', va='center', xycoords='axes fraction', 
            bbox=dict(boxstyle='square', fc='0.8'),
            arrowprops=dict(arrowstyle='-[, widthB=4.2, lengthB=.5', lw=2.0))
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

def classify(X, y, n, saveto=None, save = True):
    print("Ridge classifier")
    warnings.filterwarnings('ignore')
    kf = StratifiedKFold(n_splits=n)
    out_reg = []
    out_coefs = []
    out_predictions = []
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

def probe(X, y, n, saveto=None, save = True):
    print("Ridge linear regression")
    kf = KFold(n_splits=n, shuffle=True, random_state=0)
    out_reg = []
    out_coefs = []
    out_predictions = []
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

# def probe(X, y, n, saveto=None, save = True):
#     print("Ridge linear regression")
#     kf = KFold(n_splits=n, shuffle=True, random_state=0)
#     out_reg = []
#     out_coefs = []
#     out_predictions = []
#     for train_index, test_index in tqdm(kf.split(X), total=n):
#         X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#         reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
#         reg.fit(X_train, y_train)
#         y_pred = reg.predict(X_test)
#         r = pearsonr(y_test, y_pred)
#         out_predictions.append([y_test, y_pred])
#         coefs = reg.coef_  #; print(coefs)
#         out_coefs.append(coefs)
#         out_reg.append(r)
#     coefs_avg = np.mean(out_coefs, axis=0)
#     coefs_std = np.std(out_coefs, axis=0)
#     if save:
#         with open("probing_LMs/results/"+saveto+"_coefs_avg", 'wb') as handle:
#             pickle.dump(coefs_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         with open("probing_LMs/results/"+saveto+"_coefs_std", 'wb') as handle:
#             pickle.dump(coefs_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("R =", round(np.mean(out_reg), 2), "SD =", round(np.std(out_reg), 2))
#     return coefs_avg

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

weights_sentlen = probe(embeddings, sentlen_y, 10, "sentlen_10k")
weights_sentlen = load_weights("sentlen_10k", "probing_LMs/")
# R = 0.47 SD = 0.47

for data, name in zip(datasets, names):
    print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))

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
weights_bshift = classify(embeddings, bshift_y, 10, "bshift_10k")
weights_bshift = load_weights("bshift_10k", "probing_LMs/")

# Accuracy = 0.81 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_bshift[0], data))

############
# Scambled #
############

sents = list(np.array(bshift_sent)[bshift_y == 0])

shuffled = []
for s in sents:
    b = s.split()
    random.shuffle(b)
    shuffled.append(b)

shuffled_sent = sents + shuffled
embeddings = get_sentence_embeddings(shuffled_sent, "shuf_10k")
weights_shuf = classify(embeddings, bshift_y, 10, "shuf_10k")
weights_shuf = load_weights("shuf_10k", "probing_LMs/")

# Accuracy = 1.0 SD = 0.0

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_shuf[0], data))

##############
# Tree depth # # RERUN
##############

tdepth = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"tree_depth.txt").readlines()])
tdepth.columns = ["train", "class", "sent"]
tdepth["class"] = tdepth["class"].astype(int)

n = 1428 # 10,000 total
classes = tdepth['class'].unique()
samples = [tdepth[tdepth['class'] == cls].sample(n) for cls in classes]
tdepth_sample = pd.concat(samples)
tdepth_sent = list(tdepth_sample["sent"])
tdepth_y = np.array(tdepth["class"])

embeddings = get_sentence_embeddings(tdepth_sent, "tdepth_10k")
weights_tdepth = probe(embeddings, tdepth_y, 10, "tdepth_10k")
weights_tdepth = load_weights("tdepth_10k", "probing_LMs/")

# R = 0.16 SD = 0.22

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_tdepth, data))

########
# SOMO # RERUN
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
weights_somo = classify(embeddings, somo_y, 10, "somo_10k")
weights_somo = load_weights("somo_10k", "probing_LMs/")

# Accuracy = 0.58 SD = 0.02

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_somo[0], data))

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
weights_objn = classify(embeddings, objn_y, 10, "objn_10k")
weights_objn = load_weights("objn_10k", "probing_LMs/")

# Accuracy = 0.83 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_objn[0], data))

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
weights_subjn = classify(embeddings, subjn_y, 10, "subjn_10k")
weights_subjn = load_weights("subjn_10k", "probing_LMs/")

# Accuracy = 0.9 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_subjn[0], data))

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
weights_tense = classify(embeddings, tense_y, 10, "tense_10k")
weights_tense = load_weights("tense_10k", "probing_LMs/")


for data, name in zip(datasets, names):
    print(name, pearsonr(weights_tense[0], data))
    
    
###############################################################################

############
# CoordInv # RERUN
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
weights_coord_inv = classify(embeddings, coord_inv_y, 10, "coord_inv_10k") 
weights_coord_inv = load_weights("coord_inv_10k", "probing_LMs/")

# Accuracy = 0.65 SD = 0.02

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_coord_inv[0], data))
    
###############################################################################

#####################
# Sentence ordering #
#####################

with open("probing_LMs/discourse/sentence_ordering.json") as f:
    d = json.load(f)

d1 = []
for obj in d:
    ctx = " ".join([re.sub("\n", "", line) for line in obj["sentences"]]).split()
    d1.append([len(ctx), obj])
d1 = sorted(d1, key=lambda x: x[0])
d1 = [d[1] for d in d1]

sent1 = []; sent2 = []
for obj in d[:5000]:
    sent = obj["sentences"]
    order = obj["order"]
    paired = list(zip(sent, order))
    sorted_pair = sorted(paired, key=lambda x: x[1])
    sorted_a = [element[0] for element in sorted_pair]
    sent1.append(" ".join(sorted_a))
    sent2.append(" ".join(sent))
    
embeddings1 = get_sentence_embeddings(sent1, "ordering_1_10k")
embeddings2 = get_sentence_embeddings(sent2, "ordering_2_10k")

sent_ord_y = np.array([1]*len(embeddings1)+[0]*len(embeddings2))

embeddings = np.concatenate([embeddings1, embeddings2])
print(embeddings.shape)

weights_ordering = classify(embeddings, sent_ord_y, 10, "ordering_10k")
weights_ordering = load_weights("ordering_10k", "probing_LMs/")
# Accuracy = 0.37 SD = 0.02 terrible accuracy

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_ordering[0], data))

#######
# NSP # 
#######

with open("probing_LMs/discourse/nsp.json") as f:
    d = json.load(f)

d1 = []
for obj in d:
    ctx = " ".join([re.sub("\n", "", line) for line in obj["context"]]).split()
    d1.append([len(ctx), obj])
d1 = sorted(d1, key=lambda x: x[0])
d1 = [d[1] for d in d1]

right_embeddings = []
wrong_embeddings = []
for obj in tqdm(d1[:2500]):
    ctx = [re.sub("\n", "", line) for line in obj["context"]]
    ctx_len = len(tokenizer.encode(" ".join(ctx), return_tensors='pt')[0])-1
    answ = sorted(obj["answer"], key=lambda x : x[0], reverse=True)
    right = " ".join(ctx + [re.sub("\n", "",answ[0][1])])
    wrong = " ".join(ctx + [re.sub("\n", "",answ[1][1])])
    input_ids_right = tokenizer.encode(right, return_tensors='pt')
    input_ids_wrong = tokenizer.encode(wrong, return_tensors='pt')
    with torch.no_grad():
        outputs_right = model(input_ids_right)[0][0][ctx_len:].mean(dim=0).tolist()
        outputs_wrong = model(input_ids_wrong)[0][0][ctx_len:].mean(dim=0).tolist()
    right_embeddings.append(outputs_right); wrong_embeddings.append(outputs_wrong)


right_embeddings1 = np.array(right_embeddings)
wrong_embeddings1 = np.array(wrong_embeddings)

nsp_y = np.array([1]*len(right_embeddings)+[0]*len(wrong_embeddings))

embeddings = np.concatenate([right_embeddings1, wrong_embeddings1])

weights_nsp = classify(embeddings, nsp_y, 10, "nsp_10k") # Accuracy = 0.86 SD = 0.02
weights_nsp = load_weights("nsp_10k", "probing_LMs/")

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_nsp[0], data))

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

a_embeddings = get_sentence_embeddings(w_a, "affective")
p_embeddings = get_sentence_embeddings(w_p, "perceptual")

weights_aud = probe(p_embeddings, aud, 10, "auditory")  # R = 0.32 SD = 0.32
weights_viz = probe(p_embeddings, vis, 10, "visual")    # R = 0.31 SD = 0.31
weights_gus = probe(p_embeddings, gus, 10, "gustatory") # R = 0.34 SD = 0.34
weights_olf = probe(p_embeddings, olf, 10, "olfactory") # R = 0.31 SD = 0.31
weights_hap = probe(p_embeddings, hap, 10, "haptic")    # R = 0.34 SD = 0.34
weights_max = probe(p_embeddings, max_p, 10, "max_p")   # R = 0.3 SD = 0.3

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
        print(name, round(r, 2), round(p, 4))
        
        
# Socialness https://osf.io/2dqnj/

socialness = pd.read_csv("/home/dev/Documents/Datasets/socialness.csv")
social_w = [row.Word for index, row in socialness.iterrows()]
social_y = np.array([row.Mean for index, row in socialness.iterrows()])

embeddings = get_sentence_embeddings(social_w, "social")
weights_social = probe(embeddings, social_y, 10, "social") # R = 0.38 SD = 0.38

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_social, data))

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
frequency = np.array([get_f(w) for sent in sents for w in sent.split()])
length = np.array([get_l(w) for sent in sents for w in sent.split()])

weights_surp = probe(embeddings_med, s_med, 10, "surprisal_10k") # R = 0.35 SD = 0.35
weights_freq = probe(embeddings_med, frequency, 10, "frequency_10k") # R = 0.16 SD = 0.27
weights_len  = probe(embeddings_med, length, 10, "length_10k") # R = 0.35 SD = 0.38


weights_surp = load_weights("surprisal_10k", "probing_LMs/")
weights_freq = load_weights("frequency_10k", "probing_LMs/")
weights_len = load_weights("length_10k", "probing_LMs/")

for w in [weights_surp, weights_freq, weights_len]:
    print("\n\n")
    for data, name in zip(datasets, names):
        r, p = pearsonr(w, data)
        print(name, round(r, 2), round(p, 4))

#############################################################################
# PreTENS-Evaluating Neural Networks on Presuppositional Semantic Knowledge #
#############################################################################

pretens0 = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open("probing_LMs/PreTens/En-Subtask1-fold_0.tsv")])
pretens0.columns = pretens0.iloc[0] 
pretens0 = pretens0.iloc[1:]
pretens1 = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open("probing_LMs/PreTens/En-Subtask1-fold_1.tsv")])
pretens1.columns = pretens1.iloc[0] 
pretens1 = pretens1.iloc[1:]
pretens2 = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open("probing_LMs/PreTens/En-Subtask1-fold_2.tsv")])
pretens2.columns = pretens2.iloc[0] 
pretens2 = pretens2.iloc[1:]
pretens = pd.concat([pretens0[["ID", "Sentence", "Labels"]], pretens1[["ID", "Sentence", "Labels"]], pretens2[["ID", "Sentence", "Labels"]]], ignore_index=True) # 5837, using all

pretens_sent = list(pretens["Sentence"])
pretens_y = np.array(pretens["Labels"])
pretens_idx = list(pretens["ID"])

embeddings = get_sentence_embeddings(pretens_sent, "pretens") # not really 10k, but naming consistency
weights_pretens = classify(embeddings, pretens_y, 10, "pretens")
weights_pretens = load_weights("pretens", "probing_LMs/")

# Accuracy = 0.9 SD = 0.02

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_pretens[0], data))
    

    
############
# Plotting #
############

# Surface
weights_sentlen = load_weights("sentlen_10k", "probing_LMs/")
weights_surp = load_weights("surprisal_10k", "probing_LMs/")
weights_freq = load_weights("frequency_10k", "probing_LMs/")
weights_len = load_weights("length_10k", "probing_LMs/")

# Syntax
weights_bshift = load_weights("bshift_10k", "probing_LMs/")
weights_shuf = load_weights("shuf_10k", "probing_LMs/")
weights_tdepth = load_weights("tdepth_10k", "probing_LMs/")

# Semantics, words
weights_aud = load_weights("auditory", "probing_LMs/")
weights_viz = load_weights("visual", "probing_LMs/")
weights_gus = load_weights("gustatory", "probing_LMs/")
weights_olf = load_weights("olfactory", "probing_LMs/")
weights_hap = load_weights("haptic", "probing_LMs/")
weights_max = load_weights("max_p", "probing_LMs/")
weights_val = load_weights("valence", "probing_LMs/")
weights_ara = load_weights("arousal", "probing_LMs/")
weights_dom = load_weights("dominance", "probing_LMs/")
weights_social = load_weights("social", "probing_LMs/")

# Semantics, sentence
weights_somo = load_weights("somo_10k", "probing_LMs/")
weights_objn = load_weights("objn_10k", "probing_LMs/")
weights_subjn = load_weights("subjn_10k", "probing_LMs/")
weights_tense = load_weights("tense_10k", "probing_LMs/")
weights_pretens = load_weights("pretens", "probing_LMs/")

# Discourse
weights_coord_inv = load_weights("coord_inv_10k", "probing_LMs/")
#weights_ordering = load_weights("ordering_10k", "probing_LMs/")
weights_nsp = load_weights("nsp_10k", "probing_LMs/")
weights_pretense = load_weights("pretens", "probing_LMs/")




weights_tasks = [weights_sentlen, weights_surp, weights_freq, weights_len, weights_shuf, weights_bshift, weights_tdepth, weights_aud, weights_viz, weights_gus, weights_olf, weights_hap, weights_max, weights_val, weights_ara, weights_dom, weights_social, weights_somo, weights_objn, #weights_subjn, weights_tense, weights_pretense, weights_coord_inv,
                 weights_nsp]
weights_names =  ["SentLen", "Surprisal", "Frequency", "Length", "Shuffle", "BShift", "TDepth", "Auditory", "Visual", "Gustatory", "Olfactory", "Haptic", "Max", "Valence", "Arousal", "Dominance", "Social", "SOMO", "ObjN", #"SubjN", "Tense", "PreTens", "CoordInv", 
                  "NSP"]


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


# Create a mask for non-significant correlations
mask_non_sig = p_values > 0.01
mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
corr[mask] = np.nan

df = pd.DataFrame(corr, columns=weights_names, index=names)
#np.fill_diagonal(df.values, np.nan)
#df = df.iloc[1:, :-1]
plt.figure(figsize=(15, 10), dpi=150)
sns.heatmap(df, annot=True, fmt=".2f", cmap='jet', vmin=-.55, vmax=.55, square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 10})
plt.title('Correlation between probe weights', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
# plt.annotate('Behav.', xy=(-0.29, 0.692), xytext=(-0.44, 0.692),
#             fontsize=19, ha='center', va='center', xycoords='axes fraction', 
#             bbox=dict(boxstyle='square', fc='0.8'),
#             arrowprops=dict(arrowstyle='-[, widthB=7, lengthB=.5', lw=2.0))
# plt.annotate('Neural', xy=(-.29, 0.199), xytext=(-.44, 0.199),
#             fontsize=19, ha='center', va='center', xycoords='axes fraction', 
#             bbox=dict(boxstyle='square', fc='0.8'),
#             arrowprops=dict(arrowstyle='-[, widthB=4.2, lengthB=.5', lw=2.0))
plt.show()



# Might add: NPI (Investigating Bert's knowledge of language)
