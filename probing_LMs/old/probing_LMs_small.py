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

hp = load_weights("hp_all_small_probing", "fMRI/")
hp_split = np.mean(np.array_split(hp, 4), axis=0) # averaging for spillover
#hp_split = np.array_split(hp, 4)[-1]
pereira = load_weights("pereira_all_small_probing", "pereira/") ###
michaelov = load_weights("michaelov_small_probing")
ucl_fpass = load_weights("UCL_fpass_small_probing")
ucl_n400 = load_weights("UCL_n400_small_probing")
ucl_p600 = load_weights("UCL_p600_small_probing")
ucl_spr = load_weights("UCL_spr_small") ###
ucl_spr = np.mean(np.array_split(ucl_spr, 3), axis=0) # averaging for spillover
meco = load_weights("meco_small_probing")
provo = load_weights("provo_small_probing")
zuco = load_weights("zuco_small", "ZuCo/") ####
maze = load_weights("maze_small") ####
natstor = load_weights("natstor_small") ####
natstor = np.mean(np.array_split(natstor, 3), axis=0) # averaging for spillover
brown = load_weights("brown_small")
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
plt.title('Correlation between probe weights', fontsize=20)
plt.xticks(fontsize=12, rotation=45, ha="right")
plt.yticks(fontsize=12)
plt.show()

#####################
# linguistic probes #
#####################

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

def get_sentence_embeddings(sentences):
    embeddings = []
    for sentence in tqdm(sentences):
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
        sentence_embedding = outputs[0].mean(dim=1)
        embeddings.append(sentence_embedding.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    return embeddings

def classify(X, y, n, saveto=None, save = True):
    print("Ridge classifier")
    warnings.filterwarnings('ignore')
    kf = StratifiedKFold(n_splits=n)
    out_reg = []
    out_coefs = []
    out_predictions = []
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

# def probe(X, y, n, saveto=None, save = True):
#     print("Ridge linear regression")
#     kf = KFold(n_splits=n, shuffle=True, random_state=0)
#     out_reg = []
#     out_coefs = []
#     out_predictions = []
#     for train_index, test_index in tqdm(kf.split(X), total=n):
#         X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#         ridge = Ridge()
#         parameters = {'alpha': (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)}
#         reg = GridSearchCV(ridge, parameters, cv=5)
#         reg.fit(X_train, y_train)
#         y_pred = reg.predict(X_test)
#         r = pearsonr(y_test, y_pred)
#         out_predictions.append([y_test, y_pred])
#         coefs = reg.best_estimator_.coef_  #; print(coefs)
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

def probe(X, y, n, saveto=None, save = True):
    print("Ridge linear regression")
    kf = KFold(n_splits=n, shuffle=True, random_state=0)
    out_reg = []
    out_coefs = []
    out_predictions = []
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r = pearsonr(y_test, y_pred)
        out_predictions.append([y_test, y_pred])
        coefs = reg.coef_  #; print(coefs)
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

n = 5000 # 1200 total
classes = sentlen['class'].unique()
samples = [sentlen[sentlen['class'] == cls].sample(n) for cls in classes]
sentlen_sample = pd.concat(samples)
sentlen_sent = list(sentlen_sample["sent"])
sentlen_y = np.array([len(sent.split()) for sent in sentlen_sent])

embeddings = get_sentence_embeddings(sentlen_sent)
embeddings = zscore(embeddings); time.sleep(180)
weights_sentlen = probe(embeddings, sentlen_y, 10, "sentlen_SMALL")
weights_sentlen = load_weights("sentlen_SMALL", "probing_LMs/")
# R = 0.46 SD = 0.46

for data, name in zip(datasets, names):
    print(name, round(pearsonr(weights_sentlen, data)[0], 2), round(pearsonr(weights_sentlen, data)[1], 4))

##########
# BShift #
##########

bshift = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"bigram_shift.txt").readlines()])
bshift.columns = ["train", "class", "sent"]

n = 5000 # 2400 total
classes = bshift['class'].unique()
samples = [bshift[bshift['class'] == cls].sample(n) for cls in classes]
bshift_sample = pd.concat(samples)
bshift_sent = list(bshift_sample["sent"])
bshift_y = np.array([{"O":0, "I":1}[s] for s in bshift_sample["class"]])

embeddings = get_sentence_embeddings(bshift_sent)
embeddings = zscore(embeddings); time.sleep(180)
weights_bshift = classify(embeddings, bshift_y, 10, "bshift_SMALL")
weights_bshift = load_weights("bshift_SMALL", "probing_LMs/")

# Accuracy = 0.78 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_bshift[0], data))

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

embeddings = get_sentence_embeddings(tdepth_sent)
embeddings = zscore(embeddings); time.sleep(180)
weights_tdepth = probe(embeddings, tdepth_y, 10, "tdepth_SMALL")
weights_tdepth = load_weights("tdepth_SMALL", "probing_LMs/")
# R = 0.27 SD = 0.33

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_tdepth, data))

########
# SOMO # RERUN
########

somo = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"odd_man_out.txt").readlines()])
somo.columns = ["train", "class", "sent"]

n = 5000 # 50,000 total
classes = somo['class'].unique()
samples = [somo[somo['class'] == cls].sample(n) for cls in classes]
somo_sample = pd.concat(samples)
somo_sent = list(somo_sample["sent"])
somo_y = np.array([{"C":0, "O":1}[s] for s in somo_sample["class"]])

embeddings = get_sentence_embeddings(somo_sent)
embeddings = zscore(embeddings); time.sleep(180)
weights_somo = classify(embeddings, somo_y, 10, "somo_SMALL")
weights_somo = load_weights("somo_SMALL", "probing_LMs/")
# Accuracy = 0.57 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_somo[0], data))

######### 
# Tense #
#########

tense = pd.DataFrame([re.sub("\n", "", line).split("\t") for line in open(SE+"past_present.txt").readlines()])
tense.columns = ["train", "class", "sent"]

n = 5000 # 2400 total
classes = tense['class'].unique()
samples = [tense[tense['class'] == cls].sample(n) for cls in classes]
tense_sample = pd.concat(samples)
tense_sent = list(tense_sample["sent"])
tense_y = np.array([{"PRES":0, "PAST":1}[s] for s in tense_sample["class"]])

embeddings = get_sentence_embeddings(tense_sent)
embeddings = zscore(embeddings)
weights_tense = classify(embeddings, tense_y, 10, "tense_SMALL")
weights_tense = load_weights("tense_SMALL", "probing_LMs/")

# Accuracy = 0.88 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_tense[0], data))

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

embeddings = get_sentence_embeddings(coord_inv_sent)
embeddings = zscore(embeddings)
weights_coord_inv = classify(embeddings, coord_inv_y, 10, "coord_inv_SMALL")

# Accuracy = 0.62 SD = 0.01

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_coord_inv[0], data))
    
###############################################################################

#####################
# Sentence ordering # RERUN
#####################

with open("probing_LMs/discourse/sentence_ordering.json") as f:
    d = json.load(f)

len(d) # 8000

sent1 = []; sent2 = []
for obj in d:
    sent = obj["sentences"]
    order = obj["order"]
    paired = list(zip(sent, order))
    sorted_pair = sorted(paired, key=lambda x: x[1])
    sorted_a = [element[0] for element in sorted_pair]
    sent1.append(" ".join(sorted_a))
    sent2.append(" ".join(sent))
    
embeddings1 = get_sentence_embeddings(sent1)
embeddings1 = zscore(embeddings1)


# RUN FROM HERE!!!
embeddings2 = get_sentence_embeddings(sent2)
embeddings2 = zscore(embeddings2)

sent_ord_y = np.array([1]*len(embeddings1)+[0]*len(embeddings2))

embeddings = np.concatenate([embeddings1, embeddings2])
print(embeddings.shape)


for s1, s2 in zip(sent1, sent2):
    if len(s1)==len(s2):
        pass
    else:
        print(s1, s2)

weights_ordering = classify(embeddings, sent_ord_y, 10, "ordering_SMALL")
weights_ordering = load_weights("ordering_SMALL", "probing_LMs/")
# Accuracy = 0.37 SD = 0.02 terrible accuracy

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_ordering[0], data))

#######
# NSP # RERUN
#######

with open("probing_LMs/discourse/nsp.json") as f:
    d = json.load(f)
    
nsp_1 = []; nsp_2 = []
for obj in d:
    ctx = [re.sub("\n", "", line) for line in obj["context"]]
    answ =  sorted(obj["answer"], key=lambda x: x[0], reverse=True)
    right = " ".join(ctx + [re.sub("\n", "",answ[0][1])])
    wrong = " ".join(ctx + [re.sub("\n", "",answ[1][1])])
    nsp_1.append(right); nsp_2.append(wrong)

embeddings1 = get_sentence_embeddings(nsp_1)
embeddings1 = zscore(embeddings1)

embeddings2 = get_sentence_embeddings(nsp_2)
embeddings2 = zscore(embeddings2)

sent_ord_y = np.array([1]*len(embeddings1)+[0]*len(embeddings2))

embeddings = np.concatenate([embeddings1, embeddings2])

weights_nsp = classify(embeddings, sent_ord_y, 10, "nsp_SMALL")
weights_nsp = load_weights("nsp_SMALL", "probing_LMs/")

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_nsp[0], data))

# FOR NOW, I'VE RUN IT SINCE HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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

a_embeddings = get_sentence_embeddings(w_a)
p_embeddings = get_sentence_embeddings(w_p)

weights_aud = probe(p_embeddings, aud, 10, "auditory_SMALL")  # R = 0.32 SD = 0.32
weights_viz = probe(p_embeddings, vis, 10, "visual_SMALL")    # R = 0.31 SD = 0.31
weights_gus = probe(p_embeddings, gus, 10, "gustatory_SMALL") # R = 0.34 SD = 0.34
weights_olf = probe(p_embeddings, olf, 10, "olfactory_SMALL") # R = 0.31 SD = 0.31
weights_hap = probe(p_embeddings, hap, 10, "haptic_SMALL")    # R = 0.34 SD = 0.34
weights_max = probe(p_embeddings, max_p, 10, "max_p_SMALL")   # R = 0.3 SD = 0.3

weights_val = probe(a_embeddings, np.array(v), 10, "valence_SMALL") # R = 0.36 SD = 0.36
weights_ara = probe(a_embeddings, np.array(a), 10, "arousal_SMALL") # R = 0.29 SD = 0.29
weights_dom = probe(a_embeddings, np.array(d), 10, "dominance_SMALL") # R = 0.32 SD = 0.32

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

sentences = extract_sentences(2000)
sentences_clean = [tokenize(sent) for sent in tqdm(sentences)]
filtered_sentences = [sentence[0] for sentence in sentences_clean if is_known(sentence[0])]
sents = sorted(filtered_sentences, key=len, reverse=True)[:1000]

model1 = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
toker1 = AutoTokenizer.from_pretrained("gpt2-medium")
s_med = np.array([s for sent in tqdm(sents) for s in get_surprisal_tokens(sent.split(), sep = "Ġ", toker=toker1, model=model1, cased=True)])
embeddings_med = np.array([list(s) for sent in tqdm(sents) for s in get_embeddings_tokens(sent.split(), sep = "Ġ")])
embeddings_med = zscore(embeddings_med)
frequency = np.array([get_f(w) for sent in sents for w in sent.split()])
length = np.array([get_l(w) for sent in sents for w in sent.split()])

weights_surp = probe(embeddings_med, s_med, 10, "surprisal") # R = 0.37 SD = 0.37
weights_freq = probe(embeddings_med, frequency, 10, "frequency") # R = 0.18 SD = 0.24
weights_len  = probe(embeddings_med, length, 10, "length")


for data, name in zip(datasets, names):
    r, p = pearsonr(weights_surp, data)
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

embeddings = get_sentence_embeddings(pretens_sent)
embeddings = zscore(embeddings)
weights_pretens = classify(embeddings, pretens_y, 10, "pretens_SMALL") # I have overwritten by mstake, should recompute

# Accuracy = 0.9 SD = 0.02

for data, name in zip(datasets, names):
    print(name, pearsonr(weights_pretens[0], data))
    

    
############
# Plotting #
############

# Surface
weights_sentlen = load_weights("sentlen_SMALL", "probing_LMs/")

# Syntax
weights_shuf = load_weights("shuf_SMALL", "probing_LMs/")
weights_bshift = load_weights("bshift_SMALL", "probing_LMs/")
weights_tdepth = load_weights("tdepth_SMALL", "probing_LMs/")

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

# Semantics, sentence
weights_somo = load_weights("somo_SMALL", "probing_LMs/")
weights_tense = load_weights("tense_SMALL", "probing_LMs/")
weights_pretens = load_weights("pretens_SMALL", "probing_LMs/")

# Discourse
weights_coord_inv = load_weights("coord_inv_SMALL", "probing_LMs/")


weights_tasks = [weights_sentlen, weights_shuf, weights_bshift, weights_tdepth, #weights_aud, weights_viz, weights_gus, weights_olf, weights_hap, weights_max, weights_val, weights_ara, weights_dom, 
                 weights_somo, weights_tense, weights_coord_inv, weights_pretens]
weights_names =  ["SentLen", "Shuffle", "BShift", "TDepth", #"Auditory", "Visual", "Gustatory", "Olfactory", "Haptic", "Max", "Valence", "Arousal", "Dominance", 
                  "SOMO", "Tense", "CoordInv", "PreTENS"]


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


# done with large sample size:
    # SOMO
    # Ordering (not sure)
    # NSP

