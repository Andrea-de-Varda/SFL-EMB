# utils file
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import pickle
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import scipy.io as io
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.patches as mpatches
import re
import pandas as pd
from pygam import LinearGAM, s, l, GAM
from pygam.terms import TermList as TL
import itertools
import copy
from math import sqrt
from joblib import dump, load
from scipy.stats import zscore
import fasttext.util
import copy
from scipy import spatial
from pathlib import Path as _Path

# --- reproducible paths: utils.py lives at the repo root, so its parent IS the root ---
PROJECT_ROOT = _Path(__file__).resolve().parent

# model testing function that saves predictions (to look at later) and coefficients

def loadmat1(filename):
    '''
    Credit to Mergen; https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    Credit to Mergen; https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    Credit to Mergen; https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def swap_elements(a, b, indices):
    a, b = copy.deepcopy(a), copy.deepcopy(b)  # avoid changing the original arrays
    for i in indices:
        a[i], b[i] = b[i], a[i]
    return [a, b]

def generate_swaps(a1, a2):
    # Define inner function for swapping elements
    # Generate all combinations of index sets for swapping
    swap_indices = []
    for r in range(1, len(a1) + 1):  # start from 1 to exclude no swap scenario
        swap_indices.extend(itertools.combinations(range(len(a1)), r))

    # Apply each set of swaps
    result = []
    for indices in swap_indices:
        result.append(swap_elements(a1, a2, indices))

    return result

def permutation_statistic(a1, a2):
    mean_diff = np.mean(a1) - np.mean(a2)
    swaps = generate_swaps(a1, a2)
    diffs = [(np.mean(xa1) - np.mean(xa2)) for xa1, xa2 in swaps]
    p = sum([(diff >= mean_diff) for diff in diffs])/2**len(a1) # adding the held-out one, mean diff
    return p

def significance_testing(name1, name2, thedir=""):
    with open(thedir+'results/'+name1+'_predictions', 'rb') as handle:
        a1 = pickle.load(handle)
    with open(thedir+'results/'+name2+'_predictions', 'rb') as handle:
        a2 = pickle.load(handle)
    p = permutation_statistic(a1, a2)
    return p


def decompose_sfl(X, y, saveto, idx_s = 0, idx_f = 1, idx_l = 2, save=True):
    if type(idx_s) == int:
        s = X[:, idx_s].reshape(-1, 1)
        f = X[:, idx_f].reshape(-1, 1)
        l = X[:, idx_l].reshape(-1, 1)
        
        combs = [(s), (f), (l), (s, f), (f, l), (s, l), (s, f, l)]
        names = ["S", "F", "L", "SF",   "FL",   "SL",   "SFL"]
        
        out = {}
        for c, n in zip(combs, names):
            the_x = np.hstack(c)
            if len(the_x.shape) == 1:
                m, sd = test_model_Ridge(the_x.reshape(-1, 1), y, 10, save=False)
                out[n] = [m, sd]
            else:
                m, sd = test_model_Ridge(the_x, y, 10, save=False)
                out[n] = [m, sd]
        df = pd.DataFrame(out)
        df.index = ["m", "sd"]
        print("\n", df)
        if save:
            df.to_csv("results/"+saveto+"_sfl_decomposed.csv")
    else:
        s = X[:, idx_s]
        f = X[:, idx_f]
        l = X[:, idx_l]
        
        combs = [(s), (f), (l), (s, f), (f, l), (s, l), (s, f, l)]
        names = ["S", "F", "L", "SF",   "FL",   "SL",   "SFL"]
        
        out = {}
        for c, n in zip(combs, names):
            if n in ["S", "F", "L"]:
                m, sd = test_model_Ridge(c, y, 10, save=False)
                out[n] = [m, sd]
            else:
                the_x = np.hstack(c)
                m, sd = test_model_Ridge(the_x, y, 10, save=False)
                out[n] = [m, sd]
        df = pd.DataFrame(out)
        df.index = ["m", "sd"]
        print("\n", df)
        if save:
            df.to_csv("results/"+saveto+"_sfl_decomposed.csv")
        

def test_model_Ridge(X, y, n, saveto=None, save = True):
    kf = KFold(n_splits=n)
    out_reg = []
    out_coefs = []
    out_predictions = []
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_predictions.append([y_test, y_pred])
        coefs = reg.coef_#; print(coefs)
        out_coefs.append(coefs)
        out_reg.append(r)
    coefs_avg = np.mean(out_coefs, axis=0)
    coefs_std = np.std(out_coefs, axis=0)
    if save:
        with open("results/"+saveto+"_coefs_avg", 'wb') as handle:
            pickle.dump(coefs_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_coefs", 'wb') as handle:
            pickle.dump(out_coefs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_coefs_std", 'wb') as handle:
            pickle.dump(coefs_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_predictions", "wb") as handle:
            pickle.dump(out_reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return np.mean(out_reg), np.std(out_reg)

def probe(X, y, num_alphas=10, saveto=None, save=True, normalize=True):
    print("Fitting final Ridge regression on all data")
    if normalize:
        X = zscore(X)
        y = zscore(y)
    alphas = np.logspace(-5, 4, num=num_alphas)
    reg = RidgeCV(alphas=alphas, store_cv_values=False)
    reg.fit(X, y)
    coefs_final = reg.coef_
    best_alpha = reg.alpha_
    if save:
        with open("results/" + saveto + "_coefs_final", 'wb') as handle:
            pickle.dump(coefs_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/" + saveto + "_best_alpha", 'wb') as handle:
            pickle.dump(best_alpha, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return coefs_final

def test_model_GAM(X, y, n, condition="all"):
    kf = KFold(n_splits=n)
    if condition == "all":
        terms = TL(*[s(0), s(1), s(2)] + [l(n) for n in range(3, X.shape[1])]) # SFL as smooth terms, EMB linear
    elif condition == "SFL":
        terms = TL(*[s(0), s(1), s(2)]) # all smooth
    elif condition == "EMB":
        terms = TL(*[l(n) for n in range(X.shape[1])])
    out_reg = []
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        reg = LinearGAM(terms=terms).gridsearch(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_reg.append(r)
    print(np.mean(out_reg))
    return np.mean(out_reg), np.std(out_reg)

def test_model_Ridge_nofolds(X, y, normalize=False, return_predictions = False, return_model = False):
    if normalize:
        X = normalize(X, norm='l2', axis=0, copy=True, return_norm=False)
    reg = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000), cv = 5)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r, _ = pearsonr(y, y_pred)
    coefs = reg.coef_#; print(coefs)
    print("R =", round(r, 4))
    to_return = [coefs]
    if return_predictions:
        to_return.append(y_pred)
    if return_model:
        to_return.append(reg)
    return to_return

def get_surprisal(prompt, toker, model):
    inputs = toker(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return [-item[0] for item in logprobs.tolist()[0]]

def tok_maker(a, sep, toker, cased = False):
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

def get_surprisal_tokens(tokens, sep, toker, model, cased=False):
    s = get_surprisal(" ".join(tokens), toker, model)
    toks = tok_maker(tokens, sep, toker, cased)
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

# this old function used ridge regression for estimating affective properties; turns out the best way is to use neighbours
# def get_sem_old(wordlist):
#     v, a, d, aud, gus, hap, nte, olf, vis, foo, han, hea, mou, tor = [load("/home/dev/Documents/PhD/MIT/project/SEM/"+dic+".joblib") for dic in ["v", "a", "d", "aud", "gus", "hap", "nte", "olf", "vis", "foo", "han", "hea", "mou", "tor"]]; print("Loaded regression weights")
#     ft = fasttext.load_model('/home/dev/Documents/PhD/MIT/project/cc.en.300.bin'); print("Loaded FastText model")
#     d_perceptual = load("/home/dev/Documents/PhD/MIT/project/SEM/d_perceptual.joblib")
#     d_affective  = load("/home/dev/Documents/PhD/MIT/project/SEM/d_affective.joblib"); print("Loaded dicts")
#     missing_a = 0
#     missing_p = 0
#     SEM  = []
#     for w in tqdm(wordlist):
#         if type(w) == str:
#             condition = 1
#             try:
#                 affect = d_affective[w]
#             except KeyError:
#                 wordvec = ft.get_word_vector(w).reshape(1, -1)
#                 aff_v = v.predict(wordvec)[0]
#                 aff_a = a.predict(wordvec)[0]
#                 aff_d = d.predict(wordvec)[0]
#                 affect = [aff_v, aff_a, aff_d]
#                 missing_a += 1
#             try:
#                 percept = d_perceptual[w]
#             except KeyError:
#                 wordvec = ft.get_word_vector(w).reshape(1, -1)
#                 p_aud = aud.predict(wordvec)[0]
#                 p_gus = gus.predict(wordvec)[0]
#                 p_hap = hap.predict(wordvec)[0]
#                 p_nte = nte.predict(wordvec)[0]
#                 p_olf = olf.predict(wordvec)[0]
#                 p_vis = vis.predict(wordvec)[0]
#                 p_foo = foo.predict(wordvec)[0]
#                 p_han = han.predict(wordvec)[0]
#                 p_hea = hea.predict(wordvec)[0]
#                 p_mou = mou.predict(wordvec)[0]
#                 p_tor = tor.predict(wordvec)[0]
#                 #p_maxm = max_m.predict(wordvec)[0]
#                 #p_maxp = max_p.predict(wordvec)[0]
#                 #p_excp = exc_p.predict(wordvec)[0]
#                 #p_exca = exc_a.predict(wordvec)[0]
#                 percept = [p_aud, p_gus, p_hap, p_nte, p_olf, p_vis, p_foo, p_han, p_hea, p_mou, p_tor]
#                 missing_p += 1
#             SEM.append([*affect, *percept])
#         elif type(w) == list:
#             condition = 2
#             temp_SEM = []
#             for w1 in w:
#                 w1 = w1.lower()
#                 try:
#                     affect = d_affective[w1]
#                 except KeyError:
#                     wordvec = ft.get_word_vector(w1).reshape(1, -1)
#                     aff_v = v.predict(wordvec)[0]
#                     aff_a = a.predict(wordvec)[0]
#                     aff_d = d.predict(wordvec)[0]
#                     affect = [aff_v, aff_a, aff_d]
#                     missing_a += 1
#                 try:
#                     percept = d_perceptual[w1]
#                 except KeyError:
#                     wordvec = ft.get_word_vector(w1).reshape(1, -1)
#                     p_aud = aud.predict(wordvec)[0]
#                     p_gus = gus.predict(wordvec)[0]
#                     p_hap = hap.predict(wordvec)[0]
#                     p_nte = nte.predict(wordvec)[0]
#                     p_olf = olf.predict(wordvec)[0]
#                     p_vis = vis.predict(wordvec)[0]
#                     p_foo = foo.predict(wordvec)[0]
#                     p_han = han.predict(wordvec)[0]
#                     p_hea = hea.predict(wordvec)[0]
#                     p_mou = mou.predict(wordvec)[0]
#                     p_tor = tor.predict(wordvec)[0]
#                     #p_maxm = max_m.predict(wordvec)[0]
#                     #p_maxp = max_p.predict(wordvec)[0]
#                     #p_excp = exc_p.predict(wordvec)[0]
#                     #p_exca = exc_a.predict(wordvec)[0]
#                     percept = [p_aud, p_gus, p_hap, p_nte, p_olf, p_vis, p_foo, p_han, p_hea, p_mou, p_tor]
#                     missing_p += 1
#                 temp_SEM.append([*affect, *percept])
#             temp_SEM = np.array(temp_SEM).mean(axis=0)
#             SEM.append(list(temp_SEM))
#     if condition == 1:
#         print("\nMissing affective =", round(missing_a/len(wordlist)*100, 2))
#         print("Missing perceptual =", round(missing_p/len(wordlist)*100, 2))
#     else:
#         print("\nMissing affective =", round(missing_a/len([w for sent in wordlist for w in sent])*100, 2))
#         print("Missing perceptual =", round(missing_p/len([w for sent in wordlist for w in sent])*100, 2))
#     SEM = zscore(np.array(SEM), axis=0)
#     return SEM

def get_nearest_neighbor(word, k, candidates, ft):
    word_vector = ft.get_word_vector(word)
    candidate_vectors = [ft.get_word_vector(c) for c in candidates]
    distances = spatial.distance.cdist([word_vector], candidate_vectors, "cosine")[0]
    top_k_indices = distances.argsort()[:k]
    return [(candidates[i], 1 - distances[i]) for i in top_k_indices]

def get_sem(wordlist):
    ft = fasttext.load_model(str(PROJECT_ROOT / "cc.en.300.bin")); print("Loaded FastText model")
    d_perceptual = load(PROJECT_ROOT / "SEM/d_perceptual.joblib")
    d_affective  = load(PROJECT_ROOT / "SEM/d_affective.joblib"); print("Loaded dicts")
    missing_a = 0
    missing_p = 0
    percept_words =  list(filter(lambda item: isinstance(item, str), list(d_perceptual.keys())))
    affect_words  = list(filter(lambda item: isinstance(item, str), list(d_affective.keys())))
    SEM  = []
    for w in tqdm(wordlist):
        if type(w) == str:
            w = re.sub('[\.\,\:\-\?\!\)\(\"]', "", w)
            condition = 1
            try:
                affect = d_affective[w]
            except KeyError:
                w_neigh = get_nearest_neighbor(w, 1, affect_words, ft)[0][0]
                affect = d_affective[w_neigh]
                missing_a +=1
            try:
                percept = d_perceptual[w]
            except KeyError:
                w_neigh = get_nearest_neighbor(w, 1, percept_words, ft)[0][0]
                percept = d_perceptual[w_neigh]
                missing_p += 1
            SEM.append([*affect, *percept])
        elif type(w) == list:
            condition = 2
            temp_SEM = []
            for w1 in w:
                w1 = re.sub('[\.\,\:\-\?\!\)\(\"]', "", w1)
                try:
                    affect = d_affective[w1]
                except KeyError:
                    w_neigh = get_nearest_neighbor(w1, 1, affect_words, ft)[0][0]
                    affect = d_affective[w_neigh]    
                    missing_a += 1
                try:
                    percept = d_perceptual[w1]
                except KeyError:
                    w_neigh = get_nearest_neighbor(w1, 1, percept_words, ft)[0][0]
                    percept = d_perceptual[w_neigh]  
                    missing_p += 1
                temp_SEM.append([*affect, *percept])
            temp_SEM = np.array(temp_SEM).mean(axis=0)
            SEM.append(list(temp_SEM))
    if condition == 1:
        print("\nMissing affective =", round(missing_a/len(wordlist)*100, 2))
        print("Missing perceptual =", round(missing_p/len(wordlist)*100, 2))
    else:
        print("\nMissing affective =", round(missing_a/len([w for sent in wordlist for w in sent])*100, 2))
        print("Missing perceptual =", round(missing_p/len([w for sent in wordlist for w in sent])*100, 2))
    SEM = np.array(SEM)
    print(SEM.var(axis=0))
    SEM = zscore(SEM, axis=0)
    return SEM

def plot_weights_sem(filename, title, subtitle, ymin=-.33, ymax=.23):
    plt.rcParams.update({'font.size': 13})
    color1 = '#9b90c2'  # Pastel purple
    color2 = '#87d3f8'  # Pastel blue
    color3 = '#f8b195'  # Pastel peach
    with open('results/' + filename + '_coefs_avg', 'rb') as handle:
        the_m = pickle.load(handle)
    with open('results/' + filename + '_coefs_std', 'rb') as handle:
        the_sd = pickle.load(handle)
    fig1, ax1 = plt.subplots(dpi=300, figsize=(6, 3))
    x = [0,1,2,4,5,6,7,8,9,11,12,13,14,15]
    idx_reorder = [0,1,2,8,3,5,7,4,6,11,12,10,9,13] # want to show first dominant senses e.g. vision audition
    y = [the_m[idx] for idx in idx_reorder]
    the_sd = [the_sd[idx] for idx in idx_reorder]
    labels = ["valence", "arousal", "dominance", "auditory", "gustatory", "haptic", "interoceptive", "olfactory", "visual", "foot",
      "hand", "head", "mouth", "torso"]
    labels = [labels[idx] for idx in idx_reorder]
    # First X-axis
    ax1.bar(x, y, color=[color1] * 3 + [color2] * 6 + [color3] * 5, width=0.9, zorder=4)
    ax1.errorbar(x, y, yerr=the_sd, fmt='none', elinewidth=1, capsize=4, color="black", zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        labels, rotation=30, ha='right', fontsize=11)  # Rotate labels by 30 degrees
    ax1.yaxis.set_label_text("Weights")
    ax1.set_ylim(ymin, ymax)
    ax1.axhline(y=0, color='black', linestyle='-', zorder=6, linewidth=1)
    ax1.grid(axis="y", b=True, which='major', color='#666666', linestyle='-', zorder=3)
    ax1.grid(axis="y", b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=2)
    plt.title(title)
    plt.show()

def base_plot(df_results, col1, col2, col3, title="", legend=False, ylim1=0.1, ylim2=0.9, cv_size = 10):
    # fpass
    plt.rcParams.update({
        "text.usetex": False})
    fig = plt.figure(figsize = (5/1.7, 3.5/1.7), dpi=300)
    plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
    plt.minorticks_on()
    plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
    yerr = [df_results.loc["GPT", col1+"_sd"], df_results.loc["GPT", col2+"_sd"], df_results.loc["GPT", col3+"_sd"],
     df_results.loc["GPT-2$_{124M}$", col1+"_sd"], df_results.loc["GPT-2$_{124M}$", col2+"_sd"], df_results.loc["GPT-2$_{124M}$", col3+"_sd"],
     df_results.loc["GPT-2$_{355M}$", col1+"_sd"], df_results.loc["GPT-2$_{355M}$", col2+"_sd"], df_results.loc["GPT-2$_{355M}$", col3+"_sd"],
     df_results.loc["GPT-2$_{774M}$", col1+"_sd"], df_results.loc["GPT-2$_{774M}$", col2+"_sd"], df_results.loc["GPT-2$_{774M}$", col3+"_sd"],
     df_results.loc["GPT-2$_{1.5B}$", col1+"_sd"], df_results.loc["GPT-2$_{1.5B}$", col2+"_sd"], df_results.loc["GPT-2$_{1.5B}$", col3+"_sd"]]
    yerr = [y/sqrt(cv_size) for y in yerr]
    plt.bar([1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], 
            [df_results.loc["GPT", col1], df_results.loc["GPT", col2], df_results.loc["GPT", col3],
             df_results.loc["GPT-2$_{124M}$", col1], df_results.loc["GPT-2$_{124M}$", col2], df_results.loc["GPT-2$_{124M}$", col3],
             df_results.loc["GPT-2$_{355M}$", col1], df_results.loc["GPT-2$_{355M}$", col2], df_results.loc["GPT-2$_{355M}$", col3],
             df_results.loc["GPT-2$_{774M}$", col1], df_results.loc["GPT-2$_{774M}$", col2], df_results.loc["GPT-2$_{774M}$", col3],
             df_results.loc["GPT-2$_{1.5B}$", col1], df_results.loc["GPT-2$_{1.5B}$", col2], df_results.loc["GPT-2$_{1.5B}$", col3]],
            color = ['lightblue', 'lightgreen', "indianred"]*5,width=0.9,
            yerr=yerr,capsize=2,
            tick_label=["", "GPT", "", "", "GPT-2$_{124M}$", "", "", "GPT-2$_{355M}$","", "",  "GPT-2$_{774M}$", "", "", "GPT-2$_{1.5B}$", ""], zorder=5
            )
    plt.ylabel("r")
    plt.xticks(fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
    plt.ylim(ylim1, ylim2)
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
    if legend:
        act  = mpatches.Patch(color='lightblue', label='EMB')
        bsl  = mpatches.Patch(color='lightgreen', label='SFL')
        all_ = mpatches.Patch(color='indianred', label='EMB$^\\frown$SFL')
        plt.legend(handles=[act, bsl, all_], bbox_to_anchor=(1.03, 1.01))
    plt.title(title, fontsize=16)
    plt.show()
    
def plot_sem(df_results, col1, col2, col3, col4, col5, col6, title="", legend=False, ylim1=0., ylim2=1, subtract = False, cv_size = 10):
    if subtract:
        bars1 = [df_results.loc["GPT", col1], df_results.loc["GPT", col2], df_results.loc["GPT", col3],
         df_results.loc["GPT-2$_{124M}$", col1], df_results.loc["GPT-2$_{124M}$", col2], df_results.loc["GPT-2$_{124M}$", col3],
         df_results.loc["GPT-2$_{355M}$", col1], df_results.loc["GPT-2$_{355M}$", col2], df_results.loc["GPT-2$_{355M}$", col3],
         df_results.loc["GPT-2$_{774M}$", col1], df_results.loc["GPT-2$_{774M}$", col2], df_results.loc["GPT-2$_{774M}$", col3],
         df_results.loc["GPT-2$_{1.5B}$", col1], df_results.loc["GPT-2$_{1.5B}$", col2], df_results.loc["GPT-2$_{1.5B}$", col3]]
        bars2 = np.array([df_results.loc["GPT", col4], df_results.loc["GPT", col5], df_results.loc["GPT", col6],
         df_results.loc["GPT-2$_{124M}$", col4], df_results.loc["GPT-2$_{124M}$", col5], df_results.loc["GPT-2$_{124M}$", col6],
         df_results.loc["GPT-2$_{355M}$", col4], df_results.loc["GPT-2$_{355M}$", col5], df_results.loc["GPT-2$_{355M}$", col6],
         df_results.loc["GPT-2$_{774M}$", col4], df_results.loc["GPT-2$_{774M}$", col5], df_results.loc["GPT-2$_{774M}$", col6],
         df_results.loc["GPT-2$_{1.5B}$", col4], df_results.loc["GPT-2$_{1.5B}$", col5], df_results.loc["GPT-2$_{1.5B}$", col6]])-np.array(bars1)
    
    else:
        bars1 = [df_results.loc["GPT", col1], df_results.loc["GPT", col2], df_results.loc["GPT", col3],
         df_results.loc["GPT-2$_{124M}$", col1], df_results.loc["GPT-2$_{124M}$", col2], df_results.loc["GPT-2$_{124M}$", col3],
         df_results.loc["GPT-2$_{355M}$", col1], df_results.loc["GPT-2$_{355M}$", col2], df_results.loc["GPT-2$_{355M}$", col3],
         df_results.loc["GPT-2$_{774M}$", col1], df_results.loc["GPT-2$_{774M}$", col2], df_results.loc["GPT-2$_{774M}$", col3],
         df_results.loc["GPT-2$_{1.5B}$", col1], df_results.loc["GPT-2$_{1.5B}$", col2], df_results.loc["GPT-2$_{1.5B}$", col3]]
        bars2 = [df_results.loc["GPT", col4], df_results.loc["GPT", col5], df_results.loc["GPT", col6],
         df_results.loc["GPT-2$_{124M}$", col4], df_results.loc["GPT-2$_{124M}$", col5], df_results.loc["GPT-2$_{124M}$", col6],
         df_results.loc["GPT-2$_{355M}$", col4], df_results.loc["GPT-2$_{355M}$", col5], df_results.loc["GPT-2$_{355M}$", col6],
         df_results.loc["GPT-2$_{774M}$", col4], df_results.loc["GPT-2$_{774M}$", col5], df_results.loc["GPT-2$_{774M}$", col6],
         df_results.loc["GPT-2$_{1.5B}$", col4], df_results.loc["GPT-2$_{1.5B}$", col5], df_results.loc["GPT-2$_{1.5B}$", col6]]
    
    colormap = {True:"#FFA54C", False:"darkred"}
    plt.rcParams.update({
        "text.usetex": False})
    fig = plt.figure(figsize = (5/1.7, 3.5/1.7), dpi=300)
    plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
    plt.minorticks_on()
    plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
    colorlist = pd.Series(pd.Series(bars2) > 0).map(colormap)
    yerr=[df_results.loc["GPT", col3+"_sd"], df_results.loc["GPT", col4+"_sd"], df_results.loc["GPT", col5+"_sd"],
     df_results.loc["GPT-2$_{124M}$", col3+"_sd"], df_results.loc["GPT-2$_{124M}$", col4+"_sd"], df_results.loc["GPT-2$_{124M}$", col5+"_sd"],
     df_results.loc["GPT-2$_{355M}$", col3+"_sd"], df_results.loc["GPT-2$_{355M}$", col4+"_sd"], df_results.loc["GPT-2$_{355M}$", col5+"_sd"],
     df_results.loc["GPT-2$_{774M}$", col3+"_sd"], df_results.loc["GPT-2$_{774M}$", col4+"_sd"], df_results.loc["GPT-2$_{774M}$", col5+"_sd"],
     df_results.loc["GPT-2$_{1.5B}$", col3+"_sd"], df_results.loc["GPT-2$_{1.5B}$", col4+"_sd"], df_results.loc["GPT-2$_{1.5B}$", col5+"_sd"]]
    yerr = [y/sqrt(cv_size) for y in yerr]
    plt.bar([1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], 
            bars1,
            color = ['lightblue', 'lightgreen', "indianred"]*5,width=0.9,
            tick_label=["", "GPT", "", "", "GPT-2$_{124M}$", "", "", "GPT-2$_{355M}$","", "",  "GPT-2$_{774M}$", "", "", "GPT-2$_{1.5B}$", ""], zorder=5)
    plt.bar([1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], 
            bars2,
            yerr=yerr, capsize=2,
            color = colorlist,width=0.9,zorder=5, bottom=bars1)
    plt.ylabel("r")
    plt.xticks(fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
    plt.ylim(ylim1, ylim2)
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
    if legend:
        act  = mpatches.Patch(color='lightblue', label='EMB')
        bsl  = mpatches.Patch(color='lightgreen', label='SFL')
        all_ = mpatches.Patch(color='indianred', label='EMB$^\\frown$SFL')
        plt.legend(handles=[act, bsl, all_], bbox_to_anchor=(1.03, 1.01))
    plt.title(title, fontsize=16)
    plt.show()
    
def simple_barplot(y, ymin, ymax, title=""):
    # fpass
    plt.rcParams.update({
        "text.usetex": False})
    fig = plt.figure(figsize = (5/1.2, 3.5/1.2), dpi=300)
    plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
    plt.minorticks_on()
    plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
    plt.bar([1,2,3,4,5], y,
            color = ['lightblue']*5,width=0.9,
            tick_label=["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$",  "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"], zorder=5
            )
    plt.ylabel("r")
    plt.xticks(fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
    plt.ylim(0.1, 0.9)
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
    plt.title(title)
    plt.show()

def plot_weights_nospill(filename, title, subtitle, ymin = -.01, ymax=.015):
    plt.rcParams.update({'font.size': 13})
    color1 = '#9b90c2'  # Pastel purple
    color2 = '#87d3f8'  # Pastel blue
    color3 = '#f8b195'  # Pastel peach
    with open('results/'+filename+'_coefs_avg', 'rb') as handle:
        the_m = pickle.load(handle)
    with open('results/'+filename+'_coefs_std', 'rb') as handle:
        the_sd = pickle.load(handle)

    fig1 = plt.figure(dpi = 300)
    ax1 = SubplotHost(fig1, 111)
    fig1.add_subplot(ax1)
    # Some data
    x = [1,2,3]
    y = the_m
    # First X-axis
    ax1.bar(x, y, color = [color1, color2, color3],width=0.9, zorder=4)
    ax1.errorbar(x, y, yerr=the_sd, fmt='none', elinewidth=1, capsize=4, color="black", zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["S", "F", "L"])
    #ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
    ax1.yaxis.set_label_text("Weights")
    ax1.set_xlim(0, 4)
    ax1.set_ylim(ymin, ymax)
    ax1.axhline(y=0, color='black', linestyle='-', zorder=6, linewidth=1)
    ax1.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=3)
    #plt.minorticks_on()
    ax1.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=2)
    ax2 = ax1.twiny()
    offset = 0, -25 # Position of the second axis
    new_axisline = ax2.get_grid_helper().new_fixed_axis
    ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
    ax2.axis["top"].set_visible(False)
    ax2.set_xticks([0, 1])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([0.5]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter([subtitle]))
    ax2.axis["bottom"].minor_ticks.set_ticksize(0)
    plt.title(title)
    plt.show()
    
def plot_weights_nospill_withEMB(filename, title, subtitle, ymin = -.01, ymax=.015):
    plt.rcParams.update({'font.size': 13})
    color1 = '#9b90c2'  # Pastel purple
    color2 = '#87d3f8'  # Pastel blue
    color3 = '#f8b195'  # Pastel peach
    color4 = '#FAC8CD'
    with open('results/'+filename+'_coefs_avg', 'rb') as handle:
        the_m = pickle.load(handle)
    with open('results/'+filename+'_coefs_std', 'rb') as handle:
        the_sd = pickle.load(handle)
    the_m1 = list(the_m[:3])
    the_sd1 = list(the_sd[:3])
    extra_weight_mass = sum([abs(x) for x in the_m[3:]])/len(the_m[3:])
    the_m1.append(extra_weight_mass)
    
    fig1 = plt.figure(dpi = 300)
    ax1 = SubplotHost(fig1, 111)
    fig1.add_subplot(ax1)
    # Some data
    x = [1,2,3,4]
    y = the_m1
    the_sd1.append(np.mean(the_sd[3:]))
    # First X-axis
    ax1.bar(x, y, color = [color1, color2, color3, color4],width=0.9, zorder=4)
    ax1.errorbar(x, y, yerr=the_sd1, fmt='none', elinewidth=1, capsize=4, color="black", zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["S", "F", "L", "$\overline{EMB}$"])
    #ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
    ax1.yaxis.set_label_text("Weights")
    ax1.set_xlim(0, 5)
    ax1.set_ylim(ymin, ymax)
    ax1.axhline(y=0, color='black', linestyle='-', zorder=6, linewidth=1)
    ax1.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=3)
    #plt.minorticks_on()
    ax1.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=2)
    ax2 = ax1.twiny()
    offset = 0, -25 # Position of the second axis
    new_axisline = ax2.get_grid_helper().new_fixed_axis
    ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
    ax2.axis["top"].set_visible(False)
    ax2.set_xticks([0, 1])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([0.5]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter([subtitle]))
    ax2.axis["bottom"].minor_ticks.set_ticksize(0)
    plt.title(title)
    plt.show()

def plot_weights_spill(filename, title, ymin = -.01, ymax=.015, modelname=""):
    plt.rcParams.update({'font.size': 13})
    color1 = '#9b90c2'  # Pastel purple
    color2 = '#87d3f8'  # Pastel blue
    color3 = '#f8b195'  # Pastel peach
    with open('results/'+filename+'_coefs_avg', 'rb') as handle:
        the_m = pickle.load(handle)
    with open('results/'+filename+'_coefs_std', 'rb') as handle:
        the_sd = pickle.load(handle)
    the_m = [the_m[6], the_m[7], the_m[8], the_m[3], the_m[4], the_m[5], the_m[0], the_m[1], the_m[2]] # reordering spill from -2 to i
    the_sd = [the_sd[6], the_sd[7], the_sd[8], the_sd[3], the_sd[4], the_sd[5], the_sd[0], the_sd[1], the_sd[2]]
    fig1 = plt.figure(dpi = 300)
    ax1 = SubplotHost(fig1, 111)
    fig1.add_subplot(ax1)
    # Some data
    x = [1,2,3,5,6,7,9,10,11]
    y = the_m
    # First X-axis
    ax1.bar(x, y, color = [color1, color2, color3]*3,width=0.9, zorder=4)
    ax1.errorbar(x, y, yerr=the_sd, fmt='none', elinewidth=1, capsize=4, color="black", zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["S", "F", "L"]*3)
    #ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
    ax1.yaxis.set_label_text("Weights")
    ax1.set_xlim(0, 12)
    ax1.set_ylim(ymin, ymax)
    ax1.axhline(y=0, color='black', linestyle='-', zorder=6, linewidth=1)
    # Second X-axis
    ax2 = ax1.twiny()
    offset = 0, -25 # Position of the second axis
    new_axisline = ax2.get_grid_helper().new_fixed_axis
    ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
    ax2.axis["top"].set_visible(False)
    
    ax2.set_xticks([0.0, 0.3333, 0.6666, 1])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([1/6, 0.5, 5/6]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(["w$_{i-2}$","w$_{i-1}$","w$_{i}$"]))
    ax2.axis["bottom"].minor_ticks.set_ticksize(0)
    # Third X-axis
    ax3 = ax1.twiny()
    offset = 0, -50
    new_axisline = ax3.get_grid_helper().new_fixed_axis
    ax3.axis["bottom"] = new_axisline(loc="bottom", axes=ax3, offset=offset)
    #ax3.axis["top"].set_visible(False)
    
    ax3.set_xticks([0.0, 1.0])
    ax3.xaxis.set_major_formatter(ticker.NullFormatter())
    ax3.xaxis.set_minor_locator(ticker.FixedLocator([0.5]))
    ax3.xaxis.set_minor_formatter(ticker.FixedFormatter([modelname]))
    ax3.axis["bottom"].minor_ticks.set_ticksize(0)
    ax1.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=3)
    #plt.minorticks_on()
    ax1.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=2)
    plt.title(title)
    plt.show()

def plot_weights_spill_withEMB(filename, title, ymin = -.01, ymax=.015, modelname=""):
    plt.rcParams.update({'font.size': 13})
    color1 = '#9b90c2'  # Pastel purple
    color2 = '#87d3f8'  # Pastel blue
    color3 = '#f8b195'  # Pastel peach
    color4 = '#FAC8CD'
    with open('results/'+filename+'_coefs_avg', 'rb') as handle:
        the_m = pickle.load(handle)
    with open('results/'+filename+'_coefs_std', 'rb') as handle:
        the_sd = pickle.load(handle)
    hidden_size = int((the_m.shape[0]-9)/3)
    
    hidden_i = the_m[9+hidden_size*2:]
    hidden_1 = the_m[9+hidden_size:9+hidden_size*2]
    hidden_2 = the_m[9:9+hidden_size]
    
    m_i, sd_i = np.sum([abs(w) for w in hidden_i])/len(hidden_i), np.std(hidden_i)
    m_1, sd_1 = np.sum([abs(w) for w in hidden_1])/len(hidden_1), np.std(hidden_1)
    m_2, sd_2 = np.sum([abs(w) for w in hidden_2])/len(hidden_2), np.std(hidden_2)
    
    the_m1 = [the_m[6], the_m[7], the_m[8], m_2, the_m[3], the_m[4], the_m[5], m_1, the_m[0], the_m[1], the_m[2], m_i] # reordering spill from -2 to i
    the_sd1 = [the_sd[6], the_sd[7], the_sd[8], sd_1, the_sd[3], the_sd[4], the_sd[5], sd_1,  the_sd[0], the_sd[1], the_sd[2], sd_i]
    
    fig1 = plt.figure(dpi = 300)
    ax1 = SubplotHost(fig1, 111)
    fig1.add_subplot(ax1)
    # Some data
    x = [1,2,3,4,6,7,8,9,11,12,13,14]
    y = the_m1
    # First X-axis
    ax1.bar(x, the_m1, color = [color1, color2, color3, color4]*3,width=0.9, zorder=4)
    ax1.errorbar(x, the_m1, yerr=the_sd1, fmt='none', elinewidth=1, capsize=4, color="black", zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["S", "F", "L", "$\overline{EMB}$"]*3)
    #ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
    ax1.yaxis.set_label_text("Weights")
    ax1.set_xlim(0, 15)
    ax1.set_ylim(ymin, ymax)
    ax1.axhline(y=0, color='black', linestyle='-', zorder=6, linewidth=1)
    # Second X-axis
    ax2 = ax1.twiny()
    offset = 0, -25 # Position of the second axis
    new_axisline = ax2.get_grid_helper().new_fixed_axis
    ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
    ax2.axis["top"].set_visible(False)
    
    ax2.set_xticks([0.0, 0.3333, 0.6666, 1])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([1/6, 0.5, 5/6]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(["w$_{i-2}$","w$_{i-1}$","w$_{i}$"]))
    ax2.axis["bottom"].minor_ticks.set_ticksize(0)
    # Third X-axis
    ax3 = ax1.twiny()
    offset = 0, -50
    new_axisline = ax3.get_grid_helper().new_fixed_axis
    ax3.axis["bottom"] = new_axisline(loc="bottom", axes=ax3, offset=offset)
    #ax3.axis["top"].set_visible(False)
    ax3.set_xticks([0.0, 1.0])
    ax3.xaxis.set_major_formatter(ticker.NullFormatter())
    ax3.xaxis.set_minor_locator(ticker.FixedLocator([0.5]))
    ax3.xaxis.set_minor_formatter(ticker.FixedFormatter([modelname]))
    ax3.axis["bottom"].minor_ticks.set_ticksize(0)
    ax1.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=3)
    #plt.minorticks_on()
    ax1.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=2)
    plt.title(title)
    plt.show()

        
        
