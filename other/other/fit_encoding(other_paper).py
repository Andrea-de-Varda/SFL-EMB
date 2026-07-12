print(">>> SCRIPT LAUNCHED", flush=True)
import numpy as np
import numpy.ma as ma
import pandas as pd
from os import chdir
import os
import pickle
# from sklearn.linear_model import RidgeCV
from himalaya.ridge import RidgeCV ## Apparently Kernel Ridge is faster (in most of the cases, apart some across, multilang) -->  Solving ridge is slower than solving kernel ridge when n_samples < n_features (here 117 < 1024). Using a linear kernel in himalaya.kernel_ridge.KernelRidgeCV or himalaya.kernel_ridge.solve_kernel_ridge_cv_eigenvalues would be faster. ---- default kernel is linear
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend, get_backend
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*numpy\\.core\\.numeric is deprecated.*")
import argparse
import torch

# chdir("/home/dev/Documents/PhD/Alice")

# backend stuff
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
BACKEND = "torch_cuda" if DEVICE == "cuda" else "numpy"
set_backend(BACKEND)
print(f"[INFO] Himalaya backend set to {get_backend()} ({DEVICE})", flush=True)

def to_backend(a):
    if not BACKEND.startswith("torch"):
        return a
    # np array case
    if isinstance(a, np.ndarray):
        if a.dtype != np.float32:
            a = a.astype(np.float32, copy=False)
        return torch.from_numpy(a).to(DEVICE)
    # torch tensor case
    if torch.is_tensor(a):
        return a.to(dtype=torch.float32, device=DEVICE)
    return a

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(name):
    # print(f"Loading {name}", flush=True)
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def load_split_one_layer(model_prefix, lang, layer_n):
    data = load(f"split_context/{model_prefix}_{lang}")
    return {c: data[c][layer_n] for c in sorted(data.keys())}

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def embed_words(embeddings, words_id):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2) # fMRI data is composed by 130 images, each acquired every 2 sec (total 260 sec)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0) # average embeddings for the 2 sec time window
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, embeddings):
    df = pd.read_csv("transcribed/"+lang+".csv") # load transcription obtained with whisper
    df = df[df["end"] <= 260]   # remove words after 260 sec (not scanned)
    time = np.arange(0, 260, 2) # acquired every 2 sec (see embed words)
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    for i in range(len(time_words)): # find the words corresponding to each TR (pronounced between the start and the end of a TR)
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

def preproc_align_split(lang, layer_chunks):
    df = pd.read_csv(f"transcribed/{lang}.csv")
    df = df[df["end"] <= 260].reset_index(drop=True)
    X_all = []
    for chunk_num, start in enumerate(range(0, 260, 26)):
        end = start + 26
        sub = df[(df["end"] > start) & (df["end"] < end)].copy()
        W = layer_chunks[chunk_num]
        n_words = len(sub)
        if W.shape[0] != n_words:
            n = min(W.shape[0], n_words)
            sub = sub.iloc[:n]
            W = W[:n]
        time = np.arange(start, end, 2)
        time_words = sub["end"].to_numpy()
        words_id = np.searchsorted(time, time_words, side="right") - 1
        words_id = np.clip(words_id, 0, len(time) - 1)
        X_chunk = np.full((len(time), W.shape[1]), np.nan, dtype=W.dtype)
        for i in range(len(time)):
            sel = (words_id == i)
            if np.any(sel):
                X_chunk[i] = W[sel].mean(axis=0)
        X_chunk = imputate_na(X_chunk)
        X_all.append(X_chunk)
    return np.vstack(X_all)

def test_model_Ridge(X, y_part1, y_part2, n, saveto, save_results = False, shuffle=False, prefix = ""):
    if shuffle: # note that shuffling might artificially increase the encoding scores. Default is non-shuffled. All the analyses now are w/o shuffling.
        kf = KFold(n_splits=n, shuffle=True, random_state = 0)
    else:
        kf = KFold(n_splits=n, shuffle=False)
    out_reg = []
    out_pred = []; y_tot = []
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    for train_index, test_index in kf.split(X): # tqdm(kf.split(X), total=n):
        # Normalizing embedding and responses. The normalization parameters are always estimated on the training data and transferred to the test data
        X_train = X_scaler.fit_transform(X[train_index])
        X_test = X_scaler.transform(X[test_index])
        y_train = y_scaler.fit_transform(y_part1[train_index].reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_part2[test_index].reshape(-1, 1)).flatten()
        reg = KernelRidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)) # Alpha values log spaced. Alpha chosen with leave-one-out nested CV
        # reg.fit(X_train, y_train)
        reg.fit(to_backend(X_train), to_backend(y_train[:, None]))
        y_pred = reg.predict(to_backend(X_test)).squeeze()
        if BACKEND.startswith("torch"):
            y_pred = y_pred.cpu().numpy()
        # y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_pred.extend(y_pred.tolist())
        y_tot.extend(y_test.tolist())
        out_reg.append(r)
    #print(round(np.mean(out_reg), 4))
    r_tot = pearsonr(out_pred, y_tot)[0]
    # print(round(r_tot, 4))
    out_predictions = [y_tot, out_pred]
    if save_results:
        save(r_tot, f"results/rs/{prefix}{saveto}")
        save(out_reg, f"results/out_reg/{prefix}{saveto}") # saving all rs and coefficients for later use
        save(out_predictions, f"results/predictions/{prefix}{saveto}")
    return r_tot

def monolingual_encoding(langs, model_prefix, n_layers, d, shuffle=False, prefix = "", overwrite = False):
    # This simply repeats the process for (a) all the languages in the sample and (b) all the layers of a given model from which embeddings are available
    layerwise_dict = {}
    if os.path.isfile(f"results/monolingual_{prefix}{model_prefix}_all") and overwrite == False:
        print(f"Encoding for {model_prefix} already done", flush = True)
        return
    else:
        # Encoding evaluated layer by layer
        frois = list(d[lang_code_dict[langs[0]]][list(d[lang_code_dict[langs[0]]].keys())[0]].keys())
        for froi_idx, froi in enumerate(frois):
            print(f"\n\nProcessing {froi} ({froi_idx+1}/{len(frois)})", flush = True)
            for n in range(n_layers+1):
                # print(f"Processing layer {n}")
                fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
                #########################
                m1 = []; m2 = []
                for idx, lang in enumerate(langs):
                    part1, part2 = d[lang_code_dict[lang]].keys()
                    ts1, ts2 = d[lang_code_dict[lang]][part1][froi], d[lang_code_dict[lang]][part2][froi]
                    the_r1 = test_model_Ridge(fmri_data[idx], ts1, ts2, 10, saveto = f"{model_prefix}_{lang}_{froi}_part1_{n}", shuffle=shuffle, prefix = prefix) # trying in both directions
                    the_r2 = test_model_Ridge(fmri_data[idx], ts2, ts1, 10, saveto = f"{model_prefix}_{lang}_{froi}_part1_{n}", shuffle=shuffle, prefix = prefix)
                    m1.append(the_r1)
                    m2.append(the_r2)
                #########################
                mean_r = np.mean(m1+m2)
                #print(f"All rs = {m}")
                # print(f"Mean r = {round(mean_r, 4)} ({model_prefix} - {froi} - {n})")
                print(f"WITHIN     {model_prefix} | fROI={froi} | layer={n} | mean r={mean_r:.3f}", flush=True)
                #########################
                df = pd.DataFrame(zip(langs, m1, m2), columns=["lang", "m1", "m2"])
                df["m"] = df[["m1", "m2"]].mean(axis=1)
                layerwise_dict[n] = df
            save(layerwise_dict, f"results/monolingual_{prefix}{model_prefix}_{froi}")
    return layerwise_dict

def multilingual_encoding(langs, model_prefix, n_layers, d, prefix = "", overwrite = False):
    # This function trains encoding models in one language and evaluates the encoding in all the left-out languages
    kf = KFold(n_splits=10, shuffle=False)
    if os.path.isfile(f"results/multilingual_{prefix}{model_prefix}_all") and overwrite == False:
        print(f"Encoding for {model_prefix} already done", flush = True)
        return
    else:
        kf = KFold(n_splits=10, shuffle=False)
        frois = list(d[lang_code_dict[langs[0]]][list(d[lang_code_dict[langs[0]]].keys())[0]].keys())
        for froi_idx, froi in enumerate(frois):
            layerwise_dict = {}
            print(f"\n\nProcessing {froi} ({froi_idx+1}/{len(frois)})", flush = True)
            for n in range(n_layers+1):
                # print(f"Processing layer {n}")
                fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
                ############################
                out_predictions = []
                for i in range(len(langs)): # for each language, train (KF) in that language
                    results_d_singlelang = {}
                    for train_index, test_index in kf.split(fmri_data[0]):
                        # train on 9/10 of the data in one language
                        y_name = langs[i]
                        part1, part2 = d[lang_code_dict[y_name]].keys()
                        X_train_ = fmri_data[i][train_index] # Training data (X) includes only lang_i
                        y_train_1 = d[lang_code_dict[y_name]][part1][froi][train_index] # two participants -- train on both
                        y_train_2 = d[lang_code_dict[y_name]][part2][froi][train_index]
                        X_train = np.concatenate([X_train_, X_train_])
                        y_train = np.concatenate([y_train_1, y_train_2])
                        # scaling (using for test directly)
                        X_scaler = StandardScaler()
                        y_scaler = StandardScaler()
                        X_train = X_scaler.fit_transform(X_train)
                        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                        reg = KernelRidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        # reg.fit(X_train, y_train)
                        reg.fit(to_backend(X_train), to_backend(y_train[:, None]))
                        for j in range(len(langs)):
                            #if j != i: # !! otherwise, train-test in same lang # on second thought, keeping it to have square matrices and aligned dims
                            test_lang = langs[j]
                            part1, part2 = d[lang_code_dict[test_lang]].keys()
                            X_test = fmri_data[j][test_index]
                            y_test_1 = d[lang_code_dict[test_lang]][part1][froi][test_index]
                            y_test_2 = d[lang_code_dict[test_lang]][part2][froi][test_index]
                            # y_test = np.mean([y_test_1, y_test_2], axis = 0)
                            X_test = X_scaler.transform(X_test)
                            y_test_1 = y_scaler.transform(y_test_1.reshape(-1, 1)).flatten()
                            y_test_2 = y_scaler.transform(y_test_2.reshape(-1, 1)).flatten()
                            # y_pred = reg.predict(X_test)
                            y_pred = reg.predict(to_backend(X_test)).squeeze()
                            if BACKEND.startswith("torch"):
                                y_pred = y_pred.cpu().numpy()
                            try:
                                results_d_singlelang[test_lang]["prediction"].extend(y_pred.tolist())
                                results_d_singlelang[test_lang]["target1"].extend(y_test_1.tolist())
                                results_d_singlelang[test_lang]["target2"].extend(y_test_2.tolist())
                            except KeyError:
                                results_d_singlelang[test_lang] = {"prediction" : y_pred.tolist(), 
                                                                   "target1" : y_test_1.tolist(),
                                                                   "target2" : y_test_2.tolist()}
                    # 1 X participant
                    d_corr1 = {lang : pearsonr(results_d_singlelang[lang]["prediction"], results_d_singlelang[lang]["target1"])[0] for lang in langs}
                    d_corr2 = {lang : pearsonr(results_d_singlelang[lang]["prediction"], results_d_singlelang[lang]["target2"])[0] for lang in langs}
                    d_avg = {k: (d_corr1[k] + d_corr2[k]) / 2 for k in d_corr1.keys()} # avg over 2 participants in y_test
                    d_avg["target_lang"] = y_name
                    mean_other = sum(v for k, v in d_avg.items() if k != y_name and k != "r" and k != "target_lang") / (len(d_avg) - 2)
                    d_avg["r"] = mean_other
                    out_predictions.append(d_avg)
                out_predictions = pd.DataFrame(out_predictions)
                layerwise_dict[n] = out_predictions
                mean_r = out_predictions["r"].mean()
                # print(f"Mean r = {mean_r} ({model_prefix} - {n})")
                print(f"ACROSS     {model_prefix} | fROI={froi} | layer={n} | mean r={mean_r:.3f}", flush=True)
            save(layerwise_dict, f"results/multilingual_{prefix}{model_prefix}_{froi}")
    return layerwise_dict

def multilingual_encoding_multitrain(langs, model_prefix, n_layers, d, prefix = "", overwrite = False):
    # This function trains encoding models in a set of languages (all but one) and evaluates the encoding in the left-out language
    kf = KFold(n_splits=10, shuffle=False)
    frois = list(d[lang_code_dict[langs[0]]][list(d[lang_code_dict[langs[0]]].keys())[0]].keys())
    for froi_idx, froi in enumerate(frois):
        filename = f"results/multilingual_multitrain_{prefix}{model_prefix}_{froi}"
        layerwise_dict = {}
        if os.path.isfile(filename) and overwrite == False:
            print(f"Encoding for {model_prefix} - {froi} already done", flush = True)
            continue
        else:
            print(f"\n\nProcessing {froi} ({froi_idx+1}/{len(frois)})", flush = True)
            for n in range(n_layers+1):
                # print(f"Processing layer {n}")
                fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
                ############################
                out_predictions = []
                for i in range(len(langs)): # for each language, train (KF) in that language
                    results_d_singlelang = {}
                    for train_index, test_index in kf.split(fmri_data[0]):
                        # train on 9/10 of the data in all languages but one
                        X_train_ = np.concatenate([fmri_data[j][train_index] for j in range(len(langs)) if j != i]) # Training data (X) includes all langs j but lang_i
                        y_train_1 = []; y_train_2 = []
                        for j in range(len(langs)):
                            if j != i:
                                y_name = langs[j]
                                part1, part2 = d[lang_code_dict[y_name]].keys()
                                y_train_1.append(d[lang_code_dict[y_name]][part1][froi][train_index]) # two participants -- train on both
                                y_train_2.append(d[lang_code_dict[y_name]][part2][froi][train_index])
                        y_train_1 = np.concatenate(y_train_1)
                        y_train_2 = np.concatenate(y_train_2)
                        X_train = np.concatenate([X_train_, X_train_]) # because 2 participants per language!
                        y_train = np.concatenate([y_train_1, y_train_2])
                        # scaling (using for test directly)
                        X_scaler = StandardScaler()
                        y_scaler = StandardScaler()
                        X_train = X_scaler.fit_transform(X_train)
                        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                        # here, we're training on more data so sometimes standard ridgeCV is faster (depending on embedding dim) -- choose based on shape
                        if X_train.shape[0] > X_train.shape[1]:
                            reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        else:
                            reg = KernelRidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        # reg = KernelRidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        # reg.fit(X_train, y_train)
                        reg.fit(to_backend(X_train), to_backend(y_train[:, None]))
                        # testing
                        test_lang = langs[i]
                        part1, part2 = d[lang_code_dict[test_lang]].keys()
                        X_test = fmri_data[i][test_index]
                        y_test_1 = d[lang_code_dict[test_lang]][part1][froi][test_index]
                        y_test_2 = d[lang_code_dict[test_lang]][part2][froi][test_index]
                        # y_test = np.mean([y_test_1, y_test_2], axis = 0)
                        X_test = X_scaler.transform(X_test)
                        y_test_1 = y_scaler.transform(y_test_1.reshape(-1, 1)).flatten()
                        y_test_2 = y_scaler.transform(y_test_2.reshape(-1, 1)).flatten()
                        # y_pred = reg.predict(X_test)
                        y_pred = reg.predict(to_backend(X_test)).squeeze()
                        if BACKEND.startswith("torch"):
                            y_pred = y_pred.cpu().numpy()
                        try:
                            results_d_singlelang[test_lang]["prediction"].extend(y_pred.tolist())
                            results_d_singlelang[test_lang]["target1"].extend(y_test_1.tolist())
                            results_d_singlelang[test_lang]["target2"].extend(y_test_2.tolist())
                        except KeyError:
                            results_d_singlelang[test_lang] = {"prediction" : y_pred.tolist(), 
                                                                "target1" : y_test_1.tolist(),
                                                                "target2" : y_test_2.tolist()}
                    # 1 X participant
                    d_corr1 = pearsonr(results_d_singlelang[test_lang]["prediction"], results_d_singlelang[test_lang]["target1"])[0]
                    d_corr2 = pearsonr(results_d_singlelang[test_lang]["prediction"], results_d_singlelang[test_lang]["target2"])[0]
                    d_avg = {"target_lang": test_lang,
                            "r": 0.5 * (d_corr1 + d_corr2)}
                    out_predictions.append(d_avg)
                out_predictions = pd.DataFrame(out_predictions)
                layerwise_dict[n] = out_predictions
                mean_r = out_predictions["r"].mean()
                print(f"ACROSS-multilang-train     {model_prefix} | fROI={froi} | layer={n} | mean r={mean_r:.3f}", flush=True)
            save(layerwise_dict, filename)
    return layerwise_dict

def monolingual_encoding_circshift(langs, model_prefix, n_layers, d, prefix="", overwrite=False, shift_vals=(26, 52, 78, 104), shuffle=False):
    kf = KFold(n_splits=10, shuffle=False)
    frois = list(d[lang_code_dict[langs[0]]][list(d[lang_code_dict[langs[0]]].keys())[0]].keys())
    print(f"Processing {frois}", flush = True)
    for froi_idx, froi in enumerate(sorted(frois, key=str.lower)):
        filepath = f"results/monolingual_{prefix}{model_prefix}_{froi}_circshift"
        out_all_shifts = {}
        if os.path.isfile(filepath) and not overwrite:
            print(f"Encoding for {model_prefix} – {froi} already done", flush = True)
            continue
        print(f"\n\nProcessing {froi} ({froi_idx + 1}/{len(frois)})", flush = True)
        for shift in shift_vals:
            print(f"\n  >>> Circular shift = {shift}", flush = True)
            layerwise_dict = {}
            fmri_layers = {n: [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs] for n in range(n_layers + 1)}
            for n in range(n_layers + 1):
                fmri_data = fmri_layers[n]
                m1, m2 = [], []
                for idx, lang in enumerate(langs):
                    part1, part2 = d[lang_code_dict[lang]].keys()
                    y1 = np.roll(d[lang_code_dict[lang]][part1][froi], shift)
                    y2 = np.roll(d[lang_code_dict[lang]][part2][froi], shift)
                    r1 = test_model_Ridge(fmri_data[idx], y1, y2, 10, saveto=f"{model_prefix}_{lang}_{froi}_part1_{n}_shift{shift}", shuffle=shuffle, prefix=prefix,)
                    r2 = test_model_Ridge(fmri_data[idx], y2, y1, 10, saveto=f"{model_prefix}_{lang}_{froi}_part2_{n}_shift{shift}", shuffle=shuffle, prefix=prefix,)
                    m1.append(r1)
                    m2.append(r2)
                df = pd.DataFrame(zip(langs, m1, m2), columns=["lang", "m1", "m2"])
                df["m"] = df[["m1", "m2"]].mean(axis=1)
                layerwise_dict[n] = df
                mean_r = df["m"].mean()
                print(f"WITHIN‑CIRC {model_prefix} | shift={shift} | fROI={froi} | layer={n} | mean r={mean_r:.3f}", flush=True)
            out_all_shifts[shift] = layerwise_dict
        save(out_all_shifts, filepath)
    return out_all_shifts

def multilingual_encoding_circshift(langs, model_prefix, n_layers, d, prefix = "", overwrite = False, shift_vals = (26, 52, 78, 104)):
    kf = KFold(n_splits=10, shuffle=False)
    frois = list(d[lang_code_dict[langs[0]]][list(d[lang_code_dict[langs[0]]].keys())[0]].keys())
    for froi_idx, froi in enumerate(sorted(frois, key=str.lower)):
        out_all_shifts = {}
        filepath = f"results/multilingual_{prefix}{model_prefix}_{froi}_circshift"
        if os.path.isfile(filepath) and not overwrite:
            print(f"Encoding for {model_prefix} – {froi} already done", flush = True)
            continue
        print(f"\n\nProcessing {froi} ({froi_idx+1}/{len(frois)})", flush = True)
        for shift in shift_vals:
            print(f"\n  >>> Circular shift = {shift}", flush = True)
            layerwise_dict = {}
            # pre‑compute fmri reps (X) for all layers and langs to avoid recomputation per split
            fmri_layers = {
                n: [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
                for n in range(n_layers + 1)
            }
            for n in range(n_layers + 1):
                # print(f"    Layer {n}")
                fmri_data = fmri_layers[n]
                out_predictions = []
                for i in range(len(langs)): # training language
                    y_name = langs[i]
                    results_d_singlelang = {}
                    for train_idx, test_idx in kf.split(fmri_data[0]):
                        part1, part2 = d[lang_code_dict[y_name]].keys()
                        # X train
                        X_train_ = fmri_data[i][train_idx]
                        # y train (both participants) + circular shift
                        y1 = d[lang_code_dict[y_name]][part1][froi][train_idx]
                        y2 = d[lang_code_dict[y_name]][part2][froi][train_idx]
                        # y_train = np.concatenate([y1, y2])
                        # y_train = np.roll(y_train, shift)
                        y_train = np.concatenate([np.roll(y1, shift), np.roll(y2, shift)])
                        # scaling
                        X_scaler = StandardScaler()
                        y_scaler = StandardScaler()
                        X_train = X_scaler.fit_transform(np.concatenate([X_train_, X_train_]))
                        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                        reg = KernelRidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        # reg.fit(X_train, y_train)
                        reg.fit(to_backend(X_train), to_backend(y_train[:, None]))
                        # evaluate on every language
                        for j, test_lang in enumerate(langs):
                            part1_t, part2_t = d[lang_code_dict[test_lang]].keys()
                            X_test  = X_scaler.transform(fmri_data[j][test_idx])
                            y_t1    = d[lang_code_dict[test_lang]][part1_t][froi][test_idx]
                            y_t2    = d[lang_code_dict[test_lang]][part2_t][froi][test_idx]
                            y_t1    = np.roll(y_t1, shift)  # same shift for test targets
                            y_t2    = np.roll(y_t2, shift)
                            y_t1    = y_scaler.transform(y_t1.reshape(-1,1)).flatten()
                            y_t2    = y_scaler.transform(y_t2.reshape(-1,1)).flatten()
                            # y_pred  = reg.predict(X_test)
                            y_pred = reg.predict(to_backend(X_test)).squeeze()
                            if BACKEND.startswith("torch"):
                                y_pred = y_pred.cpu().numpy()
                            try:
                                rds = results_d_singlelang[test_lang]
                                rds["prediction"].extend(y_pred.tolist())
                                rds["target1"].extend(y_t1.tolist())
                                rds["target2"].extend(y_t2.tolist())
                            except KeyError:
                                results_d_singlelang[test_lang] = {
                                    "prediction": y_pred.tolist(),
                                    "target1":    y_t1.tolist(),
                                    "target2":    y_t2.tolist()
                                }
                    # correlations (avg over two participants)
                    d_corr1 = {lang: pearsonr(rds["prediction"], rds["target1"])[0]
                               for lang, rds in results_d_singlelang.items()}
                    d_corr2 = {lang: pearsonr(rds["prediction"], rds["target2"])[0]
                               for lang, rds in results_d_singlelang.items()}
                    d_avg   = {k: (d_corr1[k] + d_corr2[k]) / 2 for k in d_corr1}
                    d_avg["target_lang"] = y_name
                    mean_other = np.mean([v for k,v in d_avg.items()
                                          if k not in {y_name, "target_lang", "r"}])
                    d_avg["r"] = mean_other
                    out_predictions.append(d_avg)
                df_layer = pd.DataFrame(out_predictions)
                layerwise_dict[n] = df_layer
                mean_r = df_layer["r"].mean()
                print(f"ACROSS-CIRC     {model_prefix} | shift={shift} | layer={n} | mean r={mean_r:.3f}", flush=True)
            out_all_shifts[shift] = layerwise_dict
        save(out_all_shifts, filepath)
    return out_all_shifts

def multilingual_encoding_multitrain_circshift(langs, model_prefix, n_layers, d, prefix="", overwrite=False, shift_vals=(26, 52, 78, 104),):
    kf = KFold(n_splits=10, shuffle=False)
    frois = list(d[lang_code_dict[langs[0]]][list(d[lang_code_dict[langs[0]]].keys())[0]].keys())
    for froi_idx, froi in enumerate(sorted(frois, key=str.lower)):
        out_all_shifts = {}
        filepath = f"results/multilingual_multitrain_{prefix}{model_prefix}_{froi}_circshift"
        if os.path.isfile(filepath) and not overwrite:
            print(f"Encoding for {model_prefix} – {froi} already done", flush=True)
            continue
        print(f"\n\nProcessing {froi} ({froi_idx+1}/{len(frois)})", flush=True)
        fmri_layers = {n: [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]for n in range(n_layers + 1)}
        for shift in shift_vals:
            print(f"\n  >>> Circular shift = {shift}", flush=True)
            layerwise_dict = {}
            for n in range(n_layers + 1):
                fmri_data = fmri_layers[n]
                out_predictions = []
                # loop over held‑out language
                for i, test_lang in enumerate(langs):
                    results_single = {"prediction": [], "target1": [], "target2": []}
                    for train_idx, test_idx in kf.split(fmri_data[i]):  # split on held‑out
                        # build training data from all languages except held‑out
                        X_train_ = np.concatenate([fmri_data[j][train_idx] for j in range(len(langs)) if j != i])
                        y_train_1, y_train_2 = [], []
                        for j in range(len(langs)):
                            if j != i:
                                lang_j = langs[j]
                                p1, p2 = d[lang_code_dict[lang_j]].keys()
                                y_train_1.append(np.roll(d[lang_code_dict[lang_j]][p1][froi][train_idx],shift))
                                y_train_2.append(np.roll(d[lang_code_dict[lang_j]][p2][froi][train_idx],shift))
                        y_train_1 = np.concatenate(y_train_1)
                        y_train_2 = np.concatenate(y_train_2)
                        X_train = np.concatenate([X_train_, X_train_])
                        y_train = np.concatenate([y_train_1, y_train_2])
                        # scale + fit
                        X_scaler = StandardScaler()
                        y_scaler = StandardScaler()
                        X_train = X_scaler.fit_transform(X_train)
                        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                        # reg = KernelRidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        if X_train.shape[0] > X_train.shape[1]:
                            reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        else:
                            reg = KernelRidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        # reg = KernelRidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                        # reg.fit(X_train, y_train)
                        reg.fit(to_backend(X_train), to_backend(y_train[:, None]))
                        # test on held‑out language
                        p1_t, p2_t = d[lang_code_dict[test_lang]].keys()
                        X_test = X_scaler.transform(fmri_data[i][test_idx])
                        y_t1 = np.roll(d[lang_code_dict[test_lang]][p1_t][froi][test_idx], shift)
                        y_t2 = np.roll(d[lang_code_dict[test_lang]][p2_t][froi][test_idx], shift)
                        y_t1 = y_scaler.transform(y_t1.reshape(-1, 1)).flatten()
                        y_t2 = y_scaler.transform(y_t2.reshape(-1, 1)).flatten()
                        # y_pred = reg.predict(X_test)
                        y_pred = reg.predict(to_backend(X_test)).squeeze()
                        if BACKEND.startswith("torch"):
                            y_pred = y_pred.cpu().numpy()
                        results_single["prediction"].extend(y_pred.tolist())
                        results_single["target1"].extend(y_t1.tolist())
                        results_single["target2"].extend(y_t2.tolist())
                    # correlation (average over 2 participants)
                    r1 = pearsonr(results_single["prediction"], results_single["target1"])[0]
                    r2 = pearsonr(results_single["prediction"], results_single["target2"])[0]
                    out_predictions.append({"target_lang": test_lang, "r": 0.5 * (r1 + r2)})
                df_layer = pd.DataFrame(out_predictions)
                layerwise_dict[n] = df_layer
                mean_r = df_layer["r"].mean()
                print(f"ACROSS‑CIRC‑MULTITRAIN {model_prefix} | shift={shift} | fROI={froi} | layer={n} | mean r={mean_r:.3f}",flush=True,)
            out_all_shifts[shift] = layerwise_dict
        save(out_all_shifts, filepath)
    return out_all_shifts

# split-context functions
def monolingual_encoding_split(langs, model_prefix, n_layers, d, shuffle=False, prefix="", overwrite=True):
    # WITHIN-SPLIT: train on 9/10 folds of subject A, test on subject B (and vice versa)
    out_tag = f"results/monolingual_{prefix}{model_prefix}_SPLIT_all"
    if os.path.isfile(out_tag) and not overwrite:
        print(f"Encoding (SPLIT) for {model_prefix} already done", flush=True)
        return
    frois = list(d[lang_code_dict[langs[0]]][list(d[lang_code_dict[langs[0]]].keys())[0]].keys())
    for froi_idx, froi in enumerate(frois):
        layerwise = {}
        print(f"\n\n[WITHIN-SPLIT] Processing {froi} ({froi_idx+1}/{len(frois)})", flush=True)
        for n in range(n_layers + 1):
            fmri_data = []
            for lang in langs:
                layer_chunks = load_split_one_layer(model_prefix, lang, n)
                X_lang = preproc_align_split(lang, layer_chunks)
                fmri_data.append(X_lang)
            m1, m2 = [], []
            for idx, lang in enumerate(langs):
                part1, part2 = d[lang_code_dict[lang]].keys()
                ts1 = d[lang_code_dict[lang]][part1][froi]
                ts2 = d[lang_code_dict[lang]][part2][froi]
                r12 = test_model_Ridge(fmri_data[idx], ts1, ts2, 10, saveto=f"{model_prefix}_{lang}_{froi}_SPLIT_part1_{n}", shuffle=shuffle, prefix=prefix)
                r21 = test_model_Ridge(fmri_data[idx], ts2, ts1, 10, saveto=f"{model_prefix}_{lang}_{froi}_SPLIT_part2_{n}", shuffle=shuffle, prefix=prefix)
                m1.append(r12); m2.append(r21)
            df = pd.DataFrame(zip(langs, m1, m2), columns=["lang", "m1", "m2"])
            df["m"] = df[["m1", "m2"]].mean(axis=1)
            layerwise[n] = df
            print(f"WITHIN-SPLIT {model_prefix} | fROI={froi} | layer={n} | mean r={df['m'].mean():.3f}", flush=True)
        save(layerwise, f"results/monolingual_{prefix}{model_prefix}_{froi}_SPLIT")
    save({"done": True}, out_tag)

def multilingual_encoding_multitrain_split(langs, model_prefix, n_layers, d, prefix="", overwrite=False):
    kf = KFold(n_splits=10, shuffle=False)
    frois = list(d[lang_code_dict[langs[0]]][list(d[lang_code_dict[langs[0]]].keys())[0]].keys())
    for froi_idx, froi in enumerate(frois):
        outpath = f"results/multilingual_multitrain_{prefix}{model_prefix}_{froi}_SPLIT"
        if os.path.isfile(outpath) and not overwrite:
            print(f"Encoding (SPLIT) for {model_prefix} – {froi} already done", flush=True)
            continue
        layerwise_dict = {}
        print(f"\n\n[ACROSS-SPLIT] Processing {froi} ({froi_idx+1}/{len(frois)})", flush=True)
        for n in range(n_layers + 1):
            fmri_data = []
            for lang in langs:
                layer_chunks = load_split_one_layer(model_prefix, lang, n)
                X_lang = preproc_align_split(lang, layer_chunks)
                fmri_data.append(X_lang)
            out_predictions = []
            for i, test_lang in enumerate(langs):
                results_single = {"prediction": [], "target1": [], "target2": []}
                for train_idx, test_idx in kf.split(fmri_data[i]):
                    X_train_ = np.concatenate([fmri_data[j][train_idx] for j in range(len(langs)) if j != i])
                    y_train_1, y_train_2 = [], []
                    for j in range(len(langs)):
                        if j != i:
                            lang_j = langs[j]
                            p1, p2 = d[lang_code_dict[lang_j]].keys()
                            y_train_1.append(d[lang_code_dict[lang_j]][p1][froi][train_idx])
                            y_train_2.append(d[lang_code_dict[lang_j]][p2][froi][train_idx])
                    y_train_1 = np.concatenate(y_train_1)
                    y_train_2 = np.concatenate(y_train_2)
                    X_train = np.concatenate([X_train_, X_train_])
                    y_train = np.concatenate([y_train_1, y_train_2])
                    X_scaler = StandardScaler()
                    y_scaler = StandardScaler()
                    X_train = X_scaler.fit_transform(X_train)
                    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                    if X_train.shape[0] > X_train.shape[1]:
                        reg = RidgeCV(alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4))
                    else:
                        reg = KernelRidgeCV(alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4))
                    reg.fit(to_backend(X_train), to_backend(y_train[:, None]))
                    def _predict(X_np):
                        y_hat = reg.predict(to_backend(X_np)).squeeze()
                        if BACKEND.startswith("torch"):
                            y_hat = y_hat.cpu().numpy()
                        return y_hat
                    p1_t, p2_t = d[lang_code_dict[test_lang]].keys()
                    X_test = X_scaler.transform(fmri_data[i][test_idx])
                    y_t1 = y_scaler.transform(d[lang_code_dict[test_lang]][p1_t][froi][test_idx].reshape(-1, 1)).flatten()
                    y_t2 = y_scaler.transform(d[lang_code_dict[test_lang]][p2_t][froi][test_idx].reshape(-1, 1)).flatten()
                    y_pred = _predict(X_test)
                    results_single["prediction"].extend(y_pred.tolist())
                    results_single["target1"].extend(y_t1.tolist())
                    results_single["target2"].extend(y_t2.tolist())
                r1 = pearsonr(results_single["prediction"], results_single["target1"])[0]
                r2 = pearsonr(results_single["prediction"], results_single["target2"])[0]
                out_predictions.append({"target_lang": test_lang, "r": 0.5 * (r1 + r2)})
            df_layer = pd.DataFrame(out_predictions)
            layerwise_dict[n] = df_layer
            print(f"ACROSS-SPLIT-MULTITRAIN {model_prefix} | fROI={froi} | layer={n} | mean r={df_layer['r'].mean():.3f}", flush=True)
        save(layerwise_dict, outpath)

###############################################################################

parser = argparse.ArgumentParser(description="choose which analyses to run")
parser.add_argument(
    "--mode",
    choices=["within", "across", "across-random", "RH", "MD", "native-within", "native-across", "within-split", "across-split"],
    required=True,
    help=("within  --> monolingual + circshift only"
          "across  --> multilingual only (no circshift)"
          "across-random  --> multilingual circshift only"
          "RH      --> right-hemisphere analyses only"
          "MD      --> MD-network analyses only"
          "native-within  --> fROIs found with native contrast"),
)
MODE = parser.parse_args().mode
print(f">>> STARTING JOB in mode: {MODE}", flush=True)

# load fMRI data
with open("data/dict_fROI", 'rb') as handle:
    d = pickle.load(handle)
print(f">>> Loaded fMRI data", flush=True)

all_langs = ['Afrikaans', 'Dutch', 'Farsi', 'French', 'Lithuanian', 'Marathi', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese']
all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}

xglm_langs  = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

#############################
# Non-shuffled (monol only) #
#############################

if MODE == "within":
    print("Processing - WITHIN mode", flush = True)
    xglm_small  = monolingual_encoding(xglm_langs, "xglm_small", 24, d)
    xglm_med    = monolingual_encoding(xglm_langs, "xglm_med", 24, d)
    xglm_large  = monolingual_encoding(xglm_langs, "xglm_large", 48, d)
    xglm_xl     = monolingual_encoding(xglm_langs, "xglm_xl", 48, d)
    mbert       = monolingual_encoding(all_codes, "bert_base", 12, d)
    distilmbert = monolingual_encoding(all_codes, "distilmbert", 6, d)
    xlmr_base   = monolingual_encoding(all_codes, "xlmr_base", 12, d)
    xlmr_large  = monolingual_encoding(all_codes, "xlmr_large", 24, d)
    mt5_small   = monolingual_encoding(all_codes, "mt5_small", 8, d)
    mt5_base    = monolingual_encoding(all_codes, "mt5_base", 12, d)
    mt5_large   = monolingual_encoding(all_codes, "mt5_large", 24, d)
    mdeberta      = monolingual_encoding(all_codes, "mdeberta", 12, d)
    xlm_align     = monolingual_encoding(all_codes, "xlm_align", 12, d)
    infoxlm       = monolingual_encoding(all_codes, "infoxlm_base", 12, d)
    infoxlm_large = monolingual_encoding(all_codes, "infoxlm_large", 24, d)
    multiminilm   = monolingual_encoding(all_codes, "multiminilm", 12, d)
    nllb_d_600m   = monolingual_encoding(all_codes, "nllb200_distilled_600M", 12, d)
    nllb_d_1b     = monolingual_encoding(all_codes, "nllb200_distilled_1B", 24, d)
    nllb_1b       = monolingual_encoding(all_codes, "nllb200_1B", 24, d) 
    mgpt          = monolingual_encoding(mgpt_langs, "mgpt", 24, d)

    # random
    xglm_small  = monolingual_encoding_circshift(xglm_langs, "xglm_small", 24, d)
    xglm_med    = monolingual_encoding_circshift(xglm_langs, "xglm_med", 24, d)
    xglm_large  = monolingual_encoding_circshift(xglm_langs, "xglm_large", 48, d)
    xglm_xl     = monolingual_encoding_circshift(xglm_langs, "xglm_xl", 48, d)
    mbert       = monolingual_encoding_circshift(all_codes, "bert_base", 12, d)
    distilmbert = monolingual_encoding_circshift(all_codes, "distilmbert", 6, d)
    xlmr_base   = monolingual_encoding_circshift(all_codes, "xlmr_base", 12, d)
    xlmr_large  = monolingual_encoding_circshift(all_codes, "xlmr_large", 24, d)
    mt5_small   = monolingual_encoding_circshift(all_codes, "mt5_small", 8, d)
    mt5_base    = monolingual_encoding_circshift(all_codes, "mt5_base", 12, d)
    mt5_large   = monolingual_encoding_circshift(all_codes, "mt5_large", 24, d)
    mdeberta      = monolingual_encoding_circshift(all_codes, "mdeberta", 12, d)
    xlm_align     = monolingual_encoding_circshift(all_codes, "xlm_align", 12, d)
    infoxlm       = monolingual_encoding_circshift(all_codes, "infoxlm_base", 12, d)
    infoxlm_large = monolingual_encoding_circshift(all_codes, "infoxlm_large", 24, d)
    multiminilm   = monolingual_encoding_circshift(all_codes, "multiminilm", 12, d)
    nllb_d_600m   = monolingual_encoding_circshift(all_codes, "nllb200_distilled_600M", 12, d)
    nllb_d_1b     = monolingual_encoding_circshift(all_codes, "nllb200_distilled_1B", 24, d)
    nllb_1b       = monolingual_encoding_circshift(all_codes, "nllb200_1B", 24, d) 
    mgpt          = monolingual_encoding_circshift(mgpt_langs, "mgpt", 24, d)

##################
# MODEL TRANSFER #
##################

elif MODE == "across":
    print("Processing - ACROSS mode", flush = True)
    xglm_small_multi  = multilingual_encoding(xglm_langs, "xglm_small", 24, d)
    xglm_med_multi    = multilingual_encoding(xglm_langs, "xglm_med", 24, d)
    xglm_large_multi  = multilingual_encoding(xglm_langs, "xglm_large", 48, d)
    xglm_xl_multi     = multilingual_encoding(xglm_langs, "xglm_xl", 48, d)
    mbert_multi       = multilingual_encoding(all_codes, "bert_base", 12, d)
    distilmbert_multi = multilingual_encoding(all_codes, "distilmbert", 6, d)
    xlmr_base_multi   = multilingual_encoding(all_codes, "xlmr_base", 12, d)
    xlmr_large_multi  = multilingual_encoding(all_codes, "xlmr_large", 24, d)
    mt5_small_multi   = multilingual_encoding(all_codes, "mt5_small", 8, d)
    mt5_base_multi    = multilingual_encoding(all_codes, "mt5_base", 12, d)
    mt5_large_multi   = multilingual_encoding(all_codes, "mt5_large", 24, d)
    mdeberta_multi      = multilingual_encoding(all_codes, "mdeberta", 12, d)
    xlm_align_multi     = multilingual_encoding(all_codes, "xlm_align", 12, d)
    infoxlm_base_multi  = multilingual_encoding(all_codes, "infoxlm_base", 12, d)
    infoxlm_large_multi = multilingual_encoding(all_codes, "infoxlm_large", 24, d)
    multiminilm_multi   = multilingual_encoding(all_codes, "multiminilm", 12, d)
    nllb_d_600m_multi   = multilingual_encoding(all_codes, "nllb200_distilled_600M", 12, d)
    nllb_d_1b_multi     = multilingual_encoding(all_codes, "nllb200_distilled_1B", 24, d)
    nllb_1b_multi       = multilingual_encoding(all_codes, "nllb200_1B", 24, d)
    mgpt_multi          = multilingual_encoding(mgpt_langs, "mgpt", 24, d)

    # multilingual, train in N-1 langs, generalize to left-out lang
    print("\n\nTraining in N-1 languages (across, standard)", flush=True)
    xglm_small_multi  = multilingual_encoding_multitrain(xglm_langs, "xglm_small", 24, d)
    xglm_med_multi    = multilingual_encoding_multitrain(xglm_langs, "xglm_med", 24, d)
    xglm_large_multi  = multilingual_encoding_multitrain(xglm_langs, "xglm_large", 48, d)
    xglm_xl_multi     = multilingual_encoding_multitrain(xglm_langs, "xglm_xl", 48, d)
    mbert_multi       = multilingual_encoding_multitrain(all_codes, "bert_base", 12, d)
    distilmbert_multi = multilingual_encoding_multitrain(all_codes, "distilmbert", 6, d)
    xlmr_base_multi   = multilingual_encoding_multitrain(all_codes, "xlmr_base", 12, d)
    xlmr_large_multi  = multilingual_encoding_multitrain(all_codes, "xlmr_large", 24, d)
    mt5_small_multi   = multilingual_encoding_multitrain(all_codes, "mt5_small", 8, d)
    mt5_base_multi    = multilingual_encoding_multitrain(all_codes, "mt5_base", 12, d)
    mt5_large_multi   = multilingual_encoding_multitrain(all_codes, "mt5_large", 24, d)
    mdeberta_multi      = multilingual_encoding_multitrain(all_codes, "mdeberta", 12, d)
    xlm_align_multi     = multilingual_encoding_multitrain(all_codes, "xlm_align", 12, d)
    infoxlm_base_multi  = multilingual_encoding_multitrain(all_codes, "infoxlm_base", 12, d)
    infoxlm_large_multi = multilingual_encoding_multitrain(all_codes, "infoxlm_large", 24, d)
    multiminilm_multi   = multilingual_encoding_multitrain(all_codes, "multiminilm", 12, d)
    nllb_d_600m_multi   = multilingual_encoding_multitrain(all_codes, "nllb200_distilled_600M", 12, d)
    nllb_d_1b_multi     = multilingual_encoding_multitrain(all_codes, "nllb200_distilled_1B", 24, d)
    nllb_1b_multi       = multilingual_encoding_multitrain(all_codes, "nllb200_1B", 24, d)
    mgpt_multi          = multilingual_encoding_multitrain(mgpt_langs, "mgpt", 24, d)

elif MODE == "across-random":
    print("Processing - ACROSS mode (random)", flush = True)
    # random
    xglm_small_multi  = multilingual_encoding_circshift(xglm_langs, "xglm_small", 24, d)
    xglm_med_multi    = multilingual_encoding_circshift(xglm_langs, "xglm_med", 24, d)
    xglm_large_multi  = multilingual_encoding_circshift(xglm_langs, "xglm_large", 48, d)
    xglm_xl_multi     = multilingual_encoding_circshift(xglm_langs, "xglm_xl", 48, d)
    mbert_multi       = multilingual_encoding_circshift(all_codes, "bert_base", 12, d)
    distilmbert_multi = multilingual_encoding_circshift(all_codes, "distilmbert", 6, d)
    xlmr_base_multi   = multilingual_encoding_circshift(all_codes, "xlmr_base", 12, d)
    xlmr_large_multi  = multilingual_encoding_circshift(all_codes, "xlmr_large", 24, d)
    mt5_small_multi   = multilingual_encoding_circshift(all_codes, "mt5_small", 8, d)
    mt5_base_multi    = multilingual_encoding_circshift(all_codes, "mt5_base", 12, d)
    mt5_large_multi   = multilingual_encoding_circshift(all_codes, "mt5_large", 24, d)
    mdeberta_multi      = multilingual_encoding_circshift(all_codes, "mdeberta", 12, d)
    xlm_align_multi     = multilingual_encoding_circshift(all_codes, "xlm_align", 12, d)
    infoxlm_base_multi  = multilingual_encoding_circshift(all_codes, "infoxlm_base", 12, d)
    infoxlm_large_multi = multilingual_encoding_circshift(all_codes, "infoxlm_large", 24, d)
    multiminilm_multi   = multilingual_encoding_circshift(all_codes, "multiminilm", 12, d)
    nllb_d_600m_multi   = multilingual_encoding_circshift(all_codes, "nllb200_distilled_600M", 12, d)
    nllb_d_1b_multi     = multilingual_encoding_circshift(all_codes, "nllb200_distilled_1B", 24, d)
    nllb_1b_multi       = multilingual_encoding_circshift(all_codes, "nllb200_1B", 24, d)
    mgpt_multi          = multilingual_encoding_circshift(mgpt_langs, "mgpt", 24, d)

    # random
    print("\n\nTraining in N-1 languages (across, random)", flush=True)
    xglm_small_multi  = multilingual_encoding_multitrain_circshift(xglm_langs, "xglm_small", 24, d)
    xglm_med_multi    = multilingual_encoding_multitrain_circshift(xglm_langs, "xglm_med", 24, d)
    xglm_large_multi  = multilingual_encoding_multitrain_circshift(xglm_langs, "xglm_large", 48, d)
    xglm_xl_multi     = multilingual_encoding_multitrain_circshift(xglm_langs, "xglm_xl", 48, d)
    mbert_multi       = multilingual_encoding_multitrain_circshift(all_codes, "bert_base", 12, d)
    distilmbert_multi = multilingual_encoding_multitrain_circshift(all_codes, "distilmbert", 6, d)
    xlmr_base_multi   = multilingual_encoding_multitrain_circshift(all_codes, "xlmr_base", 12, d)
    xlmr_large_multi  = multilingual_encoding_multitrain_circshift(all_codes, "xlmr_large", 24, d)
    mt5_small_multi   = multilingual_encoding_multitrain_circshift(all_codes, "mt5_small", 8, d)
    mt5_base_multi    = multilingual_encoding_multitrain_circshift(all_codes, "mt5_base", 12, d)
    mt5_large_multi   = multilingual_encoding_multitrain_circshift(all_codes, "mt5_large", 24, d)
    mdeberta_multi      = multilingual_encoding_multitrain_circshift(all_codes, "mdeberta", 12, d)
    xlm_align_multi     = multilingual_encoding_multitrain_circshift(all_codes, "xlm_align", 12, d)
    infoxlm_base_multi  = multilingual_encoding_multitrain_circshift(all_codes, "infoxlm_base", 12, d)
    infoxlm_large_multi = multilingual_encoding_multitrain_circshift(all_codes, "infoxlm_large", 24, d)
    multiminilm_multi   = multilingual_encoding_multitrain_circshift(all_codes, "multiminilm", 12, d)
    nllb_d_600m_multi   = multilingual_encoding_multitrain_circshift(all_codes, "nllb200_distilled_600M", 12, d)
    nllb_d_1b_multi     = multilingual_encoding_multitrain_circshift(all_codes, "nllb200_distilled_1B", 24, d)
    nllb_1b_multi       = multilingual_encoding_multitrain_circshift(all_codes, "nllb200_1B", 24, d)
    mgpt_multi          = multilingual_encoding_multitrain_circshift(mgpt_langs, "mgpt", 24, d)


###############################################################################

####################
# Right hemisphere #
####################

elif MODE == "RH":
    print("Processing - RH mode", flush = True)
    with open("data/dict_fROI_rh", 'rb') as handle:
        d_rh = pickle.load(handle)

    # sequential split
    xglm_small    = monolingual_encoding(xglm_langs, "xglm_small", 24, d_rh, prefix = "rh_")
    xglm_med      = monolingual_encoding(xglm_langs, "xglm_med", 24, d_rh, prefix = "rh_")
    xglm_large    = monolingual_encoding(xglm_langs, "xglm_large", 48, d_rh, prefix = "rh_")
    xglm_xl       = monolingual_encoding(xglm_langs, "xglm_xl", 48, d_rh, prefix = "rh_")
    mbert         = monolingual_encoding(all_codes, "bert_base", 12, d_rh, prefix = "rh_")
    distilmbert   = monolingual_encoding(all_codes, "distilmbert", 6, d_rh, prefix = "rh_")
    xlmr_base     = monolingual_encoding(all_codes, "xlmr_base", 12, d_rh, prefix = "rh_")
    xlmr_large    = monolingual_encoding(all_codes, "xlmr_large", 24, d_rh, prefix = "rh_")
    mt5_small     = monolingual_encoding(all_codes, "mt5_small", 8, d_rh, prefix = "rh_")
    mt5_base      = monolingual_encoding(all_codes, "mt5_base", 12, d_rh, prefix = "rh_")
    mt5_large     = monolingual_encoding(all_codes, "mt5_large", 24, d_rh, prefix = "rh_")
    mdeberta      = monolingual_encoding(all_codes, "mdeberta", 12, d_rh, prefix = "rh_")
    xlm_align     = monolingual_encoding(all_codes, "xlm_align", 12, d_rh, prefix = "rh_")
    infoxlm       = monolingual_encoding(all_codes, "infoxlm_base", 12, d_rh, prefix = "rh_")
    infoxlm_large = monolingual_encoding(all_codes, "infoxlm_large", 24, d_rh, prefix = "rh_")
    multiminilm   = monolingual_encoding(all_codes, "multiminilm", 12, d_rh, prefix = "rh_")
    nllb_d_600m   = monolingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_rh, prefix = "rh_")
    nllb_d_1b     = monolingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_rh, prefix = "rh_")
    nllb_1b       = monolingual_encoding(all_codes, "nllb200_1B", 24, d_rh, prefix = "rh_") 
    mgpt          = monolingual_encoding(mgpt_langs, "mgpt", 24, d_rh, prefix = "rh_")


    # multilingual
    xglm_small_multi    = multilingual_encoding(xglm_langs, "xglm_small", 24, d_rh, prefix = "rh_")
    xglm_med_multi      = multilingual_encoding(xglm_langs, "xglm_med", 24, d_rh, prefix = "rh_")
    xglm_large_multi    = multilingual_encoding(xglm_langs, "xglm_large", 48, d_rh, prefix = "rh_")
    xglm_xl_multi       = multilingual_encoding(xglm_langs, "xglm_xl", 48, d_rh, prefix = "rh_")
    mbert_multi         = multilingual_encoding(all_codes, "bert_base", 12, d_rh, prefix = "rh_")
    distilmbert_multi   = multilingual_encoding(all_codes, "distilmbert", 6, d_rh, prefix = "rh_")
    xlmr_base_multi     = multilingual_encoding(all_codes, "xlmr_base", 12, d_rh, prefix = "rh_")
    xlmr_large_multi    = multilingual_encoding(all_codes, "xlmr_large", 24, d_rh, prefix = "rh_")
    mt5_small_multi     = multilingual_encoding(all_codes, "mt5_small", 8, d_rh, prefix = "rh_")
    mt5_base_multi      = multilingual_encoding(all_codes, "mt5_base", 12, d_rh, prefix = "rh_")
    mt5_large_multi     = multilingual_encoding(all_codes, "mt5_large", 24, d_rh, prefix = "rh_")
    mdeberta_multi      = multilingual_encoding(all_codes, "mdeberta", 12, d_rh, prefix = "rh_")
    xlm_align_multi     = multilingual_encoding(all_codes, "xlm_align", 12, d_rh, prefix = "rh_")
    infoxlm_base_multi  = multilingual_encoding(all_codes, "infoxlm_base", 12, d_rh, prefix = "rh_")
    infoxlm_large_multi = multilingual_encoding(all_codes, "infoxlm_large", 24, d_rh, prefix = "rh_")
    multiminilm_multi   = multilingual_encoding(all_codes, "multiminilm", 12, d_rh, prefix = "rh_")
    nllb_d_600m_multi   = multilingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_rh, prefix = "rh_")
    nllb_d_1b_multi     = multilingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_rh, prefix = "rh_")
    nllb_1b_multi       = multilingual_encoding(all_codes, "nllb200_1B", 24, d_rh, prefix = "rh_")
    mgpt_multi          = multilingual_encoding(mgpt_langs, "mgpt", 24, d_rh, prefix = "rh_")

    # multilingual, train on N-1, test on left out
    print("\n\nTraining in N-1 languages (across, RH)", flush=True)
    xglm_small_multi    = multilingual_encoding_multitrain(xglm_langs, "xglm_small", 24, d_rh, prefix = "rh_")
    xglm_med_multi      = multilingual_encoding_multitrain(xglm_langs, "xglm_med", 24, d_rh, prefix = "rh_")
    xglm_large_multi    = multilingual_encoding_multitrain(xglm_langs, "xglm_large", 48, d_rh, prefix = "rh_")
    xglm_xl_multi       = multilingual_encoding_multitrain(xglm_langs, "xglm_xl", 48, d_rh, prefix = "rh_")
    mbert_multi         = multilingual_encoding_multitrain(all_codes, "bert_base", 12, d_rh, prefix = "rh_")
    distilmbert_multi   = multilingual_encoding_multitrain(all_codes, "distilmbert", 6, d_rh, prefix = "rh_")
    xlmr_base_multi     = multilingual_encoding_multitrain(all_codes, "xlmr_base", 12, d_rh, prefix = "rh_")
    xlmr_large_multi    = multilingual_encoding_multitrain(all_codes, "xlmr_large", 24, d_rh, prefix = "rh_")
    mt5_small_multi     = multilingual_encoding_multitrain(all_codes, "mt5_small", 8, d_rh, prefix = "rh_")
    mt5_base_multi      = multilingual_encoding_multitrain(all_codes, "mt5_base", 12, d_rh, prefix = "rh_")
    mt5_large_multi     = multilingual_encoding_multitrain(all_codes, "mt5_large", 24, d_rh, prefix = "rh_")
    mdeberta_multi      = multilingual_encoding_multitrain(all_codes, "mdeberta", 12, d_rh, prefix = "rh_")
    xlm_align_multi     = multilingual_encoding_multitrain(all_codes, "xlm_align", 12, d_rh, prefix = "rh_")
    infoxlm_base_multi  = multilingual_encoding_multitrain(all_codes, "infoxlm_base", 12, d_rh, prefix = "rh_")
    infoxlm_large_multi = multilingual_encoding_multitrain(all_codes, "infoxlm_large", 24, d_rh, prefix = "rh_")
    multiminilm_multi   = multilingual_encoding_multitrain(all_codes, "multiminilm", 12, d_rh, prefix = "rh_")
    nllb_d_600m_multi   = multilingual_encoding_multitrain(all_codes, "nllb200_distilled_600M", 12, d_rh, prefix = "rh_")
    nllb_d_1b_multi     = multilingual_encoding_multitrain(all_codes, "nllb200_distilled_1B", 24, d_rh, prefix = "rh_")
    nllb_1b_multi       = multilingual_encoding_multitrain(all_codes, "nllb200_1B", 24, d_rh, prefix = "rh_")
    mgpt_multi          = multilingual_encoding_multitrain(mgpt_langs, "mgpt", 24, d_rh, prefix = "rh_")

##############
# MD NETWORK #
##############

elif MODE == "MD":
    print("Processing - MD mode", flush = True)
    with open("data/dict_fROI_md", 'rb') as handle:
        d_md = pickle.load(handle)

    xglm_small  = monolingual_encoding(xglm_langs, "xglm_small", 24, d_md, prefix = "md_")
    xglm_med    = monolingual_encoding(xglm_langs, "xglm_med", 24, d_md, prefix = "md_")
    xglm_large  = monolingual_encoding(xglm_langs, "xglm_large", 48, d_md, prefix = "md_")
    xglm_xl     = monolingual_encoding(xglm_langs, "xglm_xl", 48, d_md, prefix = "md_")
    mbert       = monolingual_encoding(all_codes, "bert_base", 12, d_md, prefix = "md_")
    distilmbert = monolingual_encoding(all_codes, "distilmbert", 6, d_md, prefix = "md_")
    xlmr_base   = monolingual_encoding(all_codes, "xlmr_base", 12, d_md, prefix = "md_")
    xlmr_large  = monolingual_encoding(all_codes, "xlmr_large", 24, d_md, prefix = "md_")
    mt5_small   = monolingual_encoding(all_codes, "mt5_small", 8, d_md, prefix = "md_")
    mt5_base    = monolingual_encoding(all_codes, "mt5_base", 12, d_md, prefix = "md_")
    mt5_large   = monolingual_encoding(all_codes, "mt5_large", 24, d_md, prefix = "md_")
    mdeberta      = monolingual_encoding(all_codes, "mdeberta", 12, d_md, prefix = "md_")
    xlm_align     = monolingual_encoding(all_codes, "xlm_align", 12, d_md, prefix = "md_")
    infoxlm       = monolingual_encoding(all_codes, "infoxlm_base", 12, d_md, prefix = "md_")
    infoxlm_large = monolingual_encoding(all_codes, "infoxlm_large", 24, d_md, prefix = "md_")
    multiminilm   = monolingual_encoding(all_codes, "multiminilm", 12, d_md, prefix = "md_")
    nllb_d_600m   = monolingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_md, prefix = "md_")
    nllb_d_1b     = monolingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_md, prefix = "md_")
    nllb_1b       = monolingual_encoding(all_codes, "nllb200_1B", 24, d_md, prefix = "md_") 
    mgpt          = monolingual_encoding(mgpt_langs, "mgpt", 24, d_md, prefix = "md_")

    # multilingual
    xglm_small_multi    = multilingual_encoding(xglm_langs, "xglm_small", 24, d_md, prefix = "md_")
    xglm_med_multi      = multilingual_encoding(xglm_langs, "xglm_med", 24, d_md, prefix = "md_")
    xglm_large_multi    = multilingual_encoding(xglm_langs, "xglm_large", 48, d_md, prefix = "md_")
    xglm_xl_multi       = multilingual_encoding(xglm_langs, "xglm_xl", 48, d_md, prefix = "md_")
    mbert_multi         = multilingual_encoding(all_codes, "bert_base", 12, d_md, prefix = "md_")
    distilmbert_multi   = multilingual_encoding(all_codes, "distilmbert", 6, d_md, prefix = "md_")
    xlmr_base_multi     = multilingual_encoding(all_codes, "xlmr_base", 12, d_md, prefix = "md_")
    xlmr_large_multi    = multilingual_encoding(all_codes, "xlmr_large", 24, d_md, prefix = "md_")
    mt5_small_multi     = multilingual_encoding(all_codes, "mt5_small", 8, d_md, prefix = "md_")
    mt5_base_multi      = multilingual_encoding(all_codes, "mt5_base", 12, d_md, prefix = "md_")
    mt5_large_multi     = multilingual_encoding(all_codes, "mt5_large", 24, d_md, prefix = "md_")
    mdeberta_multi      = multilingual_encoding(all_codes, "mdeberta", 12, d_md, prefix = "md_")
    xlm_align_multi     = multilingual_encoding(all_codes, "xlm_align", 12, d_md, prefix = "md_")
    infoxlm_base_multi  = multilingual_encoding(all_codes, "infoxlm_base", 12, d_md, prefix = "md_")
    infoxlm_large_multi = multilingual_encoding(all_codes, "infoxlm_large", 24, d_md, prefix = "md_")
    multiminilm_multi   = multilingual_encoding(all_codes, "multiminilm", 12, d_md, prefix = "md_")
    nllb_d_600m_multi   = multilingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_md, prefix = "md_")
    nllb_d_1b_multi     = multilingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_md, prefix = "md_")
    nllb_1b_multi       = multilingual_encoding(all_codes, "nllb200_1B", 24, d_md, prefix = "md_")
    mgpt_multi          = multilingual_encoding(mgpt_langs, "mgpt", 24, d_md, prefix = "md_")

    # multilingual, train on N-1
    print("\n\nTraining in N-1 languages (across, MD)", flush=True)
    xglm_small_multi    = multilingual_encoding_multitrain(xglm_langs, "xglm_small", 24, d_md, prefix = "md_")
    xglm_med_multi      = multilingual_encoding_multitrain(xglm_langs, "xglm_med", 24, d_md, prefix = "md_")
    xglm_large_multi    = multilingual_encoding_multitrain(xglm_langs, "xglm_large", 48, d_md, prefix = "md_")
    xglm_xl_multi       = multilingual_encoding_multitrain(xglm_langs, "xglm_xl", 48, d_md, prefix = "md_")
    mbert_multi         = multilingual_encoding_multitrain(all_codes, "bert_base", 12, d_md, prefix = "md_")
    distilmbert_multi   = multilingual_encoding_multitrain(all_codes, "distilmbert", 6, d_md, prefix = "md_")
    xlmr_base_multi     = multilingual_encoding_multitrain(all_codes, "xlmr_base", 12, d_md, prefix = "md_")
    xlmr_large_multi    = multilingual_encoding_multitrain(all_codes, "xlmr_large", 24, d_md, prefix = "md_")
    mt5_small_multi     = multilingual_encoding_multitrain(all_codes, "mt5_small", 8, d_md, prefix = "md_")
    mt5_base_multi      = multilingual_encoding_multitrain(all_codes, "mt5_base", 12, d_md, prefix = "md_")
    mt5_large_multi     = multilingual_encoding_multitrain(all_codes, "mt5_large", 24, d_md, prefix = "md_")
    mdeberta_multi      = multilingual_encoding_multitrain(all_codes, "mdeberta", 12, d_md, prefix = "md_")
    xlm_align_multi     = multilingual_encoding_multitrain(all_codes, "xlm_align", 12, d_md, prefix = "md_")
    infoxlm_base_multi  = multilingual_encoding_multitrain(all_codes, "infoxlm_base", 12, d_md, prefix = "md_")
    infoxlm_large_multi = multilingual_encoding_multitrain(all_codes, "infoxlm_large", 24, d_md, prefix = "md_")
    multiminilm_multi   = multilingual_encoding_multitrain(all_codes, "multiminilm", 12, d_md, prefix = "md_")
    nllb_d_600m_multi   = multilingual_encoding_multitrain(all_codes, "nllb200_distilled_600M", 12, d_md, prefix = "md_")
    nllb_d_1b_multi     = multilingual_encoding_multitrain(all_codes, "nllb200_distilled_1B", 24, d_md, prefix = "md_")
    nllb_1b_multi       = multilingual_encoding_multitrain(all_codes, "nllb200_1B", 24, d_md, prefix = "md_")
    mgpt_multi          = multilingual_encoding_multitrain(mgpt_langs, "mgpt", 24, d_md, prefix = "md_")

####################
# native localizer #
####################

elif MODE == "native-within":
    print("Processing - native mode (within)", flush = True)
    with open("data/dict_fROI_native", 'rb') as handle:
        d_native = pickle.load(handle)

    # sequential split
    xglm_small    = monolingual_encoding(xglm_langs, "xglm_small", 24, d_native, prefix = "native_")
    xglm_med      = monolingual_encoding(xglm_langs, "xglm_med", 24, d_native, prefix = "native_")
    xglm_large    = monolingual_encoding(xglm_langs, "xglm_large", 48, d_native, prefix = "native_")
    xglm_xl       = monolingual_encoding(xglm_langs, "xglm_xl", 48, d_native, prefix = "native_")
    mbert         = monolingual_encoding(all_codes, "bert_base", 12, d_native, prefix = "native_")
    distilmbert   = monolingual_encoding(all_codes, "distilmbert", 6, d_native, prefix = "native_")
    xlmr_base     = monolingual_encoding(all_codes, "xlmr_base", 12, d_native, prefix = "native_")
    xlmr_large    = monolingual_encoding(all_codes, "xlmr_large", 24, d_native, prefix = "native_")
    mt5_small     = monolingual_encoding(all_codes, "mt5_small", 8, d_native, prefix = "native_")
    mt5_base      = monolingual_encoding(all_codes, "mt5_base", 12, d_native, prefix = "native_")
    mt5_large     = monolingual_encoding(all_codes, "mt5_large", 24, d_native, prefix = "native_")
    mdeberta      = monolingual_encoding(all_codes, "mdeberta", 12, d_native, prefix = "native_")
    xlm_align     = monolingual_encoding(all_codes, "xlm_align", 12, d_native, prefix = "native_")
    infoxlm       = monolingual_encoding(all_codes, "infoxlm_base", 12, d_native, prefix = "native_")
    infoxlm_large = monolingual_encoding(all_codes, "infoxlm_large", 24, d_native, prefix = "native_")
    multiminilm   = monolingual_encoding(all_codes, "multiminilm", 12, d_native, prefix = "native_")
    nllb_d_600m   = monolingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_native, prefix = "native_")
    nllb_d_1b     = monolingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_native, prefix = "native_")
    nllb_1b       = monolingual_encoding(all_codes, "nllb200_1B", 24, d_native, prefix = "native_") 
    mgpt          = monolingual_encoding(mgpt_langs, "mgpt", 24, d_native, prefix = "native_")

elif MODE == "native-across":
    print("Processing - native mode (across)", flush = True)
    with open("data/dict_fROI_native", 'rb') as handle:
        d_native = pickle.load(handle)
    # multilingual
    xglm_small_multi    = multilingual_encoding(xglm_langs, "xglm_small", 24, d_native, prefix = "native_")
    xglm_med_multi      = multilingual_encoding(xglm_langs, "xglm_med", 24, d_native, prefix = "native_")
    xglm_large_multi    = multilingual_encoding(xglm_langs, "xglm_large", 48, d_native, prefix = "native_")
    xglm_xl_multi       = multilingual_encoding(xglm_langs, "xglm_xl", 48, d_native, prefix = "native_")
    mbert_multi         = multilingual_encoding(all_codes, "bert_base", 12, d_native, prefix = "native_")
    distilmbert_multi   = multilingual_encoding(all_codes, "distilmbert", 6, d_native, prefix = "native_")
    xlmr_base_multi     = multilingual_encoding(all_codes, "xlmr_base", 12, d_native, prefix = "native_")
    xlmr_large_multi    = multilingual_encoding(all_codes, "xlmr_large", 24, d_native, prefix = "native_")
    mt5_small_multi     = multilingual_encoding(all_codes, "mt5_small", 8, d_native, prefix = "native_")
    mt5_base_multi      = multilingual_encoding(all_codes, "mt5_base", 12, d_native, prefix = "native_")
    mt5_large_multi     = multilingual_encoding(all_codes, "mt5_large", 24, d_native, prefix = "native_")
    mdeberta_multi      = multilingual_encoding(all_codes, "mdeberta", 12, d_native, prefix = "native_")
    xlm_align_multi     = multilingual_encoding(all_codes, "xlm_align", 12, d_native, prefix = "native_")
    infoxlm_base_multi  = multilingual_encoding(all_codes, "infoxlm_base", 12, d_native, prefix = "native_")
    infoxlm_large_multi = multilingual_encoding(all_codes, "infoxlm_large", 24, d_native, prefix = "native_")
    multiminilm_multi   = multilingual_encoding(all_codes, "multiminilm", 12, d_native, prefix = "native_")
    nllb_d_600m_multi   = multilingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_native, prefix = "native_")
    nllb_d_1b_multi     = multilingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_native, prefix = "native_")
    nllb_1b_multi       = multilingual_encoding(all_codes, "nllb200_1B", 24, d_native, prefix = "native_")
    mgpt_multi          = multilingual_encoding(mgpt_langs, "mgpt", 24, d_native, prefix = "native_")

    # multilingual, train on N-1
    print("\n\nTraining in N-1 languages (across, native localizer)", flush=True)
    xglm_small_multi    = multilingual_encoding_multitrain(xglm_langs, "xglm_small", 24, d_native, prefix = "native_")
    xglm_med_multi      = multilingual_encoding_multitrain(xglm_langs, "xglm_med", 24, d_native, prefix = "native_")
    xglm_large_multi    = multilingual_encoding_multitrain(xglm_langs, "xglm_large", 48, d_native, prefix = "native_")
    xglm_xl_multi       = multilingual_encoding_multitrain(xglm_langs, "xglm_xl", 48, d_native, prefix = "native_")
    mbert_multi         = multilingual_encoding_multitrain(all_codes, "bert_base", 12, d_native, prefix = "native_")
    distilmbert_multi   = multilingual_encoding_multitrain(all_codes, "distilmbert", 6, d_native, prefix = "native_")
    xlmr_base_multi     = multilingual_encoding_multitrain(all_codes, "xlmr_base", 12, d_native, prefix = "native_")
    xlmr_large_multi    = multilingual_encoding_multitrain(all_codes, "xlmr_large", 24, d_native, prefix = "native_")
    mt5_small_multi     = multilingual_encoding_multitrain(all_codes, "mt5_small", 8, d_native, prefix = "native_")
    mt5_base_multi      = multilingual_encoding_multitrain(all_codes, "mt5_base", 12, d_native, prefix = "native_")
    mt5_large_multi     = multilingual_encoding_multitrain(all_codes, "mt5_large", 24, d_native, prefix = "native_")
    mdeberta_multi      = multilingual_encoding_multitrain(all_codes, "mdeberta", 12, d_native, prefix = "native_")
    xlm_align_multi     = multilingual_encoding_multitrain(all_codes, "xlm_align", 12, d_native, prefix = "native_")
    infoxlm_base_multi  = multilingual_encoding_multitrain(all_codes, "infoxlm_base", 12, d_native, prefix = "native_")
    infoxlm_large_multi = multilingual_encoding_multitrain(all_codes, "infoxlm_large", 24, d_native, prefix = "native_")
    multiminilm_multi   = multilingual_encoding_multitrain(all_codes, "multiminilm", 12, d_native, prefix = "native_")
    nllb_d_600m_multi   = multilingual_encoding_multitrain(all_codes, "nllb200_distilled_600M", 12, d_native, prefix = "native_")
    nllb_d_1b_multi     = multilingual_encoding_multitrain(all_codes, "nllb200_distilled_1B", 24, d_native, prefix = "native_")
    nllb_1b_multi       = multilingual_encoding_multitrain(all_codes, "nllb200_1B", 24, d_native, prefix = "native_")
    mgpt_multi          = multilingual_encoding_multitrain(mgpt_langs, "mgpt", 24, d_native, prefix = "native_")

elif MODE == "within-split":
    print("Processing - WITHIN (SPLIT-CONTEXT) mode", flush=True)
    monolingual_encoding_split(xglm_langs,  "xglm_large", 48, d)
    monolingual_encoding_split(xglm_langs,  "xglm_small", 24, d)
    monolingual_encoding_split(xglm_langs,  "xglm_med",   24, d)
    monolingual_encoding_split(xglm_langs,  "xglm_xl",    48, d)
    monolingual_encoding_split(all_codes,   "bert_base",  12, d)
    monolingual_encoding_split(all_codes,   "distilmbert",6,  d)
    monolingual_encoding_split(all_codes,   "xlmr_base",  12, d)
    monolingual_encoding_split(all_codes,   "xlmr_large", 24, d)
    monolingual_encoding_split(all_codes,   "mt5_small",  8,  d)
    monolingual_encoding_split(all_codes,   "mt5_base",   12, d)
    monolingual_encoding_split(all_codes,   "mt5_large",  24, d)
    monolingual_encoding_split(all_codes,   "mdeberta",   12, d)
    monolingual_encoding_split(all_codes,   "xlm_align",  12, d)
    monolingual_encoding_split(all_codes,   "infoxlm_base", 12, d)
    monolingual_encoding_split(all_codes,   "infoxlm_large",24, d)
    monolingual_encoding_split(all_codes,   "multiminilm", 12, d)
    monolingual_encoding_split(all_codes,   "nllb200_distilled_600M", 12, d)
    monolingual_encoding_split(all_codes,   "nllb200_distilled_1B",   24, d)
    monolingual_encoding_split(all_codes,   "nllb200_1B",             24, d)
    monolingual_encoding_split(mgpt_langs,  "mgpt",       24, d)

elif MODE == "across-split":
    print("Processing - ACROSS (SPLIT-CONTEXT, train on N-1 langs) mode", flush=True)
    multilingual_encoding_multitrain_split(xglm_langs, "xglm_small", 24, d)
    multilingual_encoding_multitrain_split(xglm_langs, "xglm_med",   24, d)
    multilingual_encoding_multitrain_split(xglm_langs, "xglm_large", 48, d)
    multilingual_encoding_multitrain_split(xglm_langs, "xglm_xl",    48, d)
    multilingual_encoding_multitrain_split(all_codes,  "bert_base",  12, d)
    multilingual_encoding_multitrain_split(all_codes,  "distilmbert",6,  d)
    multilingual_encoding_multitrain_split(all_codes,  "xlmr_base",  12, d)
    multilingual_encoding_multitrain_split(all_codes,  "xlmr_large", 24, d)
    multilingual_encoding_multitrain_split(all_codes,  "mt5_small",  8,  d)
    multilingual_encoding_multitrain_split(all_codes,  "mt5_base",   12, d)
    multilingual_encoding_multitrain_split(all_codes,  "mt5_large",  24, d)
    multilingual_encoding_multitrain_split(all_codes,  "mdeberta",   12, d)
    multilingual_encoding_multitrain_split(all_codes,  "xlm_align",  12, d)
    multilingual_encoding_multitrain_split(all_codes,  "infoxlm_base", 12, d)
    multilingual_encoding_multitrain_split(all_codes,  "infoxlm_large",24, d)
    multilingual_encoding_multitrain_split(all_codes,  "multiminilm", 12, d)
    multilingual_encoding_multitrain_split(all_codes,  "nllb200_distilled_600M", 12, d)
    multilingual_encoding_multitrain_split(all_codes,  "nllb200_distilled_1B",   24, d)
    multilingual_encoding_multitrain_split(all_codes,  "nllb200_1B",             24, d)
    multilingual_encoding_multitrain_split(mgpt_langs, "mgpt",       24, d)