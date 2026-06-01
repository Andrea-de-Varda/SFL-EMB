!huggingface-cli login --token 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

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
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import torch
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from scipy.stats import zscore
import random
import re
import time

from transformers import file_utils
print(file_utils.default_cache_path)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT)
chdir(dirName)

modelname = "TheBloke/Llama-2-7B-GGML"

tokenizer = LlamaTokenizer.from_pretrained(modelname)
model = LlamaForCausalLM.from_pretrained(modelname, low_cpu_mem_usage=True)

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model(input_ids)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]



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