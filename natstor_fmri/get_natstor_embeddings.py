#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.ma as ma
import pandas as pd
import re
import torch
from transformers import GPT2Tokenizer, GPT2Model, OpenAIGPTModel, OpenAIGPTTokenizer, AutoTokenizer, AutoModelForCausalLM
from os import chdir
import os
import pickle
import math
import sys
from scipy.stats import pearsonr
from tqdm import tqdm


# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
path = str(PROJECT_ROOT / "natstor_fmri")
chdir(path)

parent_dir = os.path.abspath(os.path.join(path, '..'))
sys.path.append(parent_dir)

from utils import get_surprisal_tokens, loadmat1, test_model_Ridge, test_model_GAM, base_plot, significance_testing, decompose_sfl, get_sem

def save(file, name):
    with open("embeddings/"+name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def check_tok(lang, tokenizer):
    df = pd.read_csv("transcribed/"+lang+".csv")
    # df = df[df["end"] <= 260]
    text = df["text"].str.cat(sep=' ')
    
    a = [tokenizer.tokenize(w) for w in list(text)]
    for the_a, the_b in zip(a, list(text)):
        if the_a:
            if len(the_a) == 1:
                if the_a[0][1] != the_b:
                    print(the_a, the_b)
            else:
                if the_a[1] != the_b:
                    print(the_a, the_b)
                    
def check_special_tokens(tokenizer):
    # encode and then decode to see the effect of special tokens (to remove for mapping models)
    sample_text = "This is a sample text."
    encoded_input = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded_input = tokenizer.decode(encoded_input)
    print(decoded_input)
    tokenized_input = tokenizer.tokenize(sample_text)
    delta = len(encoded_input) - len(tokenized_input)
    print(f"Delta = {delta}")
            
def tok_maker(a, sep, toker, cased = True):
    out = []
    for w in a:
        tok = toker.tokenize(w)
        tok[0] = re.sub(sep, "", tok[0])
        out.append(tok)
    return out

def get_word_embeddings(sentence, tokenizer, model):
    input_ids = tokenizer.encode(sentence, is_split_into_words=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_embeddings = [hidden_state[0].cpu().numpy() for hidden_state in hidden_states]
    return layer_embeddings

def get_embeddings_tokens(tokens, sep, tokenizer, model, cased=True, emb_start = 0, emb_end = None):
    layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model)
    layer_embs = [emb[emb_start:emb_end] for emb in layer_embs] # discard embeddings for special chars
    toks = tok_maker(tokens, sep, tokenizer, cased)
    len_emb = layer_embs[0].shape[0]
    len_toks = len([t for tok in toks for t in tok])
    if len_emb != len_toks:
        raise ValueError(f"The number of embeddings does not correspond to the number of tokens ({len_emb} embeddings for {len_toks} tokens). Check the special characters added by the tokenizer.")
    out_dict = {}
    for idx, embs in enumerate(layer_embs):
        out = []
        theindex = 0
        for index, word in enumerate(toks):
            if len(word) == 1:
                emb = embs[theindex]
                theindex += 1
                out.append(emb)
            else:
                emb = embs[theindex:theindex+len(word)]
                theindex += len(word)
                out.append(np.mean(emb, axis=0))
        out = np.vstack(out)
        out_dict[idx] = out
    return out_dict

def embeddings(lang, sep, tokenizer, model, saveto, emb_start = 0, emb_end = None):
    if os.path.isfile(f"embeddings/{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        # df = df[df["end"] <= 260]
        text = df["text"].str.cat(sep=' ')
        embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end)
        save(embs, saveto+"_"+lang)
        print(f"done {lang}")
        return embs
    
# if context size < len(passage), need to split the text in chunks
# if I just split and concatenate, the information about previous chunks is lost, and the embeddings would
# not represent correctly the context. Iteratively sliding the window is too computationally intensive
# >>>> dividing text into overlapping chunks

def embeddings_chunked(lang, sep, tokenizer, model, saveto, n_splits = 2, overlap = 50, emb_start = 0, emb_end = None):
    if os.path.isfile(f"embeddings/{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        # df = df[df["end"] <= 260]
        text = df["text"].str.cat(sep=' ')
        
        # splitting in overlapping chunks
        words = text.split()
        chunk_size = math.ceil(len(words) / n_splits)

        n_splits = n_splits + math.ceil(overlap*n_splits / chunk_size) # add splits needed (lost because of overlap size)
        chunks = []
        for split in range(n_splits):
            if split == 0:
                start = split * chunk_size
                end = start + chunk_size
                chunks.append(" ".join(words[start : end]))
            elif split == n_splits -1:
                start = split * chunk_size - overlap * split
                end = start + chunk_size
                chunks.append(" ".join(words[start : ]))
            else:
                start = split * chunk_size - overlap * split
                end = start + chunk_size
                chunks.append(" ".join(words[start : end]))
            
        all_embs = []
        for chunk in chunks:
            embs = get_embeddings_tokens(tokens = chunk.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end)
            all_embs.append(embs)
        
        final_embeddings = {k: [] for k in all_embs[0].keys()}
        
        for k in all_embs[0].keys():
            for i in range(len(all_embs) - 1):
                if i == 0:
                    non_overlap = all_embs[i][k]
                    final_embeddings[k].extend(non_overlap)
                else:
                    non_overlap = all_embs[i][k][overlap:]
                    final_embeddings[k].extend(non_overlap)
            
            # Add the non-overlapping part of the last chunk
            final_embeddings[k].extend(all_embs[-1][k][overlap:])
        
        for k in final_embeddings.keys():
            final_embeddings[k] = np.array(final_embeddings[k])
            
        save(final_embeddings, saveto+"_"+lang)
        print(f"done {lang}")
        return final_embeddings

# IMPORTANT - remember to add to paper that we split the text into 2 chunks (3 in the case of the original GPT)

# gpt
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTModel.from_pretrained('openai-gpt')

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "</w>", tokenizer = tokenizer, n_splits = 3, model = model, saveto = "gpt")

# gpt2-small
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
model = GPT2Model.from_pretrained('gpt2')

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_small")

# gpt2-med
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', add_prefix_space=True)
model = GPT2Model.from_pretrained('gpt2-medium')
check_special_tokens(tokenizer) # </s> This is a sample text. - 1

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_medium")

# gpt2-large
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', add_prefix_space=True)
model = GPT2Model.from_pretrained('gpt2-large')
check_special_tokens(tokenizer) # </s> This is a sample text. - 1

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_large")
    
# gpt2-xl
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', add_prefix_space=True)
model = GPT2Model.from_pretrained('gpt2-xl')
check_special_tokens(tokenizer) # </s> This is a sample text. - 1

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_xl")
    
###########
# get SFL #
###########

# frequency
freq = pd.read_excel(PROJECT_ROOT / "main_data/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}
minfreq = min(f.values())

def get_f(word):
    word = re.sub('[\.\,\:\-\?\!\)\(\"]', "", word)
    try:
        fr = f[word]
    except KeyError:
        fr = minfreq
    return fr


def surprisal_chunked(lang, sep, tokenizer, model, saveto, n_splits = 2, overlap = 50, cased = True):
    if os.path.isfile(f"embeddings/sfl_{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        # df = df[df["end"] <= 260]
        text = df["text"].str.cat(sep=' ')
        
        # splitting in overlapping chunks
        words = text.split()
        chunk_size = math.ceil(len(words) / n_splits)

        n_splits = n_splits + math.ceil(overlap*n_splits / chunk_size) # add splits needed (lost because of overlap size)
        chunks = []
        for split in range(n_splits):
            if split == 0:
                start = split * chunk_size
                end = start + chunk_size
                chunks.append(" ".join(words[start : end]))
            elif split == n_splits -1:
                start = split * chunk_size - overlap * split
                end = start + chunk_size
                chunks.append(" ".join(words[start : ]))
            else:
                start = split * chunk_size - overlap * split
                end = start + chunk_size
                chunks.append(" ".join(words[start : end]))
            
        all_s = []
        for chunk in chunks:
            s = get_surprisal_tokens(tokens = chunk.split(), sep = sep, toker = tokenizer, model = model, cased = cased)
            all_s.append(s)
        
        final_s = []
        for i in range(len(all_s) - 1):
            if i == 0:
                non_overlap = all_s[i]
                final_s.extend(non_overlap)
            else:
                non_overlap = all_s[i][overlap-1:] # minus one because get_surprisal_tokens() does not return values for the first token
                final_s.extend(non_overlap)
            
            # Add the non-overlapping part of the last chunk
        final_s.extend(all_s[-1][overlap-1:])
        
        length = [len(w) for w in text.split()[1:]]
        frequency = [get_f(w) for w in text.split()[1:]]
        
        r, _ = pearsonr(final_s, frequency)
        print(f"Correlation between S and F -- r = {round(r, 4)}")

        sfl = np.stack([np.array(final_s), np.array(frequency), np.array(length)], axis=1)

        save(sfl, "sfl_"+saveto+"_"+lang)
        print(f"done {lang}")
        return sfl

model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

for story_n in range(1,11):
    story = surprisal_chunked(str(story_n), sep = "</w>", tokenizer = tokenizer, n_splits = 3, model = model, saveto = "gpt", cased = False)
    
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

for story_n in range(1,11):
    story = surprisal_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_small")
    
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

for story_n in range(1,11):
    story = surprisal_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_medium")
    
model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

for story_n in range(1,11):
    story = surprisal_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_large")
    
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

for story_n in range(1,11):
    story = surprisal_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_xl")


###########
# get SEM #
###########

def sem_natstor(lang, saveto):
    df = pd.read_csv("transcribed/"+lang+".csv")
    # df = df[df["end"] <= 260]
    text = df["text"].str.cat(sep=' ')
    sem = get_sem(text.split()[1:])
    save(sem, "sem_"+saveto+lang)
    print(f"done {lang}")
    return sem

for story_n in range(1,11):
    story = sem_natstor(str(story_n), saveto = "")