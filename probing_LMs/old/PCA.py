import neurox
import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import neurox.data.loader as data_loader
from sklearn.decomposition import PCA
import joblib

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

def get_layer(vec, layer, dim):
    lower_bound = layer*dim
    upper_bound = (layer+1)*dim
    return list(vec[lower_bound:upper_bound])

def get_act(act_list, layer, dim):
    out = []
    for act_sent in act_list:
        out.extend([get_layer(v, layer, dim) for v in act_sent[1:]])
    return np.array(out)

def load_emb(filename):
    act_gpt2_medium, num_layers_gpt2_medium = data_loader.load_activations(filename, 1024, is_brnn=False)
    med_filt   = get_act(act_gpt2_medium, 24, 1024)
    del act_gpt2_medium
    return med_filt
    
ucl = load_emb("activations_gpt2_medium.json")
provo = load_emb("activations_gpt2_medium_provo.json")
zuco = load_emb("ZuCo/activations_gpt2_medium.json")
meco = load_emb("activations_gpt2_medium_meco.json")
maze = load_emb("activations_gpt2_medium_maze.json")
brown = load_emb("activations_gpt2_medium_brown.json")
natst =  load_emb("activations_gpt2_medium_natstor.json")
mich = load_emb("activations_gpt2_medium_michaelov.json")
pereira = load_emb("pereira/activations_gpt2_medium.json")
hp = load_emb("fMRI/HP_emb/activations_gpt2_medium.json")

all_data = np.concatenate([ucl, provo, zuco, meco, maze, brown, natst, mich, pereira, hp])

pca = PCA(n_components=100)
pca.fit(all_data)
joblib.dump(pca, 'probing_LMs/pca.pkl')

X_pca = pca.transform(all_data)
explained_variance = pca.explained_variance_ratio_
total_variance = np.sum(explained_variance)

print(explained_variance)
print(total_variance) # .99353



from joblib import load
pca = load('probing_LMs/pca.pkl')

