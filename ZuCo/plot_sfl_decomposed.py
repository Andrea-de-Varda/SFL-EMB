import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import chdir, path
from math import sqrt

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT)
chdir(dirName)

blue1 = '#87CEFA'
blue2 = '#4169E1'
blue3 = '#0000CD'

df= pd.read_csv("pereira/results/pereira_all_gpt_base_sfl_decomposed.csv")
df = df.drop(columns=df.columns[0])
df.index = ["m", "sd"]

def plot(df, title="", ylim1=-0.1, ylim2=0.9):
    # fpass
    plt.rcParams.update({
        "text.usetex": False})
    fig = plt.figure(figsize = (5/1.2, 3.5/1.2), dpi=300)
    plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
    plt.minorticks_on()
    plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
    plt.bar([1,2,3,4.5,5.5,6.5,8], 
            df.loc["m", :],
            color = [blue1, blue1, blue1, blue2, blue2, blue2, blue3], 
            width=0.9,
            yerr=df.loc["sd", :]/sqrt(10),capsize=2,
            tick_label=df.columns, zorder=5
            )
    plt.ylabel("r")
    plt.xticks(fontsize=11)#, rotation=35, ha='right', rotation_mode='anchor')
    plt.ylim(ylim1, ylim2)
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
    plt.title(title)
    plt.show()
    
plot(df)
    
names = ["Provo", "Meco-en", "ZuCo", "UCL$_{ET}$",  "UCL$_{SPR}$", "Brown", "NatStor, SPR", "NatStor, maze",  "UCL$_{N400}$",  "UCL$_{P600}$", "Michaelov", "Harry Potter", "Pereira"]

fnames = ["results/provo_med_base_sfl_decomposed.csv", "results/meco_med_base_sfl_decomposed.csv", "ZuCo/results/zuco_med_base_sfl_decomposed.csv", "results/UCL_fpass_med_base_sfl_decomposed.csv", "results/UCL_spr_med_base_sfl_decomposed.csv", "results/brown_med_base_sfl_decomposed.csv", "results/natstor_med_base_sfl_decomposed.csv", "results/maze_med_base_sfl_decomposed.csv", "results/UCL_n400_med_base_sfl_decomposed.csv", "results/UCL_p600_med_base_sfl_decomposed.csv", "results/michaelov_med_base_sfl_decomposed.csv", "fMRI/results/hp_all_med_base_sfl_decomposed.csv", "pereira/results/pereira_all_med_base_sfl_decomposed.csv"]

len(fnames)

for fname, name in zip(fnames, names):
    df = pd.read_csv(fname)
    df = df.drop(columns=df.columns[0])
    df.index = ["m", "sd"]
    plot(df, title = name, ylim1 = -.1)
