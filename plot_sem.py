import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from math import sqrt

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import plot_sem, plot_weights_sem, significance_testing

def n_to_asterisk(sig_list):
    out_ast = []
    for p in sig_list:
        if p > .025:
            out_ast.append("")
        # elif .025 < p < .05:
        #     out_ast.append(".")
        else:
            out_ast.append("*")
        if p > .975:
            print("other tail")
    return out_ast

def get_significance(dfname):
    model_sizes = ["gpt", "small", "med", "large", "xl"]
    out_sig = []
    for size in model_sizes:
        p_act = significance_testing(dfname+"_"+size+"_sem", dfname+"_"+size)
        p_base = significance_testing(dfname+"_"+size+"_base_sem", dfname+"_"+size+"_base")
        p_all = significance_testing(dfname+"_"+size+"_all_sem", dfname+"_"+size+"_all")
        out_sig.append([p_act, p_base, p_all])
    return n_to_asterisk([p for thesig in out_sig for p in thesig])

get_significance("brown")
get_significance("provo")

# significance_testing("brown_large_all_sem", "brown_large_all")
# significance_testing("UCL_n400_gpt_base_sem", "UCL_n400_gpt_base")


# def theload_1(name1, name2, thedir=""):
#     with open(thedir+'results/'+name1+'_predictions', 'rb') as handle:
#         a1 = pickle.load(handle)
#     with open(thedir+'results/'+name2+'_predictions', 'rb') as handle:
#         a2 = pickle.load(handle)
#     for a, b in zip(a1, a2):
#         print(a > b)
#     print(a1)
#     print("\n")
#     print(a2)

def plot_sem(df_results, col1, col2, col3, col4, col5, col6, data_name, title="", legend=False, ylim1=0., ylim2=1, subtract = False, asteriskshift = 0):
    print(data_name)
    sig = get_significance(data_name)
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
        bars_tot = np.array(bars1)+bars2
    
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
        bars_tot = np.array(bars1)+np.array(bars2)
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
    yerr = [y/sqrt(10) for y in yerr]
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
    for thetext, thecoords in zip(sig, zip([1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], bars_tot)):
        plt.annotate(thetext, (thecoords[0]-0.3, thecoords[1]+.02+asteriskshift))
    plt.title(title, fontsize=16)
    plt.show()


################
# EYE-TRACKING #
################

merged = pd.read_csv("SEM/merged_provo.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_fpass", "base_fpass", "all_fpass", "act_fpass_sem", "base_fpass_sem", "all_fpass_sem", "provo", title="Provo", subtract=True)
np.mean(merged["base_fpass_sem"]-merged["base_fpass"])

dirName = "/home/dev/Documents/PhD/MIT/project/ZuCo"
chdir(dirName)
merged = pd.read_csv("SEM/merged_zuco.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_fpass", "base_fpass", "all_fpass", "act_fpass_sem", "base_fpass_sem", "all_fpass_sem", "zuco", title="ZuCo", subtract=True)
dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)
np.mean(merged["base_fpass_sem"]-merged["base_fpass"])

merged = pd.read_csv("SEM/merged_meco.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_fpass", "base_fpass", "all_fpass", "act_fpass_sem", "base_fpass_sem", "all_fpass_sem", "meco", title="Meco-en", subtract=True)
np.mean(merged["base_fpass_sem"]-merged["base_fpass"])

merged = pd.read_csv("SEM/merged_UCL.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_fpass", "base_fpass", "all_fpass", "act_fpass_sem", "base_fpass_sem", "all_fpass_sem", "UCL_fpass", title="UCL$_{ET}$", subtract=True)
np.mean(merged["base_fpass_sem"]-merged["base_fpass"])

######################
# SELF-PACED READING #
######################

merged = pd.read_csv("SEM/merged_UCL.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_spr", "base_spr", "all_spr", "act_spr_sem", "base_spr_sem", "all_spr_sem", "UCL_spr", title="UCL$_{SPR}$", subtract=True)
np.mean(merged["base_spr_sem"]-merged["base_spr"])

merged = pd.read_csv("SEM/merged_natstor_brown.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_natstor", "base_natstor", "all_natstor", "act_natstor_sem", "base_natstor_sem", "all_natstor_sem", "natstor", title="NatStor, SPR", subtract=True)
np.mean(merged["base_natstor_sem"]-merged["base_natstor"])

merged = pd.read_csv("SEM/merged_natstor_brown.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_brown", "base_brown", "all_brown", "act_brown_sem", "base_brown_sem", "all_brown_sem", "brown", title="Brown, SPR", subtract=True)
np.mean(merged["base_brown_sem"]-merged["base_brown"])

########
# MAZE #
########

merged = pd.read_csv("SEM/merged_maze.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", "maze", title="NatStor, maze", subtract=True)
np.mean(merged["base_sem"]-merged["base"])

#######
# ERP #
#######

merged = pd.read_csv("SEM/merged_UCL.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_n400", "base_n400", "all_n400", "act_n400_sem", "base_n400_sem", "all_n400_sem", "UCL_n400", title="UCL$_{N400}$", subtract=True)
np.mean(merged["base_n400_sem"]-merged["base_n400"])

merged = pd.read_csv("SEM/merged_UCL.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act_p600", "base_p600", "all_p600", "act_p600_sem", "base_p600_sem", "all_p600_sem", "UCL_p600", title="UCL$_{P600}$", subtract=True)
np.mean(merged["base_p600_sem"]-merged["base_p600"])

merged = pd.read_csv("SEM/merged_michaelov.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
#plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", "michaelov", title="Michaelov", subtract=True)
np.mean(merged["base_sem"]-merged["base"])

########
# fMRI #
########

dirName = "/home/dev/Documents/PhD/MIT/project/fMRI"
chdir(dirName)
roin =  ["all", "PostTemp", "AntTemp", "AngularG", "IFG", "MFG", "IFGorb", "pCingulate"]
for roi in roin:
    merged = pd.read_csv("results/"+roi+"_results_sem.csv")
    merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
    plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", "hp_"+roi, title=roi, subtract=True, ylim1 = -.1)
dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

# Plotting ALL for paper
dirName = "/home/dev/Documents/PhD/MIT/project/fMRI"
chdir(dirName)
roi = "all"
merged = pd.read_csv("results/"+roi+"_results_sem.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", "hp_"+roi, title="Harry Potter", subtract=True, ylim1 = 0)
np.mean(merged["base_sem"]-merged["base"]) # 0.007336600
dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)


dirName = "/home/dev/Documents/PhD/MIT/project/pereira"
chdir(dirName)
roin = ['LMidPostTemp', 'LPostTemp', 'LMidAntTemp', 'LIFG', 'LAntTemp', 'LIFGorb', 'LAngG', 'LMFG', 'all']
for roi in roin:
    if roi == "all":
        merged = pd.read_csv("results/"+roi+"_results_sem.csv")
        merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
        plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", "pereira_"+roi, title=roi, subtract=False, ylim1 = -.1, asteriskshift=.02)
    else:
        merged = pd.read_csv("results/"+roi+"_results_sem.csv")
        merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
        plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", "pereira_"+roi, title=roi[1:], subtract=False, ylim1 = -.1, asteriskshift=.02)
dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

# plotting ALL for paper
dirName = "/home/dev/Documents/PhD/MIT/project/pereira"
chdir(dirName)
merged = pd.read_csv("results/"+roi+"_results_sem.csv")
merged.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
plot_sem(merged, "act", "base", "all", "act_sem", "base_sem", "all_sem", "pereira_"+roi, title="Pereira", subtract=False, ylim1 = 0, asteriskshift=.02)
np.mean(merged["base_sem"]) # not subtracting since already subtracted before (see probe_Pereira.py)
dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

merged["base_sem"]




def theload_1(name1, name2, thedir=""):
    with open(thedir+'results/'+name1+'_predictions', 'rb') as handle:
        a1 = pickle.load(handle)
    with open(thedir+'results/'+name2+'_predictions', 'rb') as handle:
        a2 = pickle.load(handle)
    for a, b in zip(a1, a2):
        print(round(a, 4), round(b, 4), b > a)
    # print(a1)
    # print("\n")
    # print(a2)
    
theload_1("hp_AntTemp_small_base", "hp_AntTemp_small_base_sem")
    
theload_1("UCL_n400_small_base", "UCL_n400_small_base_sem")
# theload_1("UCL_fpass_small_base", "UCL_fpass_small_base_sem")

# theload_1("provo_small_base", "provo_small_base_sem")

# significance_testing("UCL_n400_gpt_base_sem", "UCL_n400_gpt_base")
# significance_testing("UCL_fpass_small_base_sem", "UCL_fpass_small_base")


dirName = "/home/dev/Documents/PhD/MIT/project/pereira"
chdir(dirName)
plot_weights_sem("pereira_all_small_base_sem", "Pereira, all", "weights")
dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)


plot_weights_sem("UCL_n400_small_base_sem", "UCL, N400", "weights", ymin = -0.1, ymax=.1)
plot_weights_sem("michaelov_small_base_sem", "Michaelov, N400", "weights", ymin = -0.15, ymax=.1)


