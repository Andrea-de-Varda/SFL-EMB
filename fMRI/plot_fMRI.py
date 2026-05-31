import numpy as np
import pandas as pd
from os import chdir, path
from scipy.stats import zscore
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.patches as mpatches

chdir("/home/dev/Documents/PhD/MIT/project/fMRI")

results_univ = pd.read_csv("results/fMRI_univariate_linear.csv")

with open('results/coefs_linear_fMRI', 'rb') as handle:
    out_coefs = pickle.load(handle)

plt.rcParams.update({
    "text.usetex": False})


def plt_fmri(theroi, ymin, ymax, title=False):
    fig = plt.figure(figsize = (5/1.2, 3.5/1.2), dpi=300)
    plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
    plt.minorticks_on()
    plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
    plt.bar([1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], 
            [results_univ.loc[theroi, "gpt_m"],   results_univ.loc[theroi, "base_gpt"],   results_univ.loc[theroi, "all_gpt"],
             results_univ.loc[theroi, "small_m"], results_univ.loc[theroi, "base_small"], results_univ.loc[theroi, "all_small"],
             results_univ.loc[theroi, "med_m"],   results_univ.loc[theroi, "base_med"],   results_univ.loc[theroi, "all_med"],
             results_univ.loc[theroi, "large_m"], results_univ.loc[theroi, "base_large"], results_univ.loc[theroi, "all_large"],
             results_univ.loc[theroi, "xl_m"],    results_univ.loc[theroi, "base_xl"],    results_univ.loc[theroi, "all_xl"]],
            color = ['lightblue', 'lightgreen', "indianred"]*5,width=0.9,
            yerr=[results_univ.loc[theroi, "gpt_sd"], results_univ.loc[theroi, "base_gpt_sd"], results_univ.loc[theroi, "all_gpt_sd"],
             results_univ.loc[theroi, "small_sd"],    results_univ.loc[theroi, "base_small_sd"], results_univ.loc[theroi, "all_small_sd"],
             results_univ.loc[theroi, "med_sd"],      results_univ.loc[theroi, "base_med_sd"], results_univ.loc[theroi, "all_med_sd"],
             results_univ.loc[theroi, "large_sd"],    results_univ.loc[theroi, "base_large_sd"], results_univ.loc[theroi, "all_large_sd"],
             results_univ.loc[theroi, "xl_sd"],       results_univ.loc[theroi, "base_xl_sd"], results_univ.loc[theroi, "all_xl_sd"]],capsize=2,
            tick_label=["", "GPT", "", "", "GPT-2$_{124M}$", "", "", "GPT-2$_{355M}$","", "",  "GPT-2$_{774M}$", "", "", "GPT-2$_{1.5B}$", ""]
            )
    plt.ylabel("r")
    plt.xticks(fontsize=11, rotation=35, ha='right', rotation_mode='anchor')
    plt.ylim(ymin, ymax)
    if title:
        plt.title(theroi)
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
    plt.show()

for the_roi in ["all", "PostTemp", "AntTemp", "AngularG", "IFG", "MFG", "IFGorb", "pCingulate"]:
    plt_fmri(the_roi, -0.08, 0.3, title=True)
    
    
#####################################################################################################################

color1 = '#9b90c2'  # Pastel purple
color2 = '#87d3f8'  # Pastel blue
color3 = '#f8b195'  # Pastel peach

plt.rcParams.update({'font.size': 13})

model_dict = {"gpt":"GPT", "small":"GPT-2$_{124M}$", "med":"GPT-2$_{355M}$", "large":"GPT-2$_{774M}$", "xl":"GPT-2$_{1.5B}$"}

def plot_weights(the_roi, the_mod, ymin = -.01, ymax=.015):
    the_m, the_sd = out_coefs[the_roi][the_mod]
    fig1 = plt.figure(dpi = 300)
    ax1 = SubplotHost(fig1, 111)
    fig1.add_subplot(ax1)
    # Some data
    x = [1,2,3,5,6,7,9,10,11,13,14,15]
    y = the_m
    # First X-axis
    ax1.bar(x, y, color = [color1, color2, color3]*4,width=0.9, zorder=4)
    ax1.errorbar(x, y, yerr=the_sd, fmt='none', elinewidth=1, capsize=4, color="black", zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["S", "F", "L"]*4)
    #ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
    ax1.yaxis.set_label_text("Weights")
    ax1.set_xlim(0, 16)
    ax1.set_ylim(ymin, ymax)
    ax1.axhline(y=0, color='black', linestyle='-', zorder=6, linewidth=1)
    # Second X-axis
    ax2 = ax1.twiny()
    offset = 0, -25 # Position of the second axis
    new_axisline = ax2.get_grid_helper().new_fixed_axis
    ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
    ax2.axis["top"].set_visible(False)
    
    ax2.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([0.125, 0.375, 0.625, 0.875]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(["TR$_{i-4}$", "TR$_{i-3}$","TR$_{i-2}$","TR$_{i-1}$"]))
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
    ax3.xaxis.set_minor_formatter(ticker.FixedFormatter([model_dict[the_mod]]))
    ax3.axis["bottom"].minor_ticks.set_ticksize(0)
    ax1.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=3)
    #plt.minorticks_on()
    ax1.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=2)
    plt.title(the_roi)
    plt.show()

for model in ["gpt", "small", "med", "large", "xl"]:
    plot_weights("all", model, -0.018, 0.024)

for model in ["gpt", "small", "med", "large", "xl"]:
    plot_weights("PostTemp", model, -0.018, 0.024)
    
for model in ["gpt", "small", "med", "large", "xl"]:
    plot_weights("AntTemp", model, -0.018, 0.024)
    
plt.figure(dpi=300)
act  = mpatches.Patch(color=color1, label='(S) Surprisal')
bsl  = mpatches.Patch(color=color2, label='(F) Frequency')
all_ = mpatches.Patch(color=color3, label='(L) Length')
plt.legend(handles=[act, bsl, all_], bbox_to_anchor=(0.5, 0.5))