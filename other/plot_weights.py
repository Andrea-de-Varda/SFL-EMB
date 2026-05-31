import pickle
import pandas as pd
import matplotlib.pyplot as plt
from os import chdir
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import pearsonr

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

def plot_weights_nospill(filename, title, subtitle, ymin = -.01, ymax=.015, addtofile = ""):
    plt.rcParams.update({'font.size': 13})
    color1 = '#9b90c2'  # Pastel purple
    color2 = '#87d3f8'  # Pastel blue
    color3 = '#f8b195'  # Pastel peach
    with open(addtofile+'results/'+filename+'_coefs_avg', 'rb') as handle:
        the_m = pickle.load(handle)
    with open(addtofile+'results/'+filename+'_coefs_std', 'rb') as handle:
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
    
    
def plot_weights_nospill_withEMB(filename, title, subtitle, ymin = -.01, ymax=.015, addtofile = ""):
    plt.rcParams.update({'font.size': 13})
    color1 = '#9b90c2'  # Pastel purple
    color2 = '#87d3f8'  # Pastel blue
    color3 = '#f8b195'  # Pastel peach
    color4 = '#FAC8CD'
    with open(addtofile+'results/'+filename+'_coefs_avg', 'rb') as handle:
        the_m = pickle.load(handle)
    with open(addtofile+'results/'+filename+'_coefs_std', 'rb') as handle:
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
    
    
    
from utils import plot_weights_spill, plot_weights_spill_withEMB

plot_weights_nospill("provo_med_base", "Provo, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill_withEMB("provo_med_all", "Provo, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25)

plot_weights_nospill("meco_med_base", "Meco, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill_withEMB("provo_med_all", "Meco, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25)

plot_weights_nospill("zuco_med_base", "ZuCo, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25, addtofile="ZuCo/")
plot_weights_nospill_withEMB("zuco_med_all", "ZuCo, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25, addtofile="ZuCo/")

plot_weights_nospill("UCL_fpass_med_base", "UCL, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25)
plot_weights_nospill_withEMB("UCL_fpass_med_all", "UCL, FPass", "GPT-2$_{355M}$", ymin = -5.5, ymax = 25)

plot_weights_nospill("michaelov_med_base", "Michaelov", "GPT-2$_{355M}$", ymin = -.2, ymax = .15)
plot_weights_nospill_withEMB("michaelov_med_all", "Michaelov", "GPT-2$_{355M}$", ymin = -.2, ymax = .15)

plot_weights_nospill("UCL_n400_med_base", "UCL, N400", "GPT-2$_{355M}$", ymin = -.1, ymax = .1)
plot_weights_nospill_withEMB("UCL_n400_med_all", "UCL, N400", "GPT-2$_{355M}$", ymin = -.1, ymax = .1)

plot_weights_nospill("UCL_p600_med_base", "UCL, P600", "GPT-2$_{355M}$", ymin = -.1, ymax = .1)
plot_weights_nospill_withEMB("UCL_p600_med_all", "UCL, P600", "GPT-2$_{355M}$", ymin = -.1, ymax = .1)

plot_weights_spill("UCL_spr_med_base", "UCL, SPR", ymin = -1, ymax = 5, modelname="GPT-2$_{355M}$")
plot_weights_spill_withEMB("UCL_spr_med_base", "UCL, SPR", ymin = -1, ymax = 5, modelname="GPT-2$_{355M}$")




def load_weights(filename, addtofile = ""):
    with open(addtofile+'results/'+filename+'_coefs_avg', 'rb') as handle:
        the_m = pickle.load(handle)
    the_m1 = list(the_m)
    return the_m1


provo = load_weights("provo_med_all")[3:]

meco = load_weights("meco_med_all")[3:]

zuco = load_weights("zuco_med_all", addtofile="ZuCo/")[3:]

ucl = load_weights("UCL_fpass_med_all")[3:]

mich = load_weights("michaelov_med_all")[3:]




from itertools import combinations

for c1, c2 in combinations([provo, meco, zuco, ucl, mich], 2):
    r, _ = pearsonr(c1, c2)
    print(r)


###################################################################################

def load_weights(name, thedir=""):
    with open(thedir+'results/'+name+'_coefs_avg', 'rb') as handle:
        weights = pickle.load(handle)
    return weights

hp = load_weights("hp_all_med_all", "fMRI/")[:-12] # SFL in the end
hp_split = np.mean(np.array_split(hp, 4), axis=0) # averaging for spillover
pereira = load_weights("pereira_all_med_all", "pereira/")[3:] # SFL in the beginning
michaelov = load_weights("michaelov_med_all")[3:] # beginning
ucl_fpass = load_weights("UCL_fpass_med_all")[3:] # beginning
ucl_n400 = load_weights("UCL_n400_med_all")[3:] # beginning
ucl_p600 = load_weights("UCL_p600_med_all")[3:] # beginning
ucl_spr = load_weights("UCL_spr_med_all")[9:] # beginning (9)
ucl_spr = np.mean(np.array_split(ucl_spr, 3), axis=0) # averaging for spillover
meco = load_weights("meco_med_all")[3:] # beginning
provo = load_weights("provo_med_all")[3:] # beginning
zuco = load_weights("zuco_med_all", "ZuCo/")[3:] # beginning
maze = load_weights("maze_med_all")[3:] # beginning
natstor = load_weights("natstor_med_all")[9:] # beginning
natstor = np.mean(np.array_split(natstor, 3), axis=0) # averaging for spillover
brown = load_weights("brown_med_all")[9:] # beginning
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
mask_non_sig = p_values > 0.01  
#mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
corr[mask] = np.nan

df = pd.DataFrame(corr, columns=names, index=names)
#np.fill_diagonal(df.values, np.nan)
#df = df.iloc[1:, :-1]
plt.figure(figsize=(7, 7), dpi=150)
sns.heatmap(df, annot=True, fmt=".2f", cmap='jet', vmin=-.55, vmax=.55, square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 10})
#plt.title('Correlation between probe weights', fontsize=26, pad=20)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16)
plt.annotate('Behav.', xy=(-0.42, 0.692), xytext=(-0.57, 0.692),
            fontsize=19, ha='center', va='center', xycoords='axes fraction', 
            bbox=dict(boxstyle='square', fc='0.8'),
            arrowprops=dict(arrowstyle='-[, widthB=4.9, lengthB=.5', lw=2.0))
plt.annotate('Neural', xy=(-.42, 0.199), xytext=(-.57, 0.199),
            fontsize=19, ha='center', va='center', xycoords='axes fraction', 
            bbox=dict(boxstyle='square', fc='0.8'),
            arrowprops=dict(arrowstyle='-[, widthB=2.9, lengthB=.5', lw=2.0))
plt.show()


