import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import chdir
from math import sqrt
from scipy.stats import pearsonr, spearmanr
from scipy.stats import zscore

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

from utils import significance_testing

res_meco   = pd.read_csv("results_meco.csv")
res_ucl    = pd.read_csv("results/results_UCL.csv")
res_zuco   = pd.read_csv("ZuCo/results_zuco.csv")
res_provo  = pd.read_csv("results_provo.csv")
res_spr   = pd.read_csv("results_natstor_brown_spill.csv")
res_natst = pd.read_csv("results_natstor_sentencewise.csv")
res_maze  = pd.read_csv("results_maze_nospill.csv")
#res_mich = pd.read_csv("results_michaelov.csv")
fmri = pd.read_csv("fMRI/results/all_results.csv")
#pereira = pd.read_csv("pereira/results/all_results.csv")
pereira = pd.read_csv("pereira_1/results/results.csv")
control = pd.read_csv("control_fmri/results/control_results.csv")
natstor_fmri = pd.read_csv("natstor_fmri/results/all_results.csv")

# other fix measurements
res_ucl_ff  = pd.read_csv("results/results_UCL_ffix.csv")
res_provo_ff  = pd.read_csv("results_provo_ff.csv")
res_provo_tf  = pd.read_csv("results_provo_tf.csv")
res_meco_ff   = pd.read_csv("results_meco_ff.csv")
res_zuco_ff   = pd.read_csv("ZuCo/results_zuco_ff.csv")

res_meco["act_fpass"] # GPT
res_meco["base_fpass"] # GPT

res_zuco["act_fpass"]  # gpt
res_zuco["base_fpass"] # small

res_provo["act_fpass"] # small
res_provo["base_fpass"] # GPT

res_ucl["act_fpass"] # small
res_ucl["base_fpass"] # small

res_ucl["act_spr"] # large
res_ucl["base_spr"] # xl

res_natst["act_natstor"] # med
res_natst["base_natstor"] # small

res_spr["act_brown"] # xl
res_spr["base_brown"] # small

res_maze["act"] # med
res_maze["base"] # GPT == large

res_ucl["act_n400"] # med
res_ucl["act_p600"] # xl

# res_mich["act"] # GPT
# res_mich["base"] # GPT

fmri["act"] # gpt
fmri["base"] # gpt

pereira["act"] # XL
pereira["base"] # GPT

###################
# stats for paper #
###################

# ET

np.mean([res_meco["act_fpass"].mean(), res_zuco["act_fpass"].mean(), res_provo["act_fpass"].mean(),  res_ucl["act_fpass"].mean()]) -np.mean([res_meco["base_fpass"].mean(), res_zuco["base_fpass"].mean(), res_provo["base_fpass"].mean(),  res_ucl["base_fpass"].mean()])

np.mean([res_meco["all_fpass"].mean(), res_zuco["all_fpass"].mean(), res_provo["all_fpass"].mean(),  res_ucl["all_fpass"].mean()]) -np.mean([res_meco["base_fpass"].mean(), res_zuco["base_fpass"].mean(), res_provo["base_fpass"].mean(),  res_ucl["base_fpass"].mean()])

# SPR
np.mean([res_spr["act_brown"].mean(), res_natst["act_natstor"].mean(), res_ucl["act_spr"].mean()]) -np.mean([res_spr["base_brown"].mean(), res_natst["base_natstor"].mean(), res_ucl["base_spr"].mean()])
np.mean([res_spr["all_brown"].mean(), res_natst["all_natstor"].mean(), res_ucl["all_spr"].mean()]) -np.mean([res_spr["base_brown"].mean(), res_natst["base_natstor"].mean(), res_ucl["base_spr"].mean()])

res_maze["all"].mean() - res_maze["base"].mean()

pereira["act"].mean() - pereira["base"].mean()
pereira["act"].mean() - pereira["all"].mean()

natstor_fmri["act"].mean() - natstor_fmri["base"].mean()
natstor_fmri["act"].mean() - natstor_fmri["all"].mean()

control["act"].mean() - control["base"].mean()
control["act"].mean() - control["all"].mean()

fmri["act"].mean() - fmri["base"].mean()
fmri["act"].mean() - fmri["all"].mean()


def z(data):
    return zscore(list(data))

def bxplt(n1, n2, n3, n4, n5, title):
    meanpointprops={"marker":'s','markerfacecolor': "black", 'markeredgecolor': "black", "markersize":4}
    plt.figure(dpi=300, figsize=(3.75/2.3, 2.5/2.3))
    labels=["GPT", "GPT2$_{124M}$", "GPT2$_{355M}$", "GPT2$_{774M}$", "GPT2$_{1.5B}$"]
    bplot = plt.boxplot([n1, n2, n3, n4, n5],vert=True, patch_artist=True, labels=labels, showmeans=True, meanprops=meanpointprops, whis = 5)
    plt.ylabel("$z(r)$", labelpad=-5)
    #plt.xlabel("Model")
    plt.xticks(fontsize=10, rotation=35, ha='right', rotation_mode='anchor')
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', "indianred"]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for element in ['medians']:
        plt.setp(bplot[element], color="black")
    #plt.grid(True)
    plt.ylim(-2.1, 2.1)
    plt.title(title)
    plt.show()

# base
res_behav = [z(res_meco["base_fpass"]), z(res_ucl["base_fpass"]), z(res_zuco["base_fpass"]), z(res_provo["base_fpass"]), z(res_ucl["base_spr"]), z(res_natst["base_natstor"]), z(res_spr["base_brown"]), z(res_maze["base"])]
n1 = [res[0] for res in res_behav]
n2 = [res[1] for res in res_behav]
n3 = [res[2] for res in res_behav]
n4 = [res[3] for res in res_behav]
n5 = [res[4] for res in res_behav]
bxplt(n1, n2, n3, n4, n5, "SFL")

# act
res_behav = [z(res_meco["act_fpass"]), z(res_ucl["act_fpass"]), z(res_zuco["act_fpass"]), z(res_provo["act_fpass"]), z(res_ucl["act_spr"]), z(res_natst["act_natstor"]), z(res_spr["act_brown"]), z(res_maze["act"])]
n1 = [res[0] for res in res_behav]
n2 = [res[1] for res in res_behav]
n3 = [res[2] for res in res_behav]
n4 = [res[3] for res in res_behav]
n5 = [res[4] for res in res_behav]
bxplt(n1, n2, n3, n4, n5, "EMB")

# all
res_behav = [z(res_meco["all_fpass"]), z(res_ucl["all_fpass"]), z(res_zuco["all_fpass"]), z(res_provo["all_fpass"]), z(res_ucl["all_spr"]), z(res_natst["all_natstor"]), z(res_spr["all_brown"]), z(res_maze["all"])]
n1 = [res[0] for res in res_behav]
n2 = [res[1] for res in res_behav]
n3 = [res[2] for res in res_behav]
n4 = [res[3] for res in res_behav]
n5 = [res[4] for res in res_behav]
bxplt(n1, n2, n3, n4, n5, "EMB$^\\frown$SFL")
np.mean(n5)

# neuro, base
res_neuro = [z(fmri["base"]), z(pereira["base"]), z(res_ucl["base_n400"]), z(pereira["base"]), z(control["base"]), z(natstor_fmri["base"])]
n1 = [res[0] for res in res_neuro]
n2 = [res[1] for res in res_neuro]
n3 = [res[2] for res in res_neuro]
n4 = [res[3] for res in res_neuro]
n5 = [res[4] for res in res_neuro]
bxplt(n1, n2, n3, n4, n5, "SFL")

# neuro, act
res_neuro = [z(fmri["act"]), z(pereira["act"]), z(res_ucl["act_n400"]), z(pereira["act"]), z(control["act"]), z(natstor_fmri["act"])]
n1 = [res[0] for res in res_neuro]
n2 = [res[1] for res in res_neuro]
n3 = [res[2] for res in res_neuro]
n4 = [res[3] for res in res_neuro]
n5 = [res[4] for res in res_neuro]
bxplt(n1, n2, n3, n4, n5, "EMB")

# neuro, all
res_neuro = [z(fmri["all"]), z(pereira["all"]), z(res_ucl["all_n400"]), z(pereira["all"]), z(control["all"]), z(natstor_fmri["all"])]
n1 = [res[0] for res in res_neuro]
n2 = [res[1] for res in res_neuro]
n3 = [res[2] for res in res_neuro]
n4 = [res[3] for res in res_neuro]
n5 = [res[4] for res in res_neuro]
bxplt(n1, n2, n3, n4, n5, "EMB$^\\frown$SFL")

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.03, barh=.05, fs=None, maxasterix=3):
    """ 
    credit to cheersmate: https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    if data > .025:
        if type(data) is str:
            text = data
        else:
            # * is p < 0.05
            # ** is p < 0.005
            # *** is p < 0.0005
            # etc.
            text = ''
            p = .025
            while data < p:
                text += '*'
                p /= 10.
    
                if maxasterix and len(text) == maxasterix:
                    break
            if len(text) == 0:
                text = 'n.s.'
        lx, ly = center[num1], height[num1]
        rx, ry = center[num2], height[num2]
        if yerr:
            ly += yerr[num1]
            ry += yerr[num2]
        ax_y0, ax_y1 = plt.gca().get_ylim()
        dh *= (ax_y1 - ax_y0)
        barh *= (ax_y1 - ax_y0)
        y = max(ly, ry) + dh
        barx = [lx, lx, rx, rx]
        bary = [y, y+barh, y+barh, y]
        mid = ((lx+rx)/2, y+barh)
        plt.plot(barx, bary, c='black')
        kwargs = dict(ha='center', va='bottom')
        if fs is not None:
            kwargs['fontsize'] = fs
        plt.text(mid[0], mid[1]+0.002, text, **kwargs)
    else:
        pass

def base_plot(df_results, col1, col2, col3, p_column="", title="", legend=False, plot_sig = True, annotate=True, ylim1=0, ylim2=1, barh1= .05, barh2=.05, barh3=.05, dh1 = .03, dh2 = .03, dh3 = .03, dhlist2 = None, dhlist3=None):

    df_results.index = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
    
    heights = [df_results.loc["GPT", col1], df_results.loc["GPT", col2], df_results.loc["GPT", col3],
     df_results.loc["GPT-2$_{124M}$", col1], df_results.loc["GPT-2$_{124M}$", col2], df_results.loc["GPT-2$_{124M}$", col3],
     df_results.loc["GPT-2$_{355M}$", col1], df_results.loc["GPT-2$_{355M}$", col2], df_results.loc["GPT-2$_{355M}$", col3],
     df_results.loc["GPT-2$_{774M}$", col1], df_results.loc["GPT-2$_{774M}$", col2], df_results.loc["GPT-2$_{774M}$", col3],
     df_results.loc["GPT-2$_{1.5B}$", col1], df_results.loc["GPT-2$_{1.5B}$", col2], df_results.loc["GPT-2$_{1.5B}$", col3]]
    
    yerr=[df_results.loc["GPT", col1+"_sd"], df_results.loc["GPT", col2+"_sd"], df_results.loc["GPT", col3+"_sd"],
     df_results.loc["GPT-2$_{124M}$", col1+"_sd"], df_results.loc["GPT-2$_{124M}$", col2+"_sd"], df_results.loc["GPT-2$_{124M}$", col3+"_sd"],
     df_results.loc["GPT-2$_{355M}$", col1+"_sd"], df_results.loc["GPT-2$_{355M}$", col2+"_sd"], df_results.loc["GPT-2$_{355M}$", col3+"_sd"],
     df_results.loc["GPT-2$_{774M}$", col1+"_sd"], df_results.loc["GPT-2$_{774M}$", col2+"_sd"], df_results.loc["GPT-2$_{774M}$", col3+"_sd"],
     df_results.loc["GPT-2$_{1.5B}$", col1+"_sd"], df_results.loc["GPT-2$_{1.5B}$", col2+"_sd"], df_results.loc["GPT-2$_{1.5B}$", col3+"_sd"]]
    yerr = [y/sqrt(10) for y in yerr]
    
    bars = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
    
    plt.rcParams.update({
        "text.usetex": False})
    fig = plt.figure(figsize = (5/1.7, 3.5/1.7), dpi=300)
    plt.grid(axis="y",b=True, which='major', color='#666666', linestyle='-', zorder=0)
    plt.minorticks_on()
    plt.grid(axis="y",b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, zorder=3)
    plt.bar(bars, 
            heights,capsize=2, color = ['lightblue', 'lightgreen', "indianred"]*5,width=0.9,
            tick_label=["", "GPT", "", "", "GPT-2$_{124M}$", "", "", "GPT-2$_{355M}$","", "",  "GPT-2$_{774M}$", "", "", "GPT-2$_{1.5B}$", ""], zorder=5, yerr=yerr
            )
    #####################
    if plot_sig:
        if annotate:
            p_base_emb = list(df_results[p_column+"_base_emb"])
            p_all_emb = list(df_results[p_column+"_all_emb"])
            p_all_base = list(df_results[p_column+"_all_base"])
            
            base_emb_tup = [[0,1], [3,4], [6,7], [9,10], [12,13]]
            all_base_tup = [[1,2], [4,5], [7,8], [10,11], [13,14]]
            all_emb_tup  = [[0,2], [3,5], [6,8], [9,11], [12,14]]
            
            for tup, p in zip(base_emb_tup, p_base_emb):
                barplot_annotate_brackets(tup[0], tup[1], p, bars, heights, yerr=yerr, barh = barh1, dh = dh1)
            for tup, p in zip(all_base_tup, p_all_base):
                barplot_annotate_brackets(tup[0], tup[1], p, bars, heights, yerr=yerr, barh = barh2, dh = dh2)
            for tup, p in zip(all_emb_tup, p_all_emb):
                barplot_annotate_brackets(tup[0], tup[1], p, bars, heights, yerr=yerr, barh = barh3, dh = dh3)
        else:
            p_base_emb = list(df_results[p_column+"_base_emb"])
            p_all_emb = list(df_results[p_column+"_all_emb"])
            p_all_base = list(df_results[p_column+"_all_base"])
            
            base_emb_tup = [[0,1], [3,4], [6,7], [9,10], [12,13]]
            all_base_tup = [[1,2], [4,5], [7,8], [10,11], [13,14]]
            all_emb_tup  = [[0,2], [3,5], [6,8], [9,11], [12,14]]
            
            for tup, p in zip(base_emb_tup, p_base_emb):
                barplot_annotate_brackets(tup[0], tup[1], p, bars, heights, yerr=yerr, barh = barh1, dh = dh1)
            for tup, p, thedh in zip(all_base_tup, p_all_base, dhlist2):
                barplot_annotate_brackets(tup[0], tup[1], p, bars, heights, yerr=yerr, barh = barh2, dh = thedh)
            for tup, p, thedh in zip(all_emb_tup, p_all_emb, dhlist3):
                barplot_annotate_brackets(tup[0], tup[1], p, bars, heights, yerr=yerr, barh = barh3, dh = thedh)
    #####################
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
    #plt.tight_layout()
    plt.show()
    

base_plot(res_natst, "act_natstor", "base_natstor", "all_natstor", "p", title="NatStor, SPR")
base_plot(res_spr, "act_brown", "base_brown", "all_brown", "p_brown", title="Brown")

base_plot(res_zuco, "act_fpass", "base_fpass", "all_fpass", "p_fpass", title="ZuCo")
base_plot(res_provo, "act_fpass", "base_fpass", "all_fpass", "p_fpass", title="Provo", barh2 = .018) # not good
base_plot(res_ucl, "act_fpass", "base_fpass", "all_fpass", "p_fpass", title="UCL$_{ET}$")
base_plot(res_meco, "act_fpass", "base_fpass", "all_fpass", "p_fpass", title="Meco-en")
base_plot(res_ucl, "act_spr", "base_spr", "all_spr", "p_spr", title="UCL$_{SPR}$", dh2 = .22)
# base_plot(res_ucl, "act_p600", "base_p600", "all_p600", "p_p600", title="UCL$_{P600}$")

base_plot(res_maze, "act", "base", "all", "p_maze", title="NatStor$_{Maze}$")

base_plot(res_ucl, "act_n400", "base_n400", "all_n400", "p_n400", title="UCL$_{N400}$")
base_plot(pereira, "act", "base", "all", "p", title="Pereira2018")
# base_plot(res_mich, "act", "base", "all", "p", title="Michaelov", annotate=False, dh1 = .07,dhlist2 = [.03, .03, .4, .36, .35], dhlist3 = [.36, .05, .52, .48, .45])
base_plot(fmri, "act", "base", "all", "p_all", title="Wehbe2014", annotate=False, dh1 = .07,dhlist2 = [.4, .3, .3, .3, .3], dhlist3 = [.52, .42, .44, .43, .43])
base_plot(control, "act", "base", "all", "p", title="Tuckute2024")
base_plot(natstor_fmri, "act", "base", "all", "p", title="NatStor$_{fMRI}$")


###############################################################################
###############################################################################
###############################################################################


# other fix measurements
base_plot(res_ucl_ff, "act_ffix", "base_ffix", "all_ffix", "p_ffix", title="UCL$_{ET}$")
base_plot(res_ucl_ff, "act_totfix", "base_totfix", "all_totfix", "p_totfix", title="UCL$_{ET}$")
base_plot(res_provo_ff, "act_ffix", "base_ffix", "all_ffix", "p_ffix", title="Provo", barh2 = .018)
base_plot(res_provo_tf, "act_totfix", "base_totfix", "all_totfix", "p_totfix", title="Provo", barh2 = .018)
base_plot(res_meco_ff, "act_ffix", "base_ffix", "all_ffix", "p_ffix", title="Meco-en")
base_plot(res_meco_ff, "act_totfix", "base_totfix", "all_totfix", "p_totfix", title="Meco-en")
base_plot(res_zuco_ff, "act_ffix", "base_ffix", "all_ffix", "p_ffix", title="ZuCo")
base_plot(res_zuco_ff, "act_totfix", "base_totfix", "all_totfix", "p_totfix", title="ZuCo")



# ratio
(np.mean(res_provo["base_fpass"])/np.mean(res_provo["act_fpass"])-1)*9
(np.mean(res_meco["base_fpass"])/np.mean(res_meco["act_fpass"])-1)*9
(np.mean(res_zuco["base_fpass"])/np.mean(res_zuco["act_fpass"])-1)*9
(np.mean(res_ucl["base_fpass"])/np.mean(res_ucl["act_fpass"])-1)*9

(np.mean(res_ucl["base_spr"])/np.mean(res_ucl["act_spr"])-1)*9
(np.mean(res_spr["base_brown"])/np.mean(res_spr["act_brown"])-1)*9
(np.mean(res_natst["act_natstor"])/np.mean(res_natst["base_natstor"])-1)*9
(np.mean(res_maze["act"])/np.mean(res_maze["base"])-1)*9

(np.mean(res_ucl["act_n400"])/np.mean(res_ucl["base_n400"])-1)*9
(np.mean(res_ucl["act_p600"])/np.mean(res_ucl["base_p600"])-1)*9
(np.mean(res_mich["act"])/np.mean(res_mich["base"])-1)*9
(np.mean(fmri["act"])/np.mean(fmri["base"])-1)*9
(np.mean(pereira["act"])/np.mean(pereira["base"])-1)*9

# delta
(np.mean(res_provo["base_fpass"])-np.mean(res_provo["act_fpass"]))*20  
(np.mean(res_meco["base_fpass"])-np.mean(res_meco["act_fpass"]))*20
(np.mean(res_zuco["base_fpass"])-np.mean(res_zuco["act_fpass"]))*20
(np.mean(res_ucl["base_fpass"])-np.mean(res_ucl["act_fpass"]))*20

(np.mean(res_ucl["base_spr"])-np.mean(res_ucl["act_spr"]))*20
(np.mean(res_spr["base_brown"])-np.mean(res_spr["act_brown"]))*20
(np.mean(res_natst["act_natstor"])-np.mean(res_natst["base_natstor"]))*20
(np.mean(res_maze["act"])-np.mean(res_maze["base"]))*20

(np.mean(res_ucl["act_n400"])-np.mean(res_ucl["base_n400"]))*20
#(np.mean(res_ucl["act_p600"])-np.mean(res_ucl["base_p600"]))*20
#(np.mean(res_mich["act"])-np.mean(res_mich["base"]))*20
(np.mean(pereira["act"])-np.mean(pereira["base"]))*20
(np.mean(control["act"])-np.mean(control["base"]))*20
(np.mean(natstor_fmri["act"])-np.mean(natstor_fmri["base"]))*20
(np.mean(fmri["act"])-np.mean(fmri["base"]))*20


res_provo.columns
np.mean(res_provo["base_fpass"])-np.mean(res_provo["all_fpass"])+ np.mean(res_meco["base_fpass"])-np.mean(res_meco["all_fpass"])+ np.mean(res_zuco["base_fpass"])-np.mean(res_zuco["all_fpass"])+ np.mean(res_ucl["base_fpass"])-np.mean(res_ucl["all_fpass"])

np.mean(res_ucl["base_spr"])-np.mean(res_ucl["all_spr"])+np.mean(res_spr["base_brown"])-np.mean(res_spr["all_brown"])+np.mean(res_natst["base_natstor"])-np.mean(res_natst["all_natstor"])


np.mean(res_maze["all"])-np.mean(res_maze["base"])


np.mean(res_ucl["act_n400"])-np.mean(res_ucl["all_n400"])
np.mean(res_ucl["act_p600"])-np.mean(res_ucl["all_p600"])
np.mean(res_mich["act"])-np.mean(res_mich["all"])

np.mean(fmri["act"])-np.mean(fmri["all"])
np.mean(pereira["act"])-np.mean(pereira["all"])

spearmanr(pereira["act"], pereira["base"])



# combine p-values
from scipy.stats import norm
from math import sqrt
from scipy import stats
from scipy.stats import chi2


def combine_pvalues(pvalues):
    # Calculate the test statistic
    chi_squared = -2 * np.sum(np.log(pvalues))
    # Find the combined p-value
    combined_pvalue = chi2.sf(chi_squared, 2 * len(pvalues))
    return combined_pvalue

combine_pvalues([res_ucl["p_n400_base_emb"][4], pereira["p_base_emb"][4], fmri["p_all_base_emb"][4], natstor_fmri["p_base_emb"][4], ])

base_plot(res_ucl, "act_spr", "base_spr", "all_spr", "p_spr", title="UCL$_{SPR}$", dh2 = .22)
base_plot(res_ucl, "act_p600", "base_p600", "all_p600", "p_p600", title="UCL$_{P600}$")

base_plot(res_maze, "act", "base", "all", "p_maze", title="NatStor, maze")

base_plot(pereira, "act", "base", "all", "p", title="Pereira")
base_plot(res_mich, "act", "base", "all", "p", title="Michaelov", annotate=False, dh1 = .07,dhlist2 = [.03, .03, .4, .36, .35], dhlist3 = [.36, .05, .52, .48, .45])
base_plot(fmri, "act", "base", "all", "p_all", title="Harry Potter", annotate=False, dh1 = .07,dhlist2 = [.4, .3, .3, .3, .3], dhlist3 = [.52, .42, .44, .43, .43])



#####################
# region-wise plots #
#####################

def sig_test(df_results, roi):
    thebase = ["hp_"+roi+"_gpt_base", "hp_"+roi+"_small_base", "hp_"+roi+"_med_base", "hp_"+roi+"_large_base", "hp_"+roi+"_xl_base"]
    theemb  = ["hp_"+roi+"_gpt", "hp_"+roi+"_small", "hp_"+roi+"_med", "hp_"+roi+"_large", "hp_"+roi+"_xl"]
    theall     = ["hp_"+roi+"_gpt_all", "hp_"+roi+"_small_all", "hp_"+roi+"_med_all", "hp_"+roi+"_large_all", "hp_"+roi+"_xl_all"]
    
    p_all = []
    for base, emb, all_ in zip(thebase, theemb, theall):
        p_base_emb = significance_testing(emb, base, thedir="fMRI/")
        p_all_base = significance_testing(all_, base, thedir="fMRI/")
        p_all_emb  = significance_testing(all_, emb, thedir="fMRI/")
        p_all.append([p_base_emb, p_all_base, p_all_emb])
    p_all = pd.DataFrame(p_all, columns=["base_emb", "all_base", "all_emb"])
    
    for df_column in p_all.columns:
        df_results["p_"+df_column] = list(p_all[df_column])
    return df_results

ifg = sig_test(ifg, "IFG")

ifg = pd.read_csv("fMRI/results/IFG_results.csv"); ifg = sig_test(ifg, "IFG")
mfg = pd.read_csv("fMRI/results/MFG_results.csv"); mfg = sig_test(mfg, "MFG")
ifgorb = pd.read_csv("fMRI/results/IFGorb_results.csv"); ifgorb = sig_test(ifgorb, "IFGorb")
anttemp = pd.read_csv("fMRI/results/AntTemp_results.csv"); attemp = sig_test(anttemp, "AntTemp")
angularg = pd.read_csv("fMRI/results/AngularG_results.csv"); angularg = sig_test(angularg, "AngularG")
posttemp = pd.read_csv("fMRI/results/PostTemp_results.csv"); posttemp = sig_test(posttemp, "PostTemp")
pcing = pd.read_csv("fMRI/results/pCingulate_results.csv"); pcing = sig_test(pcing, "pCingulate")

base_plot(ifg, "act", "base", "all", "p", title="IFG", dh3 = .07, ylim1=-.1)
base_plot(mfg, "act", "base", "all", "p", title="MFG", annotate=False, dh1 = .07,dhlist2 = [.56, .38, .3, .3, .3], dhlist3 = [.69, .54, .07, .07, .07], ylim1=-.1)
base_plot(ifgorb, "act", "base", "all", "p", title="IFG$_{orb}$", dh3 = .07, ylim1=-.1)
base_plot(anttemp, "act", "base", "all", "p", title="AntTemp", annotate=False, dh1 = .07,dhlist2 = [.05, .25, .05, .22, .05], dhlist3 = [.2, .05, .07, .38, .07], ylim1=-.1)
base_plot(angularg, "act", "base", "all", "p", title="AngularG", annotate=False, dh1 = .07,dhlist2 = [.05, .4, .05, .05, .05], dhlist3 = [.05, .6, .07, .07, .07], ylim1=-.1)
base_plot(posttemp, "act", "base", "all", "p", title="PostTemp", annotate=False, dh1 = .07,dhlist2 = [.25, .25, .25, .25, .25], dhlist3 = [.38, .38, .38, .38, .38], ylim1=-.1)
base_plot(pcing, "act", "base", "all", "p", title="pCingulate", dh3 = .07, ylim1=-.1)



ifg = pd.read_csv("pereira/results/LIFG_results.csv")
mfg = pd.read_csv("pereira/results/LMFG_results.csv")
angg = pd.read_csv("pereira/results/LAngG_results.csv")
ifgorb = pd.read_csv("pereira/results/LIFGorb_results.csv")
anttemp = pd.read_csv("pereira/results/LAntTemp_results.csv")
posttemp = pd.read_csv("pereira/results/LPostTemp_results.csv")
manttemp = pd.read_csv("pereira/results/LMidAntTemp_results.csv")
mposttemp = pd.read_csv("pereira/results/LMidPostTemp_results.csv")

base_plot(ifg, "act", "base", "all", "p", title="IFG", ylim1=-.1, dh1 = 0.12,annotate=False, dhlist2 = [.399, .35, .35, .28, .28], dhlist3 = [.58, .58, .54, .38, .38])
base_plot(mfg, "act", "base", "all", "p", title="MFG", ylim1 = -.1, dh1 = .2, annotate=False, dhlist2 = [.499, .43, .43, .38, .38], dhlist3 = [.82, .64, .54, .5, .5])
base_plot(ifgorb, "act", "base", "all", "p", title="IFG$_{orb}$", ylim1 = -.1)
base_plot(angg, "act", "base", "all", "p", title="AngularG", ylim1 = -.1, dh3 = .07)
base_plot(anttemp, "act", "base", "all", "p", title="AntTemp", ylim1 = -.1, dh3 = .07)
base_plot(posttemp, "act", "base", "all", "p", title="PostTemp", ylim1 = -.1, dh3 = .07)
base_plot(mposttemp, "act", "base", "all", "p", title="MidPostTemp", ylim1 = -.1, dh3 = .07)
base_plot(manttemp, "act", "base", "all", "p", title="MidAntTemp", ylim1 = -.1, dh3 = .07)
