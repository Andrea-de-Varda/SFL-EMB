import pandas as pd
import numpy as np
from os import chdir, path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
dirName = str(PROJECT_ROOT)
chdir(dirName)

res_meco  = pd.read_csv("results_meco_1.csv")
res_nsbr  = pd.read_csv("results_natstor_brown.csv")
res_nsbr1 = pd.read_csv("results_natstor_brown_spill.csv")
res_ucl   = pd.read_csv("results/results_UCL.csv")
res_ucl1   = pd.read_csv("results/results_UCL_punct.csv")
res_ucl2  = pd.read_csv("results/results_UCL_nopunct.csv")
# res_p600  = pd.read_csv("results/results_p600_UCL.csv")
res_prov  = pd.read_csv("results_provo_1.csv")
res_maze  = pd.read_csv("results_maze_spill.csv")
fmri = pd.read_csv("fMRI/results/all_results.csv")
pereira = pd.read_csv("pereira_1/results/results.csv")
control = pd.read_csv("control_fmri/results/control_results.csv")
natstor_fmri = pd.read_csv("natstor_fmri/results/all_results.csv") #### CONTINUE

res_ucl = res_ucl.iloc[:-1,:]

# ET
mec_dr = np.mean(res_meco["all_fpass"]-res_meco["base_fpass"])
pro_dr = np.mean(res_prov["all_fpass"]-res_prov["base_fpass"])
ucl_dr = np.mean(res_ucl["all_fpass"]-res_ucl["base_fpass"])

# SPR
brown_dr = np.mean(res_nsbr["all_brown"]-res_nsbr["base_brown"])
natst_dr = np.mean(res_nsbr["all_natstor"]-res_nsbr["base_natstor"])
uclsp_dr = np.mean(res_ucl1["all_spr"]-res_ucl1["base_spr"])

# SPR, spill
brown_dr1 = np.mean(res_nsbr1["all_brown"]-res_nsbr1["base_brown"])
natst_dr1 = np.mean(res_nsbr1["all_natstor"]-res_nsbr1["base_natstor"])
uclsp_dr1 = np.mean(res_ucl1["all_spr"]-res_ucl1["base_spr"])

# ERP
n400_dr = np.mean(res_ucl1["all_n400"]-res_ucl1["base_n400"])
#p600_dr = np.mean(res_p600["all_p600"]-res_p600["base_p600:"])

# maze
maze_dr = np.mean(res_maze["all"]-res_maze["base"])



# correlate EMB and SFl results

def corr_df(df, colname):
    return spearmanr(df["base_"+colname], df["act_"+colname])

# ET
corr_df(res_ucl, "fpass")
corr_df(res_meco, "fpass")
corr_df(res_prov, "fpass")

# SPR
corr_df(res_nsbr1, "natstor")
corr_df(res_nsbr1, "brown")
corr_df(res_ucl, "spr")

#########################################
# Quantify distance between EMB and SFL #
#########################################

res_meco  = pd.read_csv("results_meco_1.csv")
res_nsbr  = pd.read_csv("results_natstor_brown.csv")
res_nsbr1 = pd.read_csv("results_natstor_brown_spill.csv")
res_ucl   = pd.read_csv("results/results_UCL.csv")
res_ucl1   = pd.read_csv("results/results_UCL_punct.csv")
res_ucl2  = pd.read_csv("results/results_UCL_nopunct.csv")

res_prov  = pd.read_csv("results_provo_1.csv")
res_maze  = pd.read_csv("results_maze_spill.csv")
fmri = pd.read_csv("fMRI/results/all_results.csv")
pereira = pd.read_csv("pereira/results/all_results.csv")

(np.mean(res_meco["base_fpass"])/np.mean(res_meco["act_fpass"])-1)*9
(np.mean(res_ucl["base_fpass"])/np.mean(res_ucl["act_fpass"])-1)*9
(np.mean(res_prov["base_fpass"])/np.mean(res_prov["act_fpass"])-1)*9

(np.mean(res_nsbr1["base_natstor"])/np.mean(res_nsbr1["act_natstor"])-1)*9
(np.mean(res_nsbr1["base_brown"])/np.mean(res_nsbr1["act_brown"])-1)*9
(np.mean(res_ucl1["base_spr"])/np.mean(res_ucl1["act_spr"])-1)*9

(np.mean(res_maze["act"])/np.mean(res_maze["base"])-1)*9

(np.mean(fmri["act"])/np.mean(fmri["base"])-1)*9
(np.mean(pereira["act"])/np.mean(pereira["base"])-1)*9
(np.mean(res_ucl2["act_n400"])/np.mean(res_ucl2["base_n400"])-1)*9
(np.mean(res_ucl2["act_p600"])/np.mean(res_ucl2["base_p600"])-1)*9
































# check if correlation between variance explained by full model and DeltaR
# ET
mec_a = np.mean(res_meco["all_fpass"])
pro_a = np.mean(res_prov["all_fpass"])
ucl_a = np.mean(res_ucl["all_fpass"])
brown_a = np.mean(res_nsbr["all_brown"])
natst_a = np.mean(res_nsbr["all_natstor"])
uclsp_a = np.mean(res_ucl1["all_spr"])
brown1_a = np.mean(res_nsbr1["all_brown"])
natst1_a = np.mean(res_nsbr1["all_natstor"])
uclsp1_a = np.mean(res_ucl1["all_spr"])
n400_a = np.mean(res_ucl["all_n400"])
#p600_a = np.mean(res_p600["all_p600"])
maze_a = np.mean(res_maze["all"])

x = [mec_a, pro_a, ucl_a, brown1_a, natst1_a, uclsp_a, maze_a]#, n400_a, p600_a]
y = [mec_dr, pro_dr, ucl_dr, brown_dr1, natst_dr1, uclsp_dr1, maze_dr]#, n400_dr, p600_dr]
names = ["MECO-en", "Provo", "UCL, ET", "Brown, SPR", "NatStor, SPR", "UCL, SPR", "NatStor, Maze"]#, "N400", "P600"]

spearmanr(x, y)
#coef = np.polyfit(x, y, 1)
#poly1d_fn = np.poly1d(coef)

plt.figure(dpi=300)
plt.scatter(x, y)
plt.ylim(None, 0.1)
plt.xlim(None, 1)
plt.ylabel("$\overline{\Delta r}$", fontsize=13)
plt.xlabel("$r_{EMB^\\frown SFL}$", fontsize=13)
#plt.plot(x, poly1d_fn(x), '-b')
for thex, they, z in zip(x, y, names):
    if z == "MECO-en":
        plt.annotate(z, (thex+0.007, they), rotation=30)
    elif z in ["Provo", "UCL, ET"]:
        plt.annotate(z, (thex+0.003, they), rotation=30)
    else:
        plt.annotate(z, (thex+0.003, they))
plt.annotate("Spearman $\\rho$ = -0.6, p = 0.09", (0.65, 0.24), fontsize=11)
plt.show()




