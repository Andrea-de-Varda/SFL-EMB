import pickle
import pandas as pd
import matplotlib.pyplot as plt
from os import chdir
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import pearsonr
import scipy.io
from tqdm import tqdm
import re
import pyreadr
import random
from math import floor

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = _Path(__file__).resolve().parent.parent
dirName = str(PROJECT_ROOT)
chdir(dirName)

freq = pd.read_excel(PROJECT_ROOT / "data/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}

def progressive_splithalf(nested_list, n_iterations=10, verbose = False):
    shortest_length = max(len(sublist) for sublist in nested_list)  # length of the longest element in the nested list
    out_all = []
    # iterate over half of the longest sublist length
    for n_samples in tqdm(range(1, int(shortest_length / 2) + 1, 1), total = int(shortest_length / 2)):
        out_n = []
        for n_iter in range(n_iterations):
            a, b = [], []
            for sublist in nested_list:
                use_n_samples = min(n_samples, floor(len(sublist) / 2))
                random.seed(n_iter)
                random.shuffle(sublist)
                # Calculate mean avoiding empty slices
                first_half_mean = np.mean(sublist[:use_n_samples]) if sublist[:use_n_samples] else np.nan
                second_half_mean = np.mean(sublist[use_n_samples:use_n_samples * 2]) if sublist[use_n_samples:use_n_samples * 2] else np.nan
                if not np.isnan(first_half_mean) and not np.isnan(second_half_mean):
                    a.append(first_half_mean)
                    b.append(second_half_mean)
            
            # check a and b are not empty and do not contain nans
            if a and b and not any(np.isnan(a + b)):
                r, _ = pearsonr(a, b)
                out_n.append(r)
        if out_n:  # check if empty list
            out_all.append([n_samples*2, np.mean(out_n), np.std(out_n)])
            if verbose:
                print(f"Processed {n_samples} of {int(shortest_length / 2) + 1}\nR = {np.mean(out_n)}")

    out_all = pd.DataFrame(out_all, columns=["n", "m", "sd"])
    plt.figure(dpi = 300)
    plt.errorbar(out_all['n'], out_all['m'], yerr=out_all['sd'], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
    plt.xlabel('Sample size')
    plt.ylabel('R')
    plt.grid(True)
    plt.show()
    return out_all    

def nobs(dataset):
    print(np.mean([len(sublist) for sublist in dataset]))
    
########################################
# natstor & brown -- need to recompute #
########################################

def create_word_pos(sentences): # to recreate word and sentence indexes as in original analyses 
    word_pos = []
    sent_nr = []
    for sentnum, sent in enumerate(sentences):
        for toknum, tok in enumerate(sent):
            word_pos.append(toknum+1)
            sent_nr.append(sentnum)
    return word_pos, sent_nr

natstor = pd.read_csv(PROJECT_ROOT / "data/NaturalStories/NaturalStories_RAW.tsv", sep = "\t")
natstor = natstor[natstor["correct"] == 6]

natst_sent = [re.sub("\n", "", s).split() for s in open("corpora/natural_stories.txt").readlines()] 

participants = natstor['WorkerId'].unique()

pos = create_word_pos(natst_sent)
grouped_natstor = natstor.groupby(["item", "zone", "word"], as_index=False).agg({'RT': list})
grouped_natstor["word_pos"], grouped_natstor["sent_nr"] = pos

natstor_data = grouped_natstor[grouped_natstor["word_pos"]>3]["RT"].tolist()

out_natstor = progressive_splithalf(natstor_data)  
nobs(natstor_data) # 59.63207654630368

#########
# brown #
#########

brown = pd.read_csv(PROJECT_ROOT / "data/Brown/brown_spr_RAW.csv")
brown_sent = [re.sub("\n", "", s).split() for s in open("corpora/bnc_brown.txt").readlines()]

participants = brown['subject'].unique()

pos = create_word_pos(brown_sent)
grouped_brown = brown.groupby(["text_id", "text_pos", "word"], as_index=False).agg({'time': list})
grouped_brown["word_pos"], grouped_brown["sent_nr"] = pos

brown_data = grouped_brown[grouped_brown["word_pos"]>3]["time"].tolist()

out_brown = progressive_splithalf(brown_data)
nobs(brown_data) # 19.05220061412487

################
# ucl spr - et #
################

eye = pd.read_csv(PROJECT_ROOT / "data/UCL_ET_CORPUS/eyetracking.RT.txt", sep="\t")
eye = eye[eye.word_pos > 1] # in original analyses, excluded first word

participants = eye['subj_nr'].unique()

grouped_eye = eye.groupby(["sent_nr", "word_pos"], as_index=False).agg({'RTfirstfix': list})
eye_data = grouped_eye["RTfirstfix"].tolist()

out_eye = progressive_splithalf(eye_data)
nobs(eye_data) # 42.005793742757824

eye_sent_nr = set(eye["sent_nr"])

#######
# spr #
#######

spr = pd.read_csv(PROJECT_ROOT / "data/UCL_ET_CORPUS/selfpacedreading.RT.txt", sep="\t")
spr = spr[spr.sent_nr.isin(eye_sent_nr)]
spr = spr[spr.word_pos > 3]

subjects = spr['subj_nr'].unique()

grouped_spr = spr.groupby(["sent_nr", "word_pos"], as_index=False).agg({'RT': list})
spr_data = grouped_spr["RT"].tolist()

out_spr = progressive_splithalf(spr_data)
nobs(spr_data) # 71.11398176291793

###########
# ucl ERP #
###########

mat = scipy.io.loadmat(PROJECT_ROOT / "data/ERP_Frank/stimuli_erp.mat")

ucl_n400 = []
for idx, sentence in enumerate(mat["sentences"]):
    tokens = sentence[0][0]
    sentence = [t[0] for t in tokens]
    ERP = mat["ERP"][idx][0]
    artefact = mat["artefact"][idx][0]
    ERP[np.where(artefact == 1)] = np.nan # filter artefacts
    for index in range(1, len(sentence)): # from 1 since filtered sent-initial in main analyses
        word = sentence[index]
        if "." in word: # removing punct as in original analyses
            pass
        elif "," in word:
            pass
        else:
            erp = ERP[index]
            n400 = erp[:, 2].tolist()
            ucl_n400.append(n400)
           
out_n400 = progressive_splithalf(ucl_n400)
nobs(ucl_n400) # 24

#########
# provo #
#########

# note: in provo, first word is already missing -- no need to exclude anything

provo = pd.read_csv(PROJECT_ROOT / "data/Provo/Provo_Corpus-Eyetracking_Data.csv", sep=",")
provo = provo[['Participant_ID','Word_Unique_ID', 'Word_Number', 'Sentence_Number', 'Word_In_Sentence_Number', 'Word', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME']]
provo['IA_FIRST_FIXATION_DURATION'] = provo['IA_FIRST_FIXATION_DURATION'].fillna(0)
provo['IA_FIRST_RUN_DWELL_TIME'] = provo['IA_FIRST_RUN_DWELL_TIME'].fillna(0)

participant_ids = provo['Participant_ID'].unique()

grouped_provo = provo.groupby(['Word_Unique_ID', 'Word_Number', 'Sentence_Number', 'Word_In_Sentence_Number', 'Word'], as_index=False).agg({'IA_FIRST_RUN_DWELL_TIME': list})
provo_data = grouped_provo["IA_FIRST_RUN_DWELL_TIME"].tolist()

out_provo = progressive_splithalf(provo_data)
nobs(provo_data) # 84

########
# maze #
########

maze = pyreadr.read_r(PROJECT_ROOT / "data/NaturalStories/natstor_maze.rds")
df_maze = maze[None]

maze = df_maze[df_maze["native"].isin(['ENG', 'ENGLISH', 'English','english'])]

sent_n_dict = {sent : idx for idx, sent in enumerate(set(maze["sentence"]))}
sent_n_d_rev = {value : key for key, value in sent_n_dict.items()}
maze["sent_n"] = maze.sentence.map(sent_n_dict)
maze["unique"] = maze["sent_n"].astype(str)+"-"+maze["word_num"].astype(str)

maze_sent = [re.sub("\n", "", s).split() for s in open("corpora/maze_sentences.txt").readlines()]
maze_sent_dict_recover = {" ".join(l):idx for idx, l in enumerate(maze_sent)}
maze["sent_n"] = maze.sentence.map(maze_sent_dict_recover)

part_dict = {}
for subj in set(maze["subject"]):
    temp = maze[maze["subject"] == subj]
    corr = len(temp[temp["correct"] == "yes"])/len(temp) > .8
    part_dict[subj] = corr
    
error = []
previous_idx = 0
previous_cor = True
for index, row in tqdm(maze.iterrows()):
    current_idx = row["word_num"]
    current_cor = row["correct"] == "yes"
    if current_idx - previous_idx == 1:
        if current_cor and previous_cor:
            error.append(0)
        else:
            error.append(1)
    else:
        if current_cor:
            error.append(0)
        else:
            error.append(1)
    previous_idx = current_idx
    previous_cor = current_cor
maze["error"] = error

maze["is_attentive"] = maze["subject"].map(part_dict)
maze = maze[(maze["type"] != "practice") & (maze["rt"] < 5000) & (maze["rt"] > 100) & (maze["is_attentive"] == True) & (maze["word_num"] !=0) & (maze["error"] == 0)]

maze_sent_len = {idx : length for idx, length in enumerate([len(re.sub("\n", "", s).split()) for s in open("corpora/maze_sentences.txt").readlines()])}

maze_rt = []
for sent_n in tqdm(list(set(maze.sent_n))):
    temp = maze[maze.sent_n == sent_n]
    for n in range(maze_sent_len[sent_n]):
        therow = temp[temp.word_num == n]
        if therow.empty:
            pass
        else:
            if n == 0:
                pass
            else:
                maze_rt.append(therow["rt"].tolist())
maze_rt = [rt for rt in maze_rt if len(rt) > 1]

out_maze = progressive_splithalf(maze_rt)
nobs(maze_rt) # 5.85898485468686

########
# MECO #
########

meco = pd.read_csv(PROJECT_ROOT / "data/Meco/Meco_l1.csv", index_col=None)
meco = meco.iloc[: , 1:]; print("N (with NA) =", len(meco))
meco = meco[meco['ia'].notna()]; print("N (no NA) =",len(meco))
meco["trialid"] = meco.trialid.astype(int)
meco["itemid"] = meco.itemid.astype(int)
meco["ianum"] = meco.ianum.astype(int)
meco["unique"] = meco.trialid.astype(str)+"-"+meco.ianum.astype(str)+"-"+meco.lang
meco = meco[~meco["subid"].isin(["ee_09", "ee_22", "ru_8"])] # wrong itemid assignment (as in 1.2 release)

meco['firstrun.dur'] = meco['firstrun.dur'].fillna(0)

out = []
for textid in set(meco.trialid):
    print(textid)
    temp = meco[(meco["lang"] == "en") & (meco["trialid"] == textid)]
    therange = range(1, max(temp.ianum)+1)
    temp_out = []
    temp_out_unique = []
    for itemindex in therange:
        try:
            token = list(set(temp[temp.ianum == itemindex].ia))
            unique = list(set(temp[temp.ianum == itemindex].unique))
            if len(token) > 1:
                print(textid, itemindex, token)
            temp_out_unique.append(unique[0])
            temp_out.append(token[0])
        except IndexError:
            print(language, textid, itemindex)
    out.append([textid, temp_out, temp_out_unique])

meco_rt = []
for text in tqdm(out):
    ids = text[2]
    for uniq in ids[1:]:
        temp = meco[meco.unique == uniq] # fpass, count NaN (i.e. skips) as 0
        temp['firstrun.dur'] = temp['firstrun.dur'].fillna(0)
        meco_rt.append(temp["firstrun.dur"])
meco_rt = [c.tolist() for c in meco_rt]

out_meco = progressive_splithalf(meco_rt)
nobs(meco_rt) # 40.13740458015267

########
# ZuCo #
########

out = pd.read_csv(PROJECT_ROOT / "data/ZuCo/ET_data.csv")
out = out.fillna(0)

out["unique"] = out["idx"].astype(str) + "-" + out["word_idx"].astype(str)
out = out[out.word_idx != 0]
       
zuco_rt = []
for uniq in tqdm(set(out["unique"])):
    temp = out[out.unique == uniq]
    idx = len(temp) // 2
    gd = temp["gd"].tolist()
    zuco_rt.append(gd)
    
out_zuco = progressive_splithalf(zuco_rt)
nobs(zuco_rt) # 17.435381519888562

#############
# Michaelov #
#############

# [removed: Michaelov N400 dataset is not shipped with the repo]
# df = pd.read_csv(".../michaelov_n400/N400_data.csv")
# df1 = df.groupby(["Subject", "Sentence"], as_index=False, sort=False).agg({'Cloze':"mean", 'N400':"mean", 'PlausibilityJudgement':"mean", "Condition":"max"})
# grouped_mich = df1.groupby(["Sentence"], as_index=False).agg({'N400': list})
# mich_data = grouped_mich["N400"].tolist()
# out_mich = progressive_splithalf(mich_data)

###########
# Pereira #
###########

df = pd.read_parquet("pereira_1/Pereira.parquet.gzip", columns=['UID', 'Sentence', 'ROI', 'EffectSize'])

lang_rois = ["Lang_LH_IFGorb", "Lang_LH_IFG", "Lang_LH_MFG", "Lang_LH_AntTemp", "Lang_LH_PostTemp"]
pereira = df[df.ROI.isin(lang_rois)]
pereira['ROI'] = pereira['ROI'].cat.remove_unused_categories()

# get average sentence-level response
grouped  = pereira.groupby(["UID", "ROI", "Sentence"]).agg({"EffectSize" : "mean"}) # first avg voxels in [part, roi, sent]
grouped1 = grouped.groupby(["UID", "Sentence"]).agg({"EffectSize" : "mean"}) # then avg rois in [part, sent]
grouped2 = grouped1.groupby(["Sentence"]).agg({"EffectSize" : list}) # then avg rois in sent

pereira_data = grouped2["EffectSize"].tolist()
pereira_data = [[x for x in row if not (isinstance(x, float) and np.isnan(x))] for row in pereira_data]


# from utils import loadmat1

# datadir = "/media/drive2/Datasets/Pereira/data/"

# def load_brain_data_Pereira(path):
#     subject_data = loadmat1(datadir+path+"/examples_384sentences.mat")
#     subject_images = subject_data["examples"]
    
#     subject_areas = loadmat1(datadir+path+"/data_384sentences.mat")
#     lang_rois = subject_areas["meta"]["rois"][1]
#     roi_masks = subject_areas["meta"]["roiMultimask"][1]
    
#     # filtering per ROI
#     dict_roi = {}
#     for roi_name, roi_idx in zip(lang_rois, list(range(1, max(roi_masks)+1))):
#         subj_roi = subject_images[:,roi_masks == roi_idx]
#         dict_roi[roi_name] = subj_roi
    
#     dict_roi["all"] = subject_images[:,roi_masks != 0]
    
#     return dict_roi

# participants = ["M02", "M04", "M07", "M08", "M09", "M14", "M15", "P01"]
# brain_data = [load_brain_data_Pereira(p) for p in participants]

# pereira_data = {i : [] for i in range(384)}
# for part in brain_data:
#     part_act = part["all"].mean(axis=1)
#     for i, act in enumerate(part_act):
#         pereira_data[i].append(act)
# pereira_data = list(pereira_data.values())

out_pereira = progressive_splithalf(pereira_data)
nobs(pereira_data) # 7.837320574162679

######
# HP #
######

part_f = np.load("fMRI/fMRI/data_subject_F.npy")
part_h = np.load("fMRI/fMRI/data_subject_H.npy")
part_i = np.load("fMRI/fMRI/data_subject_I.npy")
part_j = np.load("fMRI/fMRI/data_subject_J.npy")
part_k = np.load("fMRI/fMRI/data_subject_K.npy")
part_l = np.load("fMRI/fMRI/data_subject_L.npy")
part_m = np.load("fMRI/fMRI/data_subject_M.npy")
part_n = np.load("fMRI/fMRI/data_subject_N.npy")

roi = np.load('fMRI/fMRI/HP_subj_roi_inds.npy', allow_pickle=True)
roi = roi.tolist()

parts = [part_f, part_h, part_i, part_j, part_k, part_l, part_m, part_n]
partn = ["F", "H", "I", "J", "K", "L", "M", "N"]
roin =  ["all", "PostTemp", "AntTemp", "AngularG", "IFG", "MFG", "IFGorb", "pCingulate"]

the_roi = "all"
hp_data = {i : [] for i in range(1211)}
for part_act, part_name in zip(parts, partn):
    part = part_act[:,roi[part_name][the_roi]].mean(axis=1)
    for i, act in enumerate(part):
        hp_data[i].append(act)
hp_data = list(hp_data.values())

out_hp = progressive_splithalf(hp_data)
nobs(hp_data) # 8

################
# NatStor fMRI #
################

ns = pd.read_csv("natstor_fmri/NaturalStories_TimeSeries_20220615.csv")

d_story_groups = {"kingofbirds" : ["StoriesAud_4_kingofbirds_N", "Stories_ToM_4_kingofbirds_N", "Stories_4_kingofbirds_N"],
                  "elvis" : ["StoriesAud_5_elvis_T", "Stories_ToM_5_elvis_T", "Stories_5_elvis_T", "ISCmega_elvis"], # ISCmega_elvis removed (different n° of TRs)
                  "aqua" : ["Stories_ToM_2_aqua_T", "Stories_2_aqua_T", "StoriesAud_2_aqua_T"],
                  "tulips" : ["Stories_9_tulips_T"],
                  "boar" : ["Stories_ToM_1_boar_T", "Stories_1_boar_T", "StoriesAud_1_boar_T"],
                  "tree" : ["ISCmega_tree"], 
                  "highschool" : ["Stories_ToM_7_highschool_N", "Stories_7_highschool_N"],
                  "matchstickseller" : ["Stories_3_matchstickseller_N"],
                  "mrsticky" : ["Stories_6_mrsticky_N"]}

def split_half(timeseries, n_splits=100):
    n_rows, n_cols = timeseries.shape
    correlations = np.zeros(n_splits)

    for i in range(n_splits):
        np.random.seed(i)
        indices = np.random.permutation(n_rows)
        half = n_rows // 2
        group1_indices = indices[:half]
        group2_indices = indices[half:]

        # get mean for two groups
        mean1 = np.mean(timeseries[group1_indices, :], axis=0)
        mean2 = np.mean(timeseries[group2_indices, :], axis=0)

        corr, _ = pearsonr(mean1, mean2)
        correlations[i] = corr

    return np.mean(correlations), np.std(correlations)

lang_rois = ['Lang_LH_AntTemp', 'Lang_LH_IFG', 'Lang_LH_IFGorb', 'Lang_LH_MFG', 'Lang_LH_PostTemp']
ns_tr_columns = ["T_" + str(n) for n in range(1, 227)]

d_timeseries = {group : [] for group in d_story_groups.keys()} # all ts of individual subjects
for story_group, story_list in d_story_groups.items():
    temp = ns[(ns.Story.isin(story_list)) & (ns.ROI.isin(lang_rois))].copy().dropna(axis=1, how="all")
    story_parts = temp.UID.unique()
    # print(len(story_parts))
    for part in story_parts:
        if story_group == "elvis": # ISCmega_elvis had different N° of TRs
            elvis_columns = ["T_" + str(n) for n in range(1, 152)]
            thepart = temp[temp.UID == part].filter(elvis_columns).mean(axis=0).tolist()
            d_timeseries[story_group].append(thepart)
        else:
            thepart = temp[temp.UID == part].filter(ns_tr_columns).mean(axis=0).tolist()
            d_timeseries[story_group].append(thepart)

d_timeseries = {k : np.array(v) for k, v in d_timeseries.items()}

# OPTION 1: one reliab plot per story
d_reliab = {}
for k, v in d_timeseries.items():
    the_ts = v.T.tolist() # from (items, parts) to (parts, items)
    d_reliab[k] = progressive_splithalf(the_ts)

# OPTION 2: one single reliab for the dataset
natstor_fmri_data = []
for k, v in d_timeseries.items():
    the_ts = v.T.tolist()
    natstor_fmri_data.extend(the_ts)

out_natstor_fmri = progressive_splithalf(natstor_fmri_data)
nobs(natstor_fmri_data) # 27.183734939759034

###########
# Control #
###########

control = pd.read_csv("control_fmri/data/brain-lang-data_participant_20230728.csv")
control = control[(control["roi"] == "lang_LH_netw")]

control_bold = []
for uniq in tqdm(set(control["sentence"])):
    temp = control[control.sentence == uniq]
    bold = temp["response_target"].tolist()
    control_bold.append(bold)

out_control = progressive_splithalf(control_bold)
nobs(control_bold) # 6.25

###############################################################################

############
# plotting #
############

def plot_group(dfs, names, title = "", themax = 0, ylim = None, do_shifts = False, shifts = None):
    plt.figure(figsize = (4, 3), dpi = 300)
    idx = 0
    for out_all, name in zip(dfs, names):
        plt.errorbar(out_all['n'], out_all['m'], yerr=out_all['sd'], fmt='-', capsize=0, linewidth = 2, elinewidth=2, markeredgewidth=0, alpha = .8)
        newmax = out_all['n'].max()
        if do_shifts:
            plt.text(out_all['n'].max() + shifts[idx][0], out_all['m'].iloc[-1] + shifts[idx][1], name, fontsize = 12)
            print(out_all['n'].max() + shifts[idx][0])
            idx += 1
        else:
            plt.text(out_all['n'].max() + 0.05 * newmax, out_all['m'].iloc[-1], name, fontsize = 12)
            if newmax > themax:
                themax = newmax
    xlim = max([1, themax + (0.3 * themax)])
    print(xlim)
    plt.xlabel('Sample size')
    plt.xlim(None, xlim)
    plt.ylim(None, ylim)
    plt.ylabel('R')
    plt.title(title, fontsize = 18)
    plt.grid(True)
    plt.show()
    
# ET
plot_group([out_eye, out_provo, out_meco, out_zuco], names = ["UCL$_{ET}$", "Provo", "MECO-en", "ZuCo-2"], title = "Eye-tracking", do_shifts = True, shifts = [[3, -.05], [3, 0], [-22, +.04], [-19, .05]], themax = 90, ylim = 1)

# SPR
plot_group([out_natstor, out_brown, out_spr], names = ["NatStor$_{SPR}$", "Brown", "UCL$_{SPR}$"], title = "Self-paced reading", themax = 90, ylim = 0.76)

# N400
#plot_group([out_n400, out_mich], names = ["UCL$_{N400}$", "Michaelov"], title = "N400", do_shifts = True, shifts = [[1, .01], [-7, .05]], themax = 24.5)
plot_group([out_n400], names = ["UCL$_{N400}$"], title = "N400", do_shifts = True, shifts = [[1, .01], [-7, .05]], themax = 24.5)

# Maze
plot_group([out_maze], names = ["NatStor$_{Maze}$"], title = "Maze", themax = 9)

# fMRI
plot_group([out_control, out_pereira], names = ["Control", "Pereira"], title = "fMRI, sentences")
plot_group([out_natstor_fmri, out_hp], names = ["NatStor$_{fMRI}$", "Harry Potter"], title = "fMRI, stories", themax = 84)

tosave = [out_eye, out_provo, out_meco, out_zuco, out_natstor, out_brown, out_spr, out_maze, out_n400, out_control, out_pereira, out_natstor_fmri, out_hp]  # [removed out_mich: Michaelov not in repo]
with open("other/reliab_size", 'wb') as file:
    pickle.dump(tosave, file)

with open("other/reliab_size", 'rb') as file:
    (
        out_eye, out_provo, out_meco, out_zuco,
        out_natstor, out_brown, out_spr, out_maze,
        out_n400, out_control, out_pereira,
        out_natstor_fmri, out_hp
    ) = pickle.load(file)
