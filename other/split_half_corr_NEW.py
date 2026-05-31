import pickle
import pandas as pd
import matplotlib.pyplot as plt
from os import chdir
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import pearsonr
import scipy.io
from tqdm import tqdm
import re
import pyreadr
import random

dirName = "/home/dev/Documents/PhD/MIT/project"
chdir(dirName)

freq = pd.read_excel("/home/dev/Documents/Datasets/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}

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

natstor = pd.read_csv("/home/dev/Documents/Datasets/NaturalStories/NaturalStories_RAW.tsv", sep = "\t")
natstor = natstor[natstor["correct"] == 6]

natst_sent = [re.sub("\n", "", s).split() for s in open("corpora/natural_stories.txt").readlines()] 

participants = natstor['WorkerId'].unique()

natstor_r = []
for n in range(100):
    random.seed(n)
    random.shuffle(participants)
    half = len(participants) // 2
    group1_participants = participants[:half]
    group2_participants = participants[half:]
    
    group1_df = natstor[natstor['WorkerId'].isin(group1_participants)]
    group2_df = natstor[natstor['WorkerId'].isin(group2_participants)]
    
    group1_means = group1_df.groupby(["item", "zone", "word"], as_index=False).agg({'RT':"mean"})
    group2_means = group2_df.groupby(["item", "zone", "word"], as_index=False).agg({'RT':"mean"})
    
    pos = create_word_pos(natst_sent)
    group1_means["word_pos"], group1_means["sent_nr"] = pos
    group2_means["word_pos"], group2_means["sent_nr"] = pos
    
    merged_df = pd.merge(group1_means, group2_means, on=["item", "zone", "word", "word_pos", "sent_nr"], suffixes=('_group1', '_group2'))
    merged_df = merged_df[merged_df["word_pos"] > 3] # spillover in original analyses
    natstor_r_temp = merged_df[f'RT_group1'].corr(merged_df[f'RT_group2'])
    natstor_r.append(natstor_r_temp)
natstor_r = np.mean(natstor_r)

#########
# brown #
#########

brown = pd.read_csv("/home/dev/Documents/Datasets/Brown/brown_spr_RAW.csv")
brown_sent = [re.sub("\n", "", s).split() for s in open("corpora/bnc_brown.txt").readlines()]

participants = brown['subject'].unique()

brown_r = []
for n in range(100):
    random.seed(n)
    random.shuffle(participants)
    half = len(participants) // 2
    group1_participants = participants[:half]
    group2_participants = participants[half:]
    
    group1_df = brown[brown['subject'].isin(group1_participants)]
    group2_df = brown[brown['subject'].isin(group2_participants)]
    
    group1_means = group1_df.groupby(["text_id", "text_pos", "word"], as_index=False).agg({'time':"mean"})
    group2_means = group2_df.groupby(["text_id", "text_pos", "word"], as_index=False).agg({'time':"mean"})
    
    pos = create_word_pos(brown_sent)
    group1_means["word_pos"], group1_means["sent_nr"] = pos
    group2_means["word_pos"], group2_means["sent_nr"] = pos
    
    merged_df = pd.merge(group1_means, group2_means, on=["text_id", "text_pos", "word", "word_pos", "sent_nr"], suffixes=('_group1', '_group2'))
    merged_df = merged_df[merged_df["word_pos"] > 3] # spillover in original analyses
    brown_r_temp = merged_df[f'time_group1'].corr(merged_df[f'time_group2'])
    brown_r.append(brown_r_temp)
brown_r = np.mean(brown_r)

################
# ucl spr - et #
################

eye = pd.read_csv('/home/dev/Documents/Datasets/UCL_ET_Corpus/eyetracking.RT.txt', sep="\t")
eye = eye[eye.word_pos > 1] # in original analyses, excluded first word

participants = eye['subj_nr'].unique()

ucl_eye_r = []
for n in range(100):
    random.seed(n)
    random.shuffle(participants)
    half = len(participants) // 2
    group1_participants = participants[:half]
    group2_participants = participants[half:]
    
    group1_df = eye[eye['subj_nr'].isin(group1_participants)]
    group2_df = eye[eye['subj_nr'].isin(group2_participants)]
    
    group1_means = group1_df.groupby(["sent_nr", "word_pos"], as_index=False).agg({'RTfirstfix':"mean", 'RTfirstpass':"mean", 'RTrightbound':"mean", 'RTgopast':"mean"})
    group2_means = group2_df.groupby(["sent_nr", "word_pos"], as_index=False).agg({'RTfirstfix':"mean", 'RTfirstpass':"mean", 'RTrightbound':"mean", 'RTgopast':"mean"})

    merged_df = pd.merge(group1_means, group2_means, on=["sent_nr", "word_pos"], suffixes=('_group1', '_group2'))

    ucl_eye_r_temp = merged_df[f'RTfirstpass_group1'].corr(merged_df[f'RTfirstpass_group2'])
    ucl_eye_r.append(ucl_eye_r_temp)
ucl_eye_r = np.mean(ucl_eye_r)
# {'RTfirstfix': 0.8002231253055534, 'RTfirstpass': 0.8383128916420272, 'RTrightbound': 0.8563922145074448, 'RTgopast': 0.8437280640265847}

eye_sent_nr = set(eye["sent_nr"])

#######
# spr #
#######

spr = pd.read_csv('/home/dev/Documents/Datasets/UCL_ET_Corpus/selfpacedreading.RT.txt', sep="\t")
spr = spr[spr.sent_nr.isin(eye_sent_nr)]
spr = spr[spr.word_pos > 3]

subjects = spr['subj_nr'].unique()

ucl_spr_r = []
for n in range(100):
    random.seed(n)
    random.shuffle(subjects)
    
    half = len(subjects) // 2
    group1_subjects = subjects[:half]
    group2_subjects = subjects[half:]
    
    group1_df = spr[spr['subj_nr'].isin(group1_subjects)]
    group2_df = spr[spr['subj_nr'].isin(group2_subjects)]
    
    group1_means = group1_df.groupby(['sent_nr', 'word_pos'], as_index=False)['RT'].mean()
    group2_means = group2_df.groupby(['sent_nr', 'word_pos'], as_index=False)['RT'].mean()
    
    merged_means = pd.merge(group1_means, group2_means, on=['sent_nr', 'word_pos'], suffixes=('_group1', '_group2'))
    
    ucl_spr_r_temp = merged_means['RT_group1'].corr(merged_means['RT_group2'])
    ucl_spr_r.append(ucl_spr_r_temp)
ucl_spr_r = np.mean(ucl_spr_r)

###########
# ucl ERP #
###########

mat = scipy.io.loadmat('/home/dev/Documents/Datasets/ERP_Frank/stimuli_erp.mat')
mat.keys()

ucl_n400_r = []
ucl_p600_r = []
for n in range(100):
    out = []
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
                all_parts = list(range(24))
                random.seed(n)
                parts_random = random.sample(all_parts, 12)
                parts_random1 = [p for p in all_parts if p not in parts_random]
                ELAN, LAN, N400, EPNP, P600, PNP = np.nanmean(erp[tuple([parts_random])], axis=0)
                ELAN1, LAN1, N4001, EPNP1, P6001, PNP1 = np.nanmean(erp[tuple([parts_random1])], axis=0)
                out.append([" ".join(sentence).strip(), index+1, ELAN, LAN, N400, EPNP, P600, PNP, ELAN1, LAN1, N4001, EPNP1, P6001, PNP1])
            
    frank = pd.DataFrame(out, columns = ["sentence", "word_pos", "ELAN", "LAN", "N400", "EPNP", "P600", "PNP", "ELAN1", "LAN1", "N4001", "EPNP1", "P6001", "PNP"])
    
    ucl_n400_r_temp = frank['N400'].corr(frank['N4001'])
    ucl_p600_r_temp = frank['P600'].corr(frank['P6001'])
    ucl_n400_r.append(ucl_n400_r_temp)
    ucl_p600_r.append(ucl_p600_r_temp)

ucl_n400_r = np.mean(ucl_n400_r)
ucl_p600_r = np.mean(ucl_p600_r)

#########
# provo #
#########

# note: in provo, first word is already missing -- no need to exclude anything

provo = pd.read_csv('/home/dev/Documents/Datasets/Provo/Provo_Corpus-Eyetracking_Data.csv', sep=",")
provo = provo[['Participant_ID', 'Word_Number', 'Sentence_Number', 'Word_In_Sentence_Number', 'Word', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME']]
provo['IA_FIRST_FIXATION_DURATION'] = provo['IA_FIRST_FIXATION_DURATION'].fillna(0)
provo['IA_FIRST_RUN_DWELL_TIME'] = provo['IA_FIRST_RUN_DWELL_TIME'].fillna(0)

participant_ids = provo['Participant_ID'].unique()

provo_r = []
for n in range(100):
    random.seed(n)
    random.shuffle(participant_ids)
    
    half = len(participant_ids) // 2
    group1_ids = participant_ids[:half]
    group2_ids = participant_ids[half:]
    
    group1_df = provo[provo['Participant_ID'].isin(group1_ids)]
    group2_df = provo[provo['Participant_ID'].isin(group2_ids)]
    
    group1_means = group1_df.groupby(['Word_Unique_ID'], as_index=False).agg({
        'IA_FIRST_FIXATION_DURATION': "mean",
        'IA_FIRST_RUN_DWELL_TIME': "mean"
    })
    group2_means = group2_df.groupby(['Word_Unique_ID'], as_index=False).agg({
        'IA_FIRST_FIXATION_DURATION': "mean",
        'IA_FIRST_RUN_DWELL_TIME': "mean"
    })
    
    merged_means = pd.merge(group1_means, group2_means, on=['Word_Unique_ID'], suffixes=('_group1', '_group2'))
    
    provo_r_temp = merged_means['IA_FIRST_RUN_DWELL_TIME_group1'].corr(merged_means['IA_FIRST_RUN_DWELL_TIME_group2'])
    provo_r.append(provo_r_temp)

provo_r = np.mean(provo_r)

########
# maze #
########

maze = pyreadr.read_r('/home/dev/Documents/Datasets/NaturalStories/natstor_maze.rds')
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

maze_r = []
for n in range(100):
    a, b = [], []
    for rt in maze_rt:
        rt = np.array(rt)
        random.seed(n)
        all_parts = list(range(len(rt)))
        parts_random = random.sample(all_parts, len(all_parts) // 2)
        parts_random1 = [p for p in all_parts if p not in parts_random]
        part1 = np.mean(rt[tuple([parts_random])])
        part2 = np.mean(rt[tuple([parts_random1])])
        a.append(part1)
        b.append(part2)
    maze_r_temp, _ = pearsonr(a, b)
    maze_r.append(maze_r_temp)
maze_r = np.mean(maze_r)


########
# MECO #
########

meco = pd.read_csv("/home/dev/Documents/PhD/side-projects/eye-tracking/Meco_l1.csv", index_col=None)
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

meco_r = []
for n in range(100):
    a, b = [], []
    for rt in meco_rt:
        rt = np.array(rt)
        random.seed(n)
        all_parts = list(range(len(rt)))
        parts_random = random.sample(all_parts, len(all_parts) // 2)
        parts_random1 = [p for p in all_parts if p not in parts_random]
        part1 = np.mean(rt[tuple([parts_random])])
        part2 = np.mean(rt[tuple([parts_random1])])
        a.append(part1)
        b.append(part2)
    meco_r_temp, _ = pearsonr(a, b)
    meco_r.append(meco_r_temp)
meco_r = np.mean(meco_r)

########
# ZuCo #
########

out = pd.read_csv("/media/drive2/Datasets/ZuCo/ET_data.csv")
out = out.fillna(0)

out["unique"] = out["idx"].astype(str) + "-" + out["word_idx"].astype(str)
out = out[out.word_idx != 0]
       
zuco_rt = []
for uniq in tqdm(set(out["unique"])):
    temp = out[out.unique == uniq]
    idx = len(temp) // 2
    gd = temp["gd"].tolist()
    zuco_rt.append(gd)

zuco_r = []
for n in range(100):
    a, b = [], []
    for rt in zuco_rt:
        rt = np.array(rt)
        random.seed(n)
        all_parts = list(range(len(rt)))
        parts_random = random.sample(all_parts, len(all_parts) // 2)
        parts_random1 = [p for p in all_parts if p not in parts_random]
        part1 = np.mean(rt[tuple([parts_random])])
        part2 = np.mean(rt[tuple([parts_random1])])
        a.append(part1)
        b.append(part2)
    zuco_r_temp, _ = pearsonr(a, b)
    zuco_r.append(zuco_r_temp)
zuco_r = np.mean(zuco_r)

#############
# Michaelov #
#############

df = pd.read_csv("/home/dev/Documents/Datasets/michaelov_n400/N400_data.csv")
df1 = df.groupby(["Subject", "Sentence"], as_index=False, sort=False).agg({'Cloze':"mean", 'N400':"mean", 'PlausibilityJudgement':"mean", "Condition":"max"})

subjects = df1['Subject'].unique()

mich_r = []
for n in range(100):
    random.seed(n)
    random.shuffle(subjects)
    
    half = len(subjects) // 2
    group1_subjects = subjects[:half]
    group2_subjects = subjects[half:]
    
    group1_data = df1[df1['Subject'].isin(group1_subjects)]
    group2_data = df1[df1['Subject'].isin(group2_subjects)]
    
    group1_aggregated = group1_data.groupby('Sentence', as_index=False)['N400'].mean()
    group2_aggregated = group2_data.groupby('Sentence', as_index=False)['N400'].mean()
    
    merged_data = pd.merge(group1_aggregated, group2_aggregated, on='Sentence', suffixes=('_group1', '_group2'))
    
    mich_r_temp, _ = pearsonr(merged_data['N400_group1'], merged_data['N400_group2'])
    mich_r.append(mich_r_temp)
mich_r = np.mean(mich_r)

###########
# Pereira #
###########

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

# pereira_r = []
# for n in range(100):
#     random.seed(n)
#     random.shuffle(brain_data)
    
#     rois = ['LMidPostTemp', 'LPostTemp', 'LMidAntTemp', 'LIFG', 'LAntTemp', 'LIFGorb', 'LAngG', 'LMFG', 'all']
    
#     brain_data_avg1 = {}
#     for roi in rois:
#         tmp = []
#         for part in brain_data[:4]:
#             act = part[roi].mean(axis=1)
#             tmp.append(act)
#         brain_data_avg1[roi] = np.mean(tmp, axis=0)
        
#     brain_data_avg2 = {}
#     for roi in rois:
#         tmp = []
#         for part in brain_data[4:]:
#             act = part[roi].mean(axis=1)
#             tmp.append(act)
#         brain_data_avg2[roi] = np.mean(tmp, axis=0)
    
#     pereira_r_temp, _ = pearsonr(brain_data_avg1["all"], brain_data_avg2["all"])
#     pereira_r.append(pereira_r_temp)
# pereira_r = np.mean(pereira_r)

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

pereira_r_ = []
for n in range(100):
    a, b = [], []
    for resp in pereira_data:
        resp = np.array(resp)
        random.seed(n)
        all_parts = list(range(len(resp)))
        parts_random = random.sample(all_parts, len(all_parts) // 2)
        parts_random1 = [p for p in all_parts if p not in parts_random]
        part1 = np.mean(resp[parts_random])
        part2 = np.mean(resp[parts_random1])
        # Check for NaN 
        if not np.isnan(part1) and not np.isinf(part1) and not np.isnan(part2) and not np.isinf(part2):
            a.append(part1)
            b.append(part2)
    if a and b:
        pereira_r_temp, _ = pearsonr(a, b)
        pereira_r_.append(pereira_r_temp)
pereira_r = np.mean(pereira_r_)

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

hp_r = []
for n in range(100):
    random.seed(n)
    zipped = list(zip(parts, partn))

    random.shuffle(zipped)
    parts, partn = zip(*zipped)
    parts = list(parts)
    partn = list(partn)

    brain_data_avg_1 = {}
    for the_roi in roin:
        out = []
        for part_act, part_name in zip(parts[:4], partn[:4]):
            part = part_act[:,roi[part_name][the_roi]].mean(axis=1)
            out.append(part)
        out = np.array(out).mean(axis=0)
        brain_data_avg_1[the_roi] = out
        
    brain_data_avg_2 = {}
    for the_roi in roin:
        out = []
        for part_act, part_name in zip(parts[4:], partn[4:]):
            part = part_act[:,roi[part_name][the_roi]].mean(axis=1)
            out.append(part)
        out = np.array(out).mean(axis=0)
        brain_data_avg_2[the_roi] = out
        
    hp_r_temp, _ = pearsonr(brain_data_avg_1["all"], brain_data_avg_2["all"])
    hp_r.append(hp_r_temp)
hp_r = np.mean(hp_r)

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

df_reliab = []
for k, v in tqdm(d_timeseries.items(), total = len(d_timeseries.keys())):
    thesplit = split_half(v)
    n = v.shape[0]
    df_reliab.append([k, thesplit[0], thesplit[1], n])
df_reliab = pd.DataFrame(df_reliab, columns = ["story", "r", "sd", "n"]).sort_values(by="r")
natstor_fmri_r = df_reliab["r"].mean()

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

control_r_ = []
for n in range(100):
    a, b = [], []
    for resp in control_bold:
        resp = np.array(resp)
        random.seed(n)
        all_parts = list(range(len(resp)))
        parts_random = random.sample(all_parts, len(all_parts) // 2)
        parts_random1 = [p for p in all_parts if p not in parts_random]
        part1 = np.mean(resp[parts_random])
        part2 = np.mean(resp[parts_random1])
        # Check for NaN 
        if not np.isnan(part1) and not np.isinf(part1) and not np.isnan(part2) and not np.isinf(part2):
            a.append(part1)
            b.append(part2)
    if a and b:
        control_r_temp, _ = pearsonr(a, b)
        control_r_.append(control_r_temp)
control_r = np.mean(control_r_)

dict_corr = {"meco" : meco_r,
          "natstor" : natstor_r,
          "brown" :  brown_r,
          "zuco" :  zuco_r,
          "provo" : provo_r,
          "ucl_et" :  ucl_eye_r,
          "ucl_spr" :  ucl_spr_r, 
          "maze" :  maze_r,
          "ucl_n400" :  ucl_n400_r,
          "ucl_p600"  :  ucl_p600_r,
          "michaelov" :  mich_r,
          "hp" : hp_r,
          "pereira" : pereira_r, 
          "control" : control_r, 
          "natstor_fmri" : natstor_fmri_r}

for k, v in dict_corr.items():
    print(f"{k:12} {round(v, 4)}")

with open("other/dict_corr_100_mean_NEW", 'wb') as file:
    pickle.dump(dict_corr, file)

with open("other/dict_corr_100_mean_NEW", 'rb') as file:
    dict_corr = pickle.load(file)
