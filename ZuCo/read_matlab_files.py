import os
from os import chdir
import numpy as np
import h5py
import sys
from tqdm import tqdm
import pandas as pd

dirName = "/home/dev/Documents/PhD/MIT/project/ZuCo"
chdir(dirName)

import data_loading_helpers as dh


task = "NR"

rootdir = "/home/dev/Documents/Datasets/ZuCo/Matlab files/"

out = []

for file in tqdm(os.listdir(rootdir)):
    if file.endswith(task+".mat"):
        #print(file)

        file_name = rootdir + file
        subject = file_name.split("ts")[1].split("_")[0]

        # exclude YMH due to incomplete data because of dyslexia
        if subject != 'YMH':

            f = h5py.File(file_name)
            sentence_data = f['sentenceData']
            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            omissionR = sentence_data['omissionRate']
            wordData = sentence_data['word']
            
            for idx in range(len(rawData)):
                obj_reference_content = contentData[idx][0]
                sent = dh.load_matlab_string(f[obj_reference_content])

                # get omission rate
                obj_reference_omr = omissionR[idx][0]
                omr = np.array(f[obj_reference_omr])
                #print(omr)

                # get word level data
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]])

                # number of tokens
                # print(len(word_data))

                for key, item in word_data.items():
                    gd = item["GD"]
                    ff = item["FFD"]
                    gopast = item["GPT"]
                    tt = item["TRT"]
                    word = item["content"]
                    word_idx = item["word_idx"]
                    out.append([subject, idx, word_idx, word, ff, gd, gopast, tt])

out = pd.DataFrame(out, columns=["subj", "idx", "word_idx", "word", "ff", "gd", "gopast", "tt"])
out["unique"] = out["idx"].astype(str)+"_"+out["word_idx"].astype(str)

#out.to_csv("/home/dev/Documents/Datasets/ZuCo/ET_data.csv")

out_mean = out.groupby(["idx", "word_idx"], as_index=False, sort=False).agg({'ff':"mean", 'gd':"mean", 'gopast':"mean", 'tt':"mean", "word":"max", 'word_idx':"max",'idx':"max"})
out_mean.to_csv("/home/dev/Documents/Datasets/ZuCo/ET_data_aggregated.csv")

sentences = []
for sent_idx in set(out_mean["idx"]):
    temp = out_mean[out_mean["idx"] == sent_idx]
    words = " ".join(temp.word)
    sentences.append(words)
    
with open("sentences.txt", 'w') as fp:
    for item in list(sentences.sentence):
        fp.write("%s\n" % item)
    print('Done')

