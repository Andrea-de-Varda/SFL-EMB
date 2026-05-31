"""
Whisper-timestamped can be installed as:
!pip3 install git+https://github.com/linto-ai/whisper-timestamped

The relevant documentation can be found at:
https://github.com/linto-ai/whisper-timestamped

Consider than, for some languages, one could use different whisper models (e.g., vasista22/whisper-tamil-large-v2)
model = whisper.load_model("vasista22/whisper-tamil-large-v2", device="cpu")

Supported languages at https://github.com/openai/whisper/blob/main/README.md 
"""

import sys
from os import chdir
import os
import whisper_timestamped as whisper
import pickle
import pandas as pd
from glob import glob
from time import sleep
import re

chdir("/home/dev/Documents/PhD/MIT/project/natstor_fmri")

def save(file, name):
    with open("transcribed/"+name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("transcribed/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def report(result, name):
    rep = []
    for segment in result["segments"]:
        for w in segment["words"]:
            rep.append([w["text"], w["start"], w["end"]])
    df = pd.DataFrame(rep, columns=["text", "start", "end"])
    df.to_csv("transcribed/"+name+".csv", index=False)
    return df

model = whisper.load_model("large", device="cpu") # whisper large

stories = [f'sound_data/{str(n)}.wav' for n in range(1, 11)]

for idx, file in enumerate(stories):
    idx = str(idx + 1)
    if os.path.isfile(f"transcribed/{idx}"):
        print(f"Story {idx} already done")
    else:
        print(f"\n\nProcessing {idx}")
        audio = whisper.load_audio(file)
        result = whisper.transcribe(model, audio, language="en")
        save(result, idx)
        df = report(result, idx)