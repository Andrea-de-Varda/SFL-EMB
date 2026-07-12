This is the data from:
Frank, S.L., Otten, L.J., Galli, G., & Vigliocco, G. The ERP response to the amount of information conveyed by words in sentences. Brain and Language. DOI: 10.1016/j.bandl.2014.10.006

Please cite this paper in any work based on the current data set.


Stimuli data, information measures, and averaged ERPs are stored in the Matlab data file 'stimuli_erp.mat'.
The explanation of variables below makes use of the following indices:
    p : participant ID-number
    s : sentence ID-number 
    w : word position within sentence

-----------------------
Stimuli data
-----------------------

sentences{s}            : words of stimulus sentence s                        
logwordfreq{s}(w)       : log of the word's relative frequency in BNC
wordlength{s}(w)        : word length (number of characters including any punctuation)

presentation_order(:,p) : sentences in their order of presentation to participant p
sentence_position(p,s)  : position of sentence s in the experimental session of participant p

SOAs{s}(w,p)            : True SOA in ms. This is the difference between the onset timestamps of words w+1 and w.
                          Hence, there is no SOA for sentence-final words. There are three trials with overly long SOAs:
                               --------------------------
                               Subj  Sent  Word  SOA (ms)
                               --------------------------               
                                08    47     1    1870                          
                                10     1     3    3111                         
                                17   126     1    1784
                               --------------------------

-------------------------
Information measures
-------------------------

surp_ngram{s}(w,n)     : word surprisal under the (n+1)-gram model trained on the 1.1M selected BNC sentences
surp_ngramfull{s}(w,n) : word surprisal under the (n+1)-gram model trained on the full written BNC
surp_rnn{s}(w,e)       : word surprisal under the RNN model. Index e refers to the size of the training data subset,
                         where e==10 means that the full data set was presented twice.
surp_psg{s}(w,e)       : word surprisal under the PSG model. Index e refers to the size of the training data subset. 
                         
surp_pos_ngram{s}(w,n) : part-of-speech surprisal under the (n+1)-gram model
surp_pos_rnn{s}(w,e)   : part-of-speech surprisal under the RNN model
surp_pos_psg{s}(w,e)   : part-of-speech surprisal under the PSG model

dH{s}(w,k,e)           : entropy reduction value over upcoming k words, under the RNN model
dH_pos{s}(w,k,e)       : entropy reduction value over upcoming k parts-of-speech, under the RNN model


-----------------------
ERP data 
-----------------------

ERP{s}(w,p,:)     : averaged ERP for six components, respectively: ELAN, LAN, N400, EPNP, P600, PNP.
                    This is after applying a 0.50Hz high-pass filter and the modulus transformation.
ERPbase{s}(w,p,:) : averaged potential over the 100ms leading up to word onset

artefact{s}(w,p)  : is 1 iff data on this trial is considered an artefact
reject{s}(w,p)    : is 1 iff data on this trial was removed from analysis for any reason


--------------
Raw EEG data
--------------

Raw EEG data for participant p is stored in the Matlab data file 'EEG<p>.mat'. This contains a single structure variable
called 'EEG' in EEGlab format. The variable contains the following relevant fields:

EEG.chanlocs(i).labels            : electrode ID of channel i, following montage M10 (see www.easycap.de)
EEG.chanlocs(i).sph_theta/sph_phi : spherical theta/phi-coordinates of the electrode site

EEG.data(i,:) : The continuous signal from channel i. Data is present from 100ms before onset of each sentence-intial word
                up to 924 after onset of each sentence-final word. Data in between sentence presentations is set to NaN.
                The data has been band-pass filtered at 0.05Hz - 25Hz, recalibrated, and rereferenced to the average of left
                and right mastoid electrodes (only left mastoid for Participant 9 because of recording error on the right mastoid).

EEG.event(n) contains information about the n-th word presentation event, where:

EEG.event(n).latency : Sample point (at 4ms resolution) of EEG.data at which the n-th word appeared
EEG.event(n).type    : Indicates which word was presented. If the value is larger than 50, then the word was the first word of
                       the sentence with ID number EEG.event(n).type-50 (e.g., type 51 is the first word of sentence 1).
                       If the value is between 2 and 15, then it identifies the word position within the sentence (e.g., type 2
                       is the second word of the current sentence).

There were 1931 trials (word tokens) per subject, but for five participants not all data was recorded. The number of recorded trials
for these participants are:
    --------------------
    Participant  #trials 
    --------------------
         6        1832   
         9        1916   
        13        1917   
        14        1901   
        20        1880   
    --------------------

Recording was interrupted for Participant 6. Consequently, EEG06.mat holds a second data variable: 'EEG_part2'.

