#https://stackoverflow.com/questions/25988749/mfcc-feature-descriptors-for-audio-classification-using-librosa

#https://dsp.stackexchange.com/questions/27417/comparing-mfcc-features-what-do-they-represent

# parameters description:
# https://github.com/cournape/talkbox/blob/master/scikits/talkbox/features/mfcc.py

# another mfcc impleme3ntation:
# https://github.com/jameslyons/python_speech_features

import os
import sys

path = 'out/'

wavfiles = []
# read file list
for file in os.listdir(path):
    current = os.path.join(path, file)
    if os.path.isfile(current):
        if current.split(".")[-1]=="wav":
            wavfiles.append(current)


import numpy as np
#import scipy.io.wavfile
#from scikits.talkbox.features import mfcc

from mfcc_compare import *


# process files

mfcc_descriptors = []

for wavfile in sorted(wavfiles):

    print(wavfile)
    
    
    wav = [0,0]
    wav[0],wav[1] = librosa.load(wavfile)
    
    
    mfcc_descriptors.append((wavfile,wav))
    
    
#print(mfcc_descriptors)


# apply similarity measure

# we need to do k-means clustering there which is available in scipy

# but at the moment lets just compute matrix for L2 and cosine

number_of_records = len(mfcc_descriptors)


print
print


L2 = np.zeros((number_of_records,number_of_records))

for i in range(0,number_of_records):
    for j in range(0,number_of_records):
        L2[i,j] = compare_librosa_wav_files(mfcc_descriptors[i][1], mfcc_descriptors[j][1],'euclidean')
        
L2 = np.vstack((range(1,14),L2))

L2 = np.append(np.array(range(0,14)).reshape(14,1),L2,axis=1)
        
print("Normalized L2 distances matrix:\n")
np.set_printoptions(precision=3,suppress=True, linewidth=200)
print(L2)

