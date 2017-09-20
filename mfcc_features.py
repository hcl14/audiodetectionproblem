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
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

# process files

mfcc_descriptors = []

for wavfile in sorted(wavfiles):

    print(wavfile)
    
    sample_rate, X = scipy.io.wavfile.read(wavfile)
    
    try:
        # raises error on very short records like noise
        #ceps, mspec, spec = mfcc(X,nwin=256, nfft=512, fs=16000, nceps=13)
        
        ceps, mspec, spec = mfcc(X,nwin=256, nfft=512, fs=16000, nceps=13)
        
    except:
        print('record too short, don\'t know what to do')
        #continue
        ceps = []
    
    
    #np.save("cache_file_name", ceps) # cache results so that ML becomes fast

    '''
    #Then while doing ML, do something like:

    X = []
    ceps = np.load("cache_file_name")
    num_ceps = len(ceps)
    X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    '''
    X = []
    num_ceps = len(ceps)
    if num_ceps>0:
        X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    else:
        X = np.zeros((1,13)) #nceps

    Vx = np.array(X)
    # use Vx as input values vector for neural net, k-means, etc
    
    print(Vx)
    
    mfcc_descriptors.append((wavfile,Vx))
    
    
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
        L2[i,j] = np.linalg.norm(mfcc_descriptors[i][1] - mfcc_descriptors[j][1])
        
print("L2 distances matrix:\n")
np.set_printoptions(precision=2,linewidth=200)
print(L2)


print
print

cos_dist = np.zeros((number_of_records,number_of_records))

for i in range(0,number_of_records):
    for j in range(0,number_of_records):
        cos_dist[i,j] = scipy.spatial.distance.cosine(mfcc_descriptors[i][1], mfcc_descriptors[j][1])
        
print("Cosine distances matrix:\n")
np.set_printoptions(precision=2,linewidth=200)
print(cos_dist)
