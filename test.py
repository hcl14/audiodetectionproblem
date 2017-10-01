#https://stackoverflow.com/questions/25988749/mfcc-feature-descriptors-for-audio-classification-using-librosa

#https://dsp.stackexchange.com/questions/27417/comparing-mfcc-features-what-do-they-represent

# parameters description:
# https://github.com/cournape/talkbox/blob/master/scikits/talkbox/features/mfcc.py

# another mfcc impleme3ntation:
# https://github.com/jameslyons/python_speech_features

import os
import sys


from split import split




path = "wav/A_0_Restricted_302593674_12_Fra_21112012_191255-in_noise_reduction_attempt2.wav"

split(path)


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
    #wav[0],wav[1] = librosa.load(wavfile)
    wav = read_wav(wavfile)
    
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
        L2[i,j] = compare_wav_files(mfcc_descriptors[i][1], mfcc_descriptors[j][1],'correlation')  # cosine and correlation are good
        

l2r = L2.ravel()
#diffs = [abs(i-j) for i in l2r for j in l2r if i != j]  #distances
#diffs = np.sort(diffs)


# find approximate starting point for eps, but that's innacurate - the supposition is that variances of 'yes' and 'no' are equal
#eps = np.max(diffs)/2

# better to operate distances to 3 nearest neighbours : http://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf


l2r = L2.ravel()

def three_nearest_neighbour_distances(i, l2r):
    
    dists = [abs(l2r[i]-l2r[j]) for j in range(len(l2r)) if i != j]
    
    dists = np.sort(dists)
    
    return dists[int(len(l2r)/4)], dists[3*int(len(l2r)/8)]   # we think that between quarter and half of the points belong to the one cluster: wew need to scan area between 1/4 and 1/2 (e.g. 3/8)


k3n_d1 = []
k3n_d2 = []


for i in range(0,len(l2r)):
    
        d1, d2 = three_nearest_neighbour_distances(i, l2r)
    
        k3n_d1.append(d1)
        
        k3n_d2.append(d2)
        
        
eps1 = np.max(k3n_d1)
eps2 = np.max(k3n_d2)
    

clusters1 = cluster(L2,eps1)
clusters2 = cluster(L2,eps2)

# let's choose the clustering which has less elements of a third cluster (-1 is outlier, let's count it as a simpler problem)

num_outliers1 = sum([x==-1 for x in clusters1])
num_outliers2 = sum([x==-1 for x in clusters2])

if num_outliers1<num_outliers2:
    eps = eps1
    clusters = clusters1
else:
    eps = eps2
    clusters = clusters2

        
L2 = np.vstack((range(1,15),L2))

L2 = np.append(np.array(range(0,15)).reshape(15,1),L2,axis=1)
        
print("Normalized L2 distances matrix:\n")
np.set_printoptions(precision=3,suppress=True, linewidth=200)
print(L2)

print('supposed eps: '+str(eps))
print('clusters:')
print(clusters)



num_answers = len([x for x in clusters if x != -1])
print(num_answers)


