#https://github.com/d4r3topk/comparing-audio-files-python/blob/master/mfcc.py

#http://www.speech.zone/exercises/dtw-in-python/

#https://github.com/pierre-rouanet/dtw

#https://github.com/pierre-rouanet/dtw/blob/master/examples/MFCC%20%2B%20DTW.ipynb

import librosa
import librosa.display
import matplotlib.pyplot as plt
#from dtw import dtw
from numpy.linalg import norm
import scipy

#************************************************************************
#https://github.com/pierre-rouanet/dtw/blob/master/dtw.py
from numpy import array, zeros, argmin, inf, equal, ndim
import numpy as np
from scipy.spatial.distance import cdist

def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def fastdtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

#************************************************************************

# best distance for histograms
def chiSquared(p,q):
    return 0.5*np.sum((p-q)**2/(p+q+1e-6))



def compare_librosa_wav_files(wav1,wav2,usenorm="euclidean"):

    #Loading audio files
    y1, sr1 = wav1[0],wav1[1] #librosa.load('out/1-6988-7076.wav') 
    y2, sr2 = wav2[0],wav2[1] #librosa.load('out/2-14068-14168.wav') 

    #Showing multiple plots using subplot
    #plt.subplot(1, 2, 1) 
    
    # parameters
    #https://stackoverflow.com/questions/37963042/python-librosa-what-is-the-default-frame-size-used-to-compute-the-mfcc-feature
    
    mfcc1 = librosa.feature.mfcc(y1,sr1, n_fft=512, hop_length=128, n_mfcc=20)   #Computing MFCC values
    
    #print(np.array(mfcc1).shape)
    
    #librosa.display.specshow(mfcc1)

    #plt.subplot(1, 2, 2)
    mfcc2 = librosa.feature.mfcc(y2, sr2, n_fft=512, hop_length=128, n_mfcc=20)
    #librosa.display.specshow(mfcc2)

    if usenorm=="chisq":
        d=lambda x, y: chiSquared(x,y)
    else:
        d=usenorm
        
    dist, cost, acc, path = fastdtw(mfcc1.T, mfcc2.T, d)
    #print("The normalized distance between the two : ",dist)   # 0 for similar audios 

    #plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
    #plt.plot(path[0], path[1], 'w')   #creating plot for DTW

    #plt.show() #To display the plots graphically 
    return dist