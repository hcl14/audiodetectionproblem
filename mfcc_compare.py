#https://github.com/d4r3topk/comparing-audio-files-python/blob/master/mfcc.py

#http://www.speech.zone/exercises/dtw-in-python/

#https://github.com/pierre-rouanet/dtw

#https://github.com/pierre-rouanet/dtw/blob/master/examples/MFCC%20%2B%20DTW.ipynb

from mfcc import *


def compare_wav_files(wav1,wav2,usenorm="euclidean"):


    
    filter_banks1,mfcc1,_,mfcc1_normalized,filter_banks1_normalized = mfcc(wav1)
    
    
    filter_banks2,mfcc2,_,mfcc2_normalized,filter_banks2_normalized = mfcc(wav2)
    
    
    #print(mfcc2_normalized.shape)


    # truncate signal which greatly differs in length, otherwise more long part is greatly different. Should be improved in the future, for example use moving window over longer signal
    
    if mfcc2_normalized.shape[0] > 2*mfcc1_normalized.shape[0]:
        
        l = 2*int(mfcc1_normalized.shape[0])
        
        mfcc2_normalized = mfcc2_normalized[:l,:]
        
    else:
        
        if mfcc1_normalized.shape[0] > 2*mfcc2_normalized.shape[0]:
        
            l = 2*int(mfcc2_normalized.shape[0])
        
            mfcc1_normalized = mfcc1_normalized[:l,:]
    
    
    
    if usenorm=="chisq":
        d=lambda x, y: chiSquared(x,y)
    else:
        d=usenorm
     
    
        
        
    # determine who is longer to organize sliding window
        
        
    if mfcc2_normalized.shape[0] > mfcc1_normalized.shape[0]:
        
        short = mfcc1_normalized
        longer = mfcc2_normalized
        
    else:
        
        short = mfcc2_normalized
        longer = mfcc1_normalized
        
    # move sliding window
    
    dists = []
    
    l = longer.shape[0] - short.shape[0]
    
    
    
    for i in range(0,l+1):
        
        dist, cost, acc, path = fastdtw(short, longer[i:(short.shape[0]+i),:], d)
        
        dists.append(dist)
        
        
    #dist, cost, acc, path = fastdtw(mfcc1_normalized, mfcc2_normalized, d)
    
    return numpy.min(dists)