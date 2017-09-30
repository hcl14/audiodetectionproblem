#https://github.com/d4r3topk/comparing-audio-files-python/blob/master/mfcc.py

#http://www.speech.zone/exercises/dtw-in-python/

#https://github.com/pierre-rouanet/dtw

#https://github.com/pierre-rouanet/dtw/blob/master/examples/MFCC%20%2B%20DTW.ipynb

from mfcc import *


def compare_wav_files(wav1,wav2,usenorm="euclidean"):

    #Loading audio files
    y1, sr1 = wav1[0],wav1[1] #librosa.load('out/1-6988-7076.wav') 
    y2, sr2 = wav2[0],wav2[1] #librosa.load('out/2-14068-14168.wav') 

    
    filter_banks1,mfcc1,_ = mfcc(wav1)
    
    
    filter_banks2,mfcc2,_ = mfcc(wav2)

    if usenorm=="chisq":
        d=lambda x, y: chiSquared(x,y)
    else:
        d=usenorm
        
    dist, cost, acc, path = fastdtw(mfcc1, mfcc2, d)
    
    return dist