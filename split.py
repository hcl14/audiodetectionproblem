import pydub
import statistics
import numpy as np
import os,shutil


def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)


def get_array_of_samples(self):
    """
    returns the raw_data as an array of samples
    """
    return array.array(self.array_type, self._data)





def split(path):

    # clean working directory

    shutil.rmtree('out')
    os.mkdir('out')


    sound_file = pydub.AudioSegment.from_wav(path)

    #sound_file = pydub.AudioSegment.from_wav("wav/A_0_Restricted_302593674_12_Fra_21112012_191255-in_noise_reduction_attempt2.wav")
    #sound_file = pydub.AudioSegment.from_wav("wav/A_726_7072305480_133643004_14_Eng_13122016_165119-in_noise_reduction_attempt2.wav")
    #sound_file = pydub.AudioSegment.from_wav("wav/A_733_7072305480_060498941_14_Eng_14122016_155847-in.wav")



    #increase sound if too quiet
    #sound_file = match_target_amplitude(sound_file, -20)
    
    
    

    '''
    loudness = sound.dBFS
    if you are looking for loudness for certain portion of a file, then you will need to split audio segment in chunks and then check loudness for each chunk.
    '''

    # let's detect noise level using median value:

    duration = len(sound_file)
    numchunks = int(duration/20) #(ms)

    loudness = []
    for i in range(0,numchunks):
        loudness.append(sound_file[20*i : 20*(i+1)].dBFS)  #we ignore last 20ms here

    loudness = [x for x in loudness if np.isfinite(x)]

    threshold =  statistics.median(loudness)*0.8
    print('Noise level supposed to be (<-50 is converted to 50):' + str(threshold))

    #threshold = max([-50,threshold])



    #silence_thresh - (in dBFS) anything quieter than this will be
    #considered silence. default: -16dBFS
    #studio value is -40 usually
    nonsilent_ranges = pydub.silence.detect_nonsilent(sound_file,min_silence_len=1000, silence_thresh=threshold )#-50)

    print(nonsilent_ranges)


    #compute average amplitudes
    loudness = []
    for x, y in nonsilent_ranges:
        
        amplitude = sound_file[x : y].dBFS
        
        amplitude = max([-50,amplitude])
        
        loudness.append(amplitude)
        
    average_amplitude = np.mean(loudness)

    print("average amplitude to be applied: "+ str(average_amplitude))



    
    count = 0

    for x, y in nonsilent_ranges:
        
        count += 1
        
        new_file = sound_file[x : y]
        
        # normalize amplitudes
        new_file = match_target_amplitude(new_file, average_amplitude)
        
        str_count = str(count)
        str_count = str_count.zfill(2)
        
        new_file.export("out/" +str_count +"-"+str(x) + "-" + str(y) +".wav", format="wav")

