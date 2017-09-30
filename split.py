import pydub
import statistics

#sound_file = pydub.AudioSegment.from_wav("wav/A_0_Restricted_302593674_12_Fra_21112012_191255-in_noise_reduction_attempt1.wav")
sound_file = pydub.AudioSegment.from_wav("wav/A_726_7072305480_133643004_14_Eng_13122016_165119-in.wav")



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

threshold =  statistics.median(loudness)*0.8
print('Noise level supposed to be:' + str(threshold))





#silence_thresh - (in dBFS) anything quieter than this will be
#considered silence. default: -16dBFS
#studio value is -40 usually
nonsilent_ranges = pydub.silence.detect_nonsilent(sound_file,min_silence_len=1000, silence_thresh=threshold )#-50)

print(nonsilent_ranges)

count = 0

for x, y in nonsilent_ranges:
    
    count += 1
    
    new_file = sound_file[x : y]
    
    str_count = str(count)
    str_count = str_count.zfill(2)
    
    new_file.export("out/" +str_count +"-"+str(x) + "-" + str(y) +".wav", format="wav")

