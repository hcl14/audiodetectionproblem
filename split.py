import pydub

#sound_file = pydub.AudioSegment.from_wav("wav/A_0_Restricted_302593674_12_Fra_21112012_191255-in_noise_reduction_attempt1.wav")
sound_file = pydub.AudioSegment.from_wav("wav/A_726_7072305480_133643004_14_Eng_13122016_165119-in_noise_reduction_attempt2.wav")

#silence_thresh - (in dBFS) anything quieter than this will be
#considered silence. default: -16dBFS
#studio value is -40 usually
nonsilent_ranges = pydub.silence.detect_nonsilent(sound_file,min_silence_len=1000, silence_thresh=-50)

print(nonsilent_ranges)

count = 0

for x, y in nonsilent_ranges:
    
    count += 1
    
    new_file = sound_file[x : y]
    
    str_count = str(count)
    str_count = str_count.zfill(2)
    
    new_file.export("out/" +str_count +"-"+str(x) + "-" + str(y) +".wav", format="wav")

