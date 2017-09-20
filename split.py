import pydub

sound_file = pydub.AudioSegment.from_wav("wav/A_0_Restricted_302593674_12_Fra_21112012_191255-in.wav")

#silence_thresh - (in dBFS) anything quieter than this will be
#considered silence. default: -16dBFS
#studio value is -40 usually
nonsilent_ranges = pydub.silence.detect_nonsilent(sound_file,min_silence_len=1000, silence_thresh=-40)

print(nonsilent_ranges)


for x, y in nonsilent_ranges:
    new_file = sound_file[x : y]
    new_file.export("out/" + str(x) + "-" + str(y) +".wav", format="wav")

