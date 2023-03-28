import sounddevice as sd
import wavio as wv
import librosa

import configparser

config = configparser.ConfigParser()
config.read("config.ini")
base_loudness = config.get('Env Settings', 'Loudness')

def record_audio(sharedlst):
    i=1
    while True:
        audio_name = "audios/audio-" + str(i) + ".wav"
        freq = 22050
        duration = 5
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
        sd.wait()

        clip_rms = librosa.feature.rms(y=recording,hop_length=512)
        clip_rms = clip_rms.squeeze()
        peak_loudness = clip_rms.argmax()

        if ((float(peak_loudness)/float(base_loudness))>1):
            wv.write(audio_name, recording, freq, sampwidth=2)
            sharedlst.append(audio_name)
            print("audio " + str(i) + " done")
        else:
            print("skipping due to low loudness levels for audio :",i)
            
        i=i+1