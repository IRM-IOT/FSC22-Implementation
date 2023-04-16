import librosa
import numpy as np
import noisereduce as nr
from scipy import signal

def f_high(y,sr):
    b,a = signal.butter(10, 2000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf

def feature_extractor(audioIndex,features):
    while True:
        if (len(audioIndex)>0):
            audio_name = audioIndex.pop(0)
            reduced_noise, sr = librosa.load(audio_name, sr=44100, mono=True, duration=5)

            # reduced_noise = nr.reduce_noise(y=normalised_audio, sr=sr, stationary=True, prop_decrease=0.8)

            feature_1 = librosa.power_to_db(librosa.feature.melspectrogram(y=reduced_noise, sr=44100, n_mels=64, n_fft=2048, hop_length=512))
            feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=reduced_noise, sr=44100, n_mels=64, n_fft=1024, hop_length=512))
            feature_3 = librosa.power_to_db(librosa.feature.melspectrogram(y=reduced_noise, sr=44100, n_mels=64, n_fft=512, hop_length=512))

            three_chanel = np.stack((feature_1, feature_2, feature_3), axis=2) 

            feature = np.expand_dims(three_chanel, axis=0)

            print(np.shape(feature))

            features.append([audio_name,feature])
            print("mel" + str(audio_name[0]) + " done")
        else:
            continue