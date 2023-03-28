import librosa
import numpy as np

def feature_extractor(audiolst,mfcclst):
    while True:
        if (len(audiolst)>0):
            audio_name = audiolst.pop(0)
            normalised_audio, sr = librosa.load(audio_name, sr=22050, mono=True)

            feature_1 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
            feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=1024, hop_length=512))
            feature_3 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=512, hop_length=512))

            three_chanel = np.stack((feature_1, feature_2, feature_3), axis=2) 

            feature = np.expand_dims(three_chanel, axis=0)

            print(np.shape(feature))

            mfcclst.append([audio_name,feature])
            print("mel" + str(audio_name[0]) + " done")
        else:
            continue