import sounddevice as sd
import wavio as wv
import librosa
import librosa.display
import multiprocessing
from multiprocessing import Manager
import numpy as np
from keras.models import load_model

def record_audio(sharedlst):
    i=1
    while True:
        audio_name = "audios/audio-" + str(i) + ".wav"
        freq = 22050
        duration = 5
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
        sd.wait()
        wv.write(audio_name, recording, freq, sampwidth=2)
        sharedlst.append([i,recording.flatten()])
        print("audio " + str(i) + " done")
        i=i+1


def feature_extractor(audiolst,mfcclst):
    while True:
        if (len(audiolst)>0):
            raw_audio = audiolst.pop(0)

            feature_1 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio[1], sr=22050, n_mels=128, n_fft=2048, hop_length=512))
            feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio[1], sr=22050, n_mels=128, n_fft=1024, hop_length=512))
            feature_3 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio[1], sr=22050, n_mels=128, n_fft=512, hop_length=512))

            three_chanel = np.stack((feature_1, feature_2, feature_3), axis=2) 

            feature = np.expand_dims(three_chanel, axis=0)

            print(np.shape(feature))

            mfcclst.append([raw_audio[0],feature])
            print("mfcc " + str(raw_audio[0]) + " done")
        else:
            continue



def classify(mfcclst):
    # print("came here")
    classifire_model = load_model("mel-model")
    while True:
        if (len(mfcclst)>0):
            feature = mfcclst.pop(0)
            label = classifire_model.predict(feature[1])
            print(feature[0],np.argmax(label))
        else:
            continue

            
if __name__ == "__main__":

    manager = Manager()
    audio_list = manager.list()
    feature_list = manager.list()

    p1 = multiprocessing.Process(target=record_audio, args=[audio_list])
    p2 = multiprocessing.Process(target=feature_extractor, args=[audio_list,feature_list])
    p3 = multiprocessing.Process(target=classify, args=[feature_list])
    
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

