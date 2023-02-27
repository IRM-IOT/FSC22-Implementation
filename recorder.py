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
            audio = audiolst.pop(0)
            mfcc_original = librosa.core.power_to_db(librosa.feature.mfcc(y=audio[1], sr=22050, n_mfcc=40))
            mfccs_scaled_features = np.mean(mfcc_original.T,axis=0)
            mfcclst.append([audio[0],mfccs_scaled_features])
            print("mfcc " + str(audio[0]) + " done")
        else:
            continue



def classify(mfcclst):
    classifire_model = load_model("model")
    f = open("decisions.txt", "a")
    while True:
        if (len(mfcclst)>0):
            feature = mfcclst.pop(0)
            label = classifire_model.predict(feature[1])
            labell = str(label[0][0]) + "  " + str(label[0][1]) + "  " + str(label[0][2])
            f.write(str("audio-" + str(feature[0]) + " is " + labell + "\n"))
            print(feature[0],label)
        else:
            continue
    f.close()

            
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

