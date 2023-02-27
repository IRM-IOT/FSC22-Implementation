import sounddevice as sd
import wavio as wv

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

