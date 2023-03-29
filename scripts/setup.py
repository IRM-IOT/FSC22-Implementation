import sounddevice as sd
import librosa
import configparser

def record_audio():
    peak_rms_lst = []
    i=1
    while i<10:
        freq = 22050
        duration = 5
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
        sd.wait()

        clip_rms = librosa.feature.rms(y=recording,hop_length=512)
        clip_rms = clip_rms.squeeze()
        peak_rms_lst.append(clip_rms.argmax())

        print("audio " + str(i) + " done")
        i=i+1

    return (sum(peak_rms_lst) / len(peak_rms_lst))

def write_to_config(config_file, section, option, value):
    config = configparser.ConfigParser()
    config[section] = {}
    config[section][option] = str(value)
    with open(config_file, 'w') as configfile:
        config.write(configfile)

def setup():
    print("Setting up started")
    avg_peak_loudness = record_audio()
    print("Average loudness is ", avg_peak_loudness)
    write_to_config('config.ini', 'Env Settings', 'Loudness', avg_peak_loudness)
    print("Setup completed !!!")

