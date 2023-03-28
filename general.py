from keras.models import load_model
import numpy as np
import librosa

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel"]	

def feature_extractor(audio_name):
            
    normalised_audio, sr = librosa.load(audio_name, sr=22050, mono=True)

    feature_1 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
    feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=1024, hop_length=512))
    feature_3 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=512, hop_length=512))

    three_chanel = np.stack((feature_1, feature_2, feature_3), axis=2) 

    feature = np.expand_dims(three_chanel, axis=0)

    return(feature)


def classify(feature):

    classifire_model = load_model("mel-model-19")

    label = classifire_model.predict(feature)

    max_index = np.argmax(label[0])
    max_prob = max(label[0])

    print(class_names[max_index],max_prob)

classify(feature_extractor("test_audio.wav"))