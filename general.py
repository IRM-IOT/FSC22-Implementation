from keras.models import load_model
import numpy as np
from tflite_runtime.interpreter import Interpreter
import librosa

def feature_extractor(audio_name):
            
    normalised_audio, sr = librosa.load(audio_name, sr=22050, mono=True)

    feature_1 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
    feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=1024, hop_length=512))
    feature_3 = librosa.power_to_db(librosa.feature.melspectrogram(y=normalised_audio, sr=22050, n_mels=128, n_fft=512, hop_length=512))

    three_chanel = np.stack((feature_1, feature_2, feature_3), axis=2) 

    feature = np.expand_dims(three_chanel, axis=0)

    return(feature)


# def classify(feature):

#     classifire_model = load_model("mel-model-19")

#     label = classifire_model.predict(feature)

#     max_index = np.argmax(label[0])
#     max_prob = max(label[0])

#     print(class_names[max_index],max_prob)

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel"]	
model_path = "lite_model.tflite" 

inp = Interpreter(model_path=model_path)

print("n--------Input Details of Model-------------------n")
input_details = inp.get_input_details()
print(input_details)

print("n--------Output Details of Model-------------------n")
output_details = inp.get_output_details()
print(output_details)

# Now allocate tensors so that we can use the set_tensor() method to feed the processed_image
inp.allocate_tensors()
#print(input_details[0]['index'])
inp.set_tensor(input_details[0]['index'], feature_extractor("test_audio.wav"))

inp.invoke()
predictions = inp.get_tensor(output_details[0]['index'])[0]

print(predictions)