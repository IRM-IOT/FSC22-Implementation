from keras.models import load_model
import numpy as np

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel"]	

def classify(featurelst):
    f = open("predictions.txt", "a")
    classifire_model = load_model("mel-model-19")
    while True:
        if (len(featurelst)>0):
            feature = featurelst.pop(0)
            label = classifire_model.predict(feature[1])
            f.write("{0} -- {1}\n".format(feature[0], class_names[np.argmax(label)]))
            print(feature[0],class_names[np.argmax(label)])
        else:
            continue