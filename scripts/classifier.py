from keras.models import load_model
import numpy as np

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel"]	

def classify(featurelst):
    f = open("predictions.txt", "a")
    classifire_model = load_model("mel-model")
    while True:
        if (len(featurelst)>0):
            feature = featurelst.pop(0)
            label = classifire_model.predict(feature[1])

            max_index = np.argmax(label[0])
            max_prob = max(label[0])
            
            f.write("{0} -- {1}\n".format(feature[0], class_names[max_index], max_prob))
            print(feature[0],class_names[np.argmax(label)],max_prob)
        else:
            continue