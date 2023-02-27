from keras.models import load_model
import numpy as np

def classify(mfcclst):
    f = open("predictions.txt", "a")
    classifire_model = load_model("mel-model")
    while True:
        if (len(mfcclst)>0):
            feature = mfcclst.pop(0)
            label = classifire_model.predict(feature[1])
            f.write("{0} -- {1}\n".format(feature[0], np.argmax(label)))
            print(feature[0],np.argmax(label))
        else:
            continue