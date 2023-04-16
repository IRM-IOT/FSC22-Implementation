import multiprocessing
from multiprocessing import Manager

from scripts.extractor import feature_extractor
from scripts.classifier import classify
from scripts.setup import setup
from scripts.recorder import record_audio


if __name__ == "__main__":

    # setup()

    manager = Manager()
    audioIndex = manager.list()
    features = manager.list()

    p1 = multiprocessing.Process(target=record_audio, args=[audioIndex])
    p3 = multiprocessing.Process(target=feature_extractor, args=[audioIndex,features])
    p4 = multiprocessing.Process(target=classify, args=[features])
    
    p1.start()
    p3.start()
    p4.start()

    p1.join()
    p3.join()
    p4.join()

