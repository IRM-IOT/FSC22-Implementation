import multiprocessing
from multiprocessing import Manager

from scripts.recorder import record_audio
from scripts.extractor import feature_extractor
from scripts.classifier import classify
from scripts.setup import setup
from scripts.loader import record_audio


if __name__ == "__main__":

    # setup()

    manager = Manager()
    audio_names = manager.list()
    feature_list = manager.list()

    p1 = multiprocessing.Process(target=record_audio, args=[audio_names])
    p3 = multiprocessing.Process(target=feature_extractor, args=[audio_names,feature_list])
    p4 = multiprocessing.Process(target=classify, args=[feature_list])
    
    p1.start()
    p3.start()
    p4.start()

    p1.join()
    p3.join()
    p4.join()

