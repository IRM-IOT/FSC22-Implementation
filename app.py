import multiprocessing
from multiprocessing import Manager

from recorder import record_audio
from extractor import feature_extractor
from classifier import classify
from setup import setup
from loader import audio_extractor


if __name__ == "__main__":

    # setup()

    manager = Manager()
    frames_list = manager.list()
    audio_list = manager.list()
    feature_list = manager.list()

    p1 = multiprocessing.Process(target=record_audio, args=[frames_list])
    p2 = multiprocessing.Process(target=audio_extractor, args=[frames_list,audio_list])
    # p3 = multiprocessing.Process(target=feature_extractor, args=[audio_list,feature_list])
    # p4 = multiprocessing.Process(target=classify, args=[feature_list])
    
    p1.start()
    p2.start()
    # p3.start()
    # p4.start()

    p1.join()
    p2.join()
    # p3.join()
    # p4.join()

