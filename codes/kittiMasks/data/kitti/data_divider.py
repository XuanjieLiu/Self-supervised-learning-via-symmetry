import pickle
from typing import List

import torch
import numpy as np


ORIGINAL_DATA_PATH = 'kitti_peds_v2.pickle'
TARGET_TRAIN_PATH = 'kitti_train.pickle'
TARGET_EVAL_PATH = 'kitti_eval.pickle'
MIN_SEQ = 12
USE_ALL = False
TEST_SET_SIZE = 320


def split_one_data(data: np.ndarray, min_seq: int):
    piece_num = int(len(data) / min_seq)
    new_data = []
    end = piece_num if USE_ALL else 1
    for i in range(end):
        new_data.append(data[i*min_seq:(i+1)*min_seq])
    return new_data


def rearrange_data(data: List, min_seq: int):
    new_data = []
    for d in data:
        if len(d) >= min_seq:
            new_data.extend(split_one_data(d, min_seq))
    return new_data


def divide_train_test():
    with open(ORIGINAL_DATA_PATH, 'rb') as data:
        data = pickle.load(data)
    pedestrians = rearrange_data(data['pedestrians'], min_seq=MIN_SEQ)
    latents = rearrange_data(data['pedestrians_latents'], min_seq=MIN_SEQ)
    train_set = {
        'pedestrians': pedestrians[TEST_SET_SIZE:],
        'pedestrians_latents': latents[TEST_SET_SIZE:],
    }
    test_set = {
        'pedestrians': pedestrians[0:TEST_SET_SIZE],
        'pedestrians_latents': latents[0:TEST_SET_SIZE],
    }
    with open(TARGET_TRAIN_PATH, 'wb') as f1:
        pickle.dump(train_set, f1)
    with open(TARGET_EVAL_PATH, 'wb') as f2:
        pickle.dump(test_set, f2)



divide_train_test()