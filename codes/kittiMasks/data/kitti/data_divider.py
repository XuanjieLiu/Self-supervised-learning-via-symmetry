import pickle
import random
from typing import List

import torch
import numpy as np


ORIGINAL_DATA_PATH = 'kitti_peds_v2.pickle'
TARGET_TRAIN_PATH = 'kitti_train.pickle'
TARGET_EVAL_PATH = 'kitti_eval.pickle'
MIN_SEQ = 12
TEST_SET_SIZE = 320
TEST_SET_RATIO = 5
BATCH_INTEGER = 64


def split_one_data(data: np.ndarray, min_seq: int, is_use_all):
    piece_num = int(len(data) / min_seq)
    new_data = []
    end = piece_num if is_use_all else 1
    for i in range(end):
        new_data.append(data[i*min_seq:(i+1)*min_seq])
    return new_data


def rearrange_data(data: List, min_seq: int, is_use_all):
    new_data = []
    for d in data:
        if len(d) >= min_seq:
            new_data.extend(split_one_data(d, min_seq, is_use_all))
    return new_data


def load_all_data(is_use_all):
    with open(ORIGINAL_DATA_PATH, 'rb') as data:
        data = pickle.load(data)
    pedestrians = rearrange_data(data['pedestrians'], min_seq=MIN_SEQ, is_use_all=is_use_all)
    latents = rearrange_data(data['pedestrians_latents'], min_seq=MIN_SEQ, is_use_all=is_use_all)
    return pedestrians, latents


def random_split_train_test():
    pedestrians, latents = load_all_data(is_use_all=True)
    test_size = int(len(pedestrians)/TEST_SET_RATIO/BATCH_INTEGER)*BATCH_INTEGER
    test_idx_list = random.sample(list(range(len(pedestrians))), test_size)
    pedestrians_train = []
    pedestrians_test = []
    latents_train = []
    latents_test = []
    for i in range(len(pedestrians)):
        if i in test_idx_list:
            pedestrians_test.append(pedestrians[i])
            latents_test.append((latents[i]))
        else:
            pedestrians_train.append(pedestrians[i])
            latents_train.append(latents[i])
    train_set = {
        'pedestrians': pedestrians_train,
        'pedestrians_latents': latents_train,
    }
    test_set = {
        'pedestrians': pedestrians_test,
        'pedestrians_latents': latents_test,
    }
    with open(TARGET_TRAIN_PATH, 'wb') as f1:
        pickle.dump(train_set, f1)
    with open(TARGET_EVAL_PATH, 'wb') as f2:
        pickle.dump(test_set, f2)


def simple_split_train_test():
    pedestrians, latents = load_all_data(is_use_all=False)

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



simple_split_train_test()