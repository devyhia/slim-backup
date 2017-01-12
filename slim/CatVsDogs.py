import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from os import listdir, environ
import pandas as pd
from IPython import embed
import sys
import random
import time
import os

from datasets import flowers
from nets import resnet_v1
# from preprocessing import inception_preprocessing

from datasets import dataset_utils
# from helpers import *

slim = tf.contrib.slim

import argparse
import cv2
import re

import Shared

ROWS = 224
COLS = 224
CHANNELS = 3

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def get_label(path):
    return 1 if re.search("(cat|dog)\.(\d+)\.", path).group(1) == 'cat' else 0

def prepare_data(fulldata=False):
    TRAIN_DIR = '/home/devyhia/cats.vs.dogs/train/'
    TEST_DIR = '/home/devyhia/cats.vs.dogs/test/'

    SLICE = 10000

    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
    train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

    if fulldata:
        train_images = train_dogs + train_cats
    else:
        train_images = train_dogs[:SLICE] + train_cats[:SLICE]

    valid_images = train_dogs[SLICE:] + train_cats[SLICE:]

    np.random.shuffle(train_images)
    np.random.shuffle(valid_images)

    #     Bagging Code
    # test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
#     bag = int(args.bag)
#     bag_start = bag * 3125
#     bag_end = (bag+1) * 3125

#     # slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
#     train_images = train_dogs[:bag_start] + train_dogs[bag_end:] + train_cats[:bag_start] + train_cats[bag_end:]
#     valid_images = train_dogs[bag_start:bag_end] + train_cats[bag_start:bag_end]

    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

        for i, image_file in enumerate(images):
            data[i] = read_image(image_file).astype(np.float32)
            if i%250 == 0: Shared.update_screen('\rProcessed {} of {}'.format(i, count))

        Shared.update_screen('\n')

        return data

    print("Prep data ...")
    X = prep_data(train_images)
    Xt = prep_data(valid_images)

    print("Train shape: {}".format(X.shape))
    print("Valid shape: {}".format(Xt.shape))

    labels_train = [get_label(i) for i in train_images]
    labels_valid = [get_label(i) for i in valid_images]

    print(pd.DataFrame(labels_train, columns=["label"])["label"].value_counts())
    print(pd.DataFrame(labels_valid, columns=["label"])["label"].value_counts())

    y = np.zeros((X.shape[0], 2))
    yt = np.zeros((Xt.shape[0], 2))

    for i in range(y.shape[0]):
        y[i, labels_train[i]] = 1

    for i in range(yt.shape[0]):
        yt[i, labels_valid[i]] = 1

    print(y)
    print(yt)

    print("X=", X.shape, "y=", y.shape)
    print("Xt=", Xt.shape, "yt=", yt.shape)

    return X, y, Xt, yt

def prepare_test_data():
    CACHE_PATH = "cache/Xt.npy"
    if os.path.isfile(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return range(1, data.shape[0]+1), data

    TEST_DIR = '../cats.vs.dogs/test/'

    file_key = lambda f: int(re.match("(\d+)\.jpg", f).group(1))
    test_images =  [TEST_DIR+i for i in sorted(os.listdir(TEST_DIR), key=file_key)]

    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
        ids = []

        for i, image_file in enumerate(images):
            data[i] = read_image(image_file)
            ids.append(re.search("(\d+)\.", image_file).group(1))

            if i%250 == 0: Shared.update_screen('Processed {} of {}'.format(i, count))

        return ids, data

    ids, data = prep_data(test_images)

    np.save(CACHE_PATH, data)

    return ids, data
