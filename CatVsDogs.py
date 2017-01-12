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

DIM= 224
CHANNELS= 3
TRAIN_DIR = '/home/devyhia/cats.vs.dogs/train/'
TEST_DIR = '/home/devyhia/cats.vs.dogs/test/'
SLICE = 10000

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (DIM, DIM), interpolation=cv2.INTER_CUBIC)

def get_label(path):
    return 1 if re.search("(cat|dog)\.(\d+)\.", path).group(1) == 'cat' else 0

def prepare_data(fulldata=False):
    print("++++ LOADING/CACHING DATASET ++++")
    # Load Cached Data
    data = prepare_training_data_for_testing()

    # Create Index & Labels
    N = data.shape[0]
    index_dogs = np.arange(N)[:N/2]
    index_cats = np.arange(N)[N/2:]
    labels = np.array([[1,0]]*(N/2) + [[0,1]]*(N/2)) # Dogs (0) --> Cats (1)

    # Shuffle the data/labels
    np.random.shuffle(index_dogs)
    np.random.shuffle(index_cats)

    # Training/Validation Indices
    if fulldata:
        train_index = np.hstack((index_dogs, index_cats))
    else:
        train_index = np.hstack((index_dogs[:SLICE], index_cats[:SLICE]))

    valid_index = np.hstack((index_dogs[SLICE:], index_cats[SLICE:]))

    print(train_index.shape, valid_index.shape)

    # Split Training/Validation Sets
    X = data[train_index]
    y = labels[train_index]
    Xt = data[valid_index]
    yt = labels[valid_index]

    print("Train shape: {}".format(X.shape))
    print("Valid shape: {}".format(Xt.shape))

    print(pd.DataFrame(y.argmax(axis=1), columns=["label"])["label"].value_counts())
    print(pd.DataFrame(yt.argmax(axis=1), columns=["label"])["label"].value_counts())

    print("X=", X.shape, "y=", y.shape)
    print("Xt=", Xt.shape, "yt=", yt.shape)

    return X, y, Xt, yt

def prepare_test_data():
    CACHE_PATH = "cache/Xt.{}.npy".format(DIM)
    if os.path.isfile(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return range(1, data.shape[0]+1), data

    file_key = lambda f: int(re.match("(\d+)\.jpg", f).group(1))
    test_images =  [TEST_DIR+i for i in sorted(os.listdir(TEST_DIR), key=file_key)]

    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, DIM, DIM, CHANNELS), dtype=np.uint8)
        ids = []

        for i, image_file in enumerate(images):
            data[i] = read_image(image_file)
            ids.append(re.search("(\d+)\.", image_file).group(1))

            if i%250 == 0: Shared.update_screen('\rProcessed {} of {}'.format(i, count))

        Shared.update_screen('\n')
        return ids, data

    ids, data = prep_data(test_images)

    np.save(CACHE_PATH, data)

    return ids, data

def prepare_training_data_for_testing():
    CACHE_PATH = "cache/X.{}.npy".format(DIM)
    if os.path.isfile(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return data

    file_key = lambda f: int(re.match(".+\.(\d+)\.jpg", f).group(1))

    test_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
    test_dogs =   [TRAIN_DIR+i for i in sorted(os.listdir(TRAIN_DIR), key=file_key) if 'dog' in i]
    test_cats =   [TRAIN_DIR+i for i in sorted(os.listdir(TRAIN_DIR), key=file_key) if 'cat' in i]

    test_images =  test_dogs + test_cats

    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, DIM, DIM, CHANNELS), dtype=np.uint8)

        for i, image_file in enumerate(images):
            data[i] = read_image(image_file)

            if i%250 == 0: Shared.update_screen('\rProcessed {} of {}'.format(i, count))

        Shared.update_screen('\n')
        return data

    data = prep_data(test_images)

    np.save(CACHE_PATH, data)

    return data
