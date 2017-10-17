import argparse
import cv2
import re
import threading
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
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import pandas as pd

# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

# Set Constants
DIR = '/home/devyhia/distracted.driver/'
ROWS = 299
COLS = 299
CHANNELS = 3
SLICE = 10000
SPLIT = 0.75

# Set up classes
CLASSES = [
    "Drive Safe",
    "Text Left",
    "Talk Left",
    "Text Right",
    "Talk Right",
    "Adjust Radio",
    "Drink",
    "Hair & Makeup",
    "Reach Behind",
    "Talk Passenger"
]

def showProgress(iterable, desc, total, progressBar):
    if progressBar:
        return tqdm(iterable, desc=desc, total=total)
    else:
        return iterable

# Load Training & Testing Images into Memory
def read_image(payload):
    idx, file_path = payload
    img = imread(file_path)
    return idx, cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC).astype(np.float32)

def save_cache(X, y, Xt, yt, which='original'):
    np.save("cache/X.{}.npy".format(which), X)
    np.save("cache/y.{}.npy".format(which), y)
    np.save("cache/Xt.{}.npy".format(which), Xt)
    np.save("cache/yt.{}.npy".format(which), yt)

def load_cache(data=[ "X", "y", "Xt", "yt" ]):
    return tuple([np.load("cache/{}.npy".format(f)) for f in data])

def load_data(progressBar=False, which='original'):
    # Load from Cache (if exists)
    data = [
        "X.{}".format(which),
        "y.{}".format(which),
        "Xt.{}".format(which),
        "yt.{}".format(which)
    ]

    if reduce(lambda prev, curr: prev and os.path.isfile("cache/{}.npy".format(curr)), data):
        return load_cache(data)

    # Load Files Data
    # Data = {}
    # for idx, klass in enumerate(CLASSES):
    #     files = glob("{}{}/*.{}.jpg".format(DIR, klass, which))
    #     Data[klass] = {
    #         "id": idx,
    #         "count": len(files),
    #         "files": files
    #     }
    #
    # # Split Files Data into training and testing
    # train_images = []
    # test_images = []
    # train_labels = []
    # test_labels = []
    # for klass in CLASSES:
    #     files = Data[klass]['files']
    #     klass_id = Data[klass]['id']
    #     np.random.shuffle(files)
    #     THRES = int(SPLIT * len(files))
    #     train_images += files[:THRES]
    #     test_images += files[THRES:]
    #     train_labels += len(files[:THRES]) * [ klass_id ]
    #     test_labels += len(files[THRES:]) * [ klass_id ]
    #
    # train_count = len(train_images)
    # test_count = len(test_images)

    dfTrain = pd.read_csv(DIR+'train.csv')
    dfTest = pd.read_csv(DIR+'test.csv')

    train_images = dfTrain.Image.tolist()
    train_labels = dfTrain.Label.tolist()
    train_count = dfTrain.Image.count()

    test_images = dfTest.Image.tolist()
    test_labels = dfTest.Label.tolist()
    test_count = dfTest.Image.count()

    format_images_path = lambda x: x.replace('.jpg', '.{}.jpg'.format(which))
    train_images = map(format_images_path, train_images)
    test_images = map(format_images_path, test_images)

    train_data = np.ndarray((train_count, ROWS, COLS, CHANNELS), dtype=np.uint8)
    test_data = np.ndarray((test_count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    # Multiprocessing-based Data Load
    p = Pool()

    # Process Training Images

    for idx, image in showProgress(p.imap_unordered(read_image, enumerate(train_images), chunksize=16), "Training Images", train_count, progressBar):
        train_data[idx] = image

    # Process Testing Images
    for idx, image in showProgress(p.imap_unordered(read_image, enumerate(test_images), chunksize=16), "Testing Images", test_count, progressBar):
        test_data[idx] = image

    X = train_data
    Xt = test_data

    y = np.eye(10)[train_labels]
    yt = np.eye(10)[test_labels]

    save_cache(X, y, Xt, yt, which=which)

    return X, y, Xt, yt

# def load_segmented_data(progressBar=False):
#     # Load from Cache (if exists)
#     data = [ "X.segmented", "y", "Xt.segmented", "yt" ]
#     if reduce(lambda prev, curr: prev and os.path.isfile("cache/{}.npy".format(curr)), data):
#         return load_cache(data)
#     else:
#         raise "You need to segment the data first ..."
