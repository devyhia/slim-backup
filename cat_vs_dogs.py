
# coding: utf-8

# In[ ]:

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

def prepare_data():
    TRAIN_DIR = '/home/devyhia/cats.vs.dogs/train/'
    TEST_DIR = '/home/devyhia/cats.vs.dogs/test/'

    ROWS = 299
    COLS = 299
    CHANNELS = 3
    SLICE = 10000

    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
    train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

    # test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


    # slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
    train_images = train_dogs[:SLICE] + train_cats[:SLICE]
    valid_images = train_dogs[SLICE:] + train_cats[SLICE:]

    np.random.shuffle(train_images)
    np.random.shuffle(valid_images)

    def read_image(file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
        return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    
    def assign_image(data, i, image_file):
        data[i] = read_image(image_file).astype(np.float32)

    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
        
        ts = []
        for i, image_file in enumerate(images):
#             data[i] = read_image(image_file).astype(np.float32)
            t = threading.Thread(target=assign_image, args = (data,i,image_file))
            t.daemon = True
            t.start()
            ts.append(t)
            
        for i,t in enumerate(ts):
            t.join()
            update_screen('\rProcessed {} of {}'.format(i, len(ts)))
        
        update_screen('\n')

        return data

    def get_label(path):
        return 1 if re.search("(cat|dog)\.(\d+)\.", path).group(1) == 'cat' else 0
    
    print("Prep data ...")
    X = prep_data(train_images)
    Xt = prep_data(valid_images)
    # test = prep_data(test_images)
    
#     pre_processing = tf.placeholder(tf.float32, shape=[None, 299,299,3])
#     after_processing = tf.sub(tf.mul(pre_processing, 2.0/255.), 1.0)
    
#     def preprocess(sess, X, i, batch_size):
#         update_screen("\rPreprocessing {} ...".format(i))
#         return sess.run(after_processing, feed_dict={pre_processing: X[i*batch_size:(i+1)*batch_size]})
    
#     with tf.Session() as sess:
#         Xs, Xts = [], []
#         batch_size = 200
        
#         for i in range(X.shape[0] / batch_size):
#             Xs.append(preprocess(sess, X, i, batch_size))
#         update_screen("\n")
        
#         for i in range(Xt.shape[0] / batch_size):
#             Xts.append(preprocess(sess, Xt, i, batch_size))
#         update_screen("\n")
        
#         X = np.vstack(Xs)
#         Xt = np.vstack(Xts)
        
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

    # print(labels_train)
    # print(labels_valid)

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

    TEST_DIR = '/home/devyhia/cats.vs.dogs/test/'

    ROWS = 224
    COLS = 224
    CHANNELS = 3

    file_key = lambda f: int(re.match("(\d+)\.jpg", f).group(1))
    test_images =  [TEST_DIR+i for i in sorted(os.listdir(TEST_DIR), key=file_key)]

    def read_image(file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
        return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
        ids = []

        for i, image_file in enumerate(images):
            data[i] = read_image(image_file)
            ids.append(re.search("(\d+)\.", image_file).group(1))

            if i%250 == 0: update_screen('\rProcessed {} of {}'.format(i, count))
        
        update_screen('\n')
        
        return ids, data

    ids, data = prep_data(test_images)

    np.save(CACHE_PATH, data)

    return ids, data

