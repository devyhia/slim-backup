
# coding: utf-8

# In[9]:

import argparse
import cv2
import re

parser = argparse.ArgumentParser(description='RNN-CNN Network.')
parser.add_argument('--gpu', default=1, help='GPU to use for train')
parser.add_argument('--name', default="cats_vs_dogs_model", help='Name of the RNN model to use for train')
parser.add_argument('--epochs', default='30', help='Number of epochs')
parser.add_argument('--resume', default="no", help='Resume from previous checkpoint')
parser.add_argument('--test', dest='test', default=False, action='store_true', help='Generate test predictions')
parser.add_argument('--bag', default="no", help='Bagging split')
parser.add_argument('--fulldata', dest='fulldata', default=False, action='store_true', help='Train on full data?')
args, unknown_args = parser.parse_known_args()


# In[10]:

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

from nets import inception_resnet_v2
# from preprocessing import inception_preprocessing

from datasets import dataset_utils
# from helpers import *

slim = tf.contrib.slim

import CatVsDogs
import Shared


# In[11]:

Shared.DIM = 299
CatVsDogs.DIM = 299

class InceptionResnetV2:
    def __init__(self, model_name, isTesting=False):
        Shared.define_model(self, model_name, self.__model)
    
    def __get_init_fn(self):
        return Shared.get_init_fn('inception_resnet_v2_2016_08_30.ckpt', ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"])
        
    def __model(self):
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            self.logits, self.end_points = inception_resnet_v2.inception_resnet_v2(self.X_Norm, 2, is_training=True)
    
    def train(self, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
        self.init_fn = self.__get_init_fn()
        return Shared.train_model(self, sess, X, y, val_X, val_y, epochs, minibatch_size, optimizer)
        
    def load_model(self, sess):
        return Shared.load_model(self, sess)
    
    def predict_proba(self, sess, X, step=10):
        return Shared.predict_proba(self, sess, X, step)


# In[ ]:

Shared.select_gpu(args.gpu)

resnet = InceptionResnetV2(args.name)

sess = tf.Session()


# In[ ]:

def test():
    print("+++++ TESTING +++++")
    resnet.load_model(sess)
    ids, Xt = CatVsDogs.prepare_test_data()
    ids = np.array(ids).astype(np.int)

    prob = resnet.predict_proba(sess, Xt, step=50)
    
    np.save("CatVsDogs.Xt.{}.npy".format(args.name), prob)


# In[ ]:

def train():    
    print("+++++ TRAINING +++++")
    X, y, Xt, yt = CatVsDogs.prepare_data(fulldata=args.fulldata)
    resnet.train(sess, X, y, Xt, yt, epochs=int(args.epochs), minibatch_size=10)


# In[ ]:

if not args.test:
    train()
else:
    test()

