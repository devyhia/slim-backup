
# coding: utf-8

# In[1]:

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
import CatVsDogs

import importlib

import pandas as pd


# In[2]:

CatVsDogs.DIM = 299
Shared.DIM = 299


# In[3]:

Xt = CatVsDogs.prepare_training_data_for_testing()


# In[4]:

# X, y, Xt, yt = CatVsDogs.prepare_data()


# In[5]:

inceptionv3 = importlib.import_module("InceptionV3 - Cat vs Dogs")


# In[6]:

inceptionv3.args.depth = ['256']


# In[7]:

Shared.select_gpu(3)
model = inceptionv3.InceptionV3("cvd_inception_v3_depth_256")
sess = tf.Session()


# In[8]:

model.load_model(sess)


# In[9]:

# model.tf_summary = tf.merge_summary(model.summaries)
# model.tf_logs = tf.train.SummaryWriter("logs/{}".format(model.name), sess.graph, flush_secs=30)


# In[12]:

prob = model.predict_proba(sess, np.vstack((Xt[:200], Xt[-200:])), step=50)


# In[13]:

prob.argmax(axis=1)


# In[42]:

fc3ls = []
step = 50
size = 5000
sample_idx = random.sample(range(0, size), size)
# sample_idx = range(size)
for i in range(size / step):
    # model.y: yt[sample_idx[i*step:(i+1)*step]]
    fc3l = sess.run(model.logits, feed_dict={model.X: Xt[sample_idx[i*step:(i+1)*step]]})
    fc3ls.append(fc3l)
    Shared.update_screen("\r{} / {}".format((i+1)*step, size))

fc3ls = np.vstack(fc3ls)
loss, accuracy = sess.run([model.total_loss, model.accuracy], feed_dict={model.X: Xt[sample_idx], model.logits: fc3ls, model.y: yt[sample_idx]})


# In[47]:

sample_idx.index()


# In[43]:

print(loss, accuracy)


# In[11]:

val_loss, val_accuracy, summary = Shared.calculate_loss(model, sess, Xt, yt, 250)


# In[21]:

type(random.sample(range(1000), 250))


# In[22]:

type(range(250))


# In[17]:

random.sample


# In[13]:

prob


# In[184]:

yt[:250]


# In[160]:

df = pd.DataFrame(prob.argmax(axis=1), columns=['A'])


# In[161]:

df.A.value_counts()


# In[ ]:



