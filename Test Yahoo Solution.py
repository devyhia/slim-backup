
# coding: utf-8

# In[1]:

from glob import glob


# In[2]:

import json, pickle


# In[9]:

with open('/home/devyhia/FaceDetection_CNN/result.pickle') as f:
    result = pickle.load(f)


# In[10]:

len(result.keys())


# In[14]:

result[result.keys()[0]]


# In[50]:

len(glob('/home/devyhia/detectiondata/train/pos/*_4.png'))


# In[57]:

len(glob('/home/devyhia/detectiondata/train/pos/*'))


# In[53]:

len(glob('/home/devyhia/detectiondata/test/pos/*'))


# In[ ]:

with open('/home/devyhia/detectiondata/train/posGt/1L_0012866_Q_6_2_5.txt', 'r') as f:
    lbls = f.readlines()


# In[67]:

import sys
sys.path.append('/home/deeplearners/caffe/python/')
sys.path.append('/home/devyhia/FaceDetection_CNN/')


# In[66]:

import caffe


# In[96]:

net_full_conv = caffe.Net('/home/devyhia/FaceDetection_CNN/face_full_conv.prototxt',
                              '/home/devyhia/FaceDetection_CNN/face_full_conv.caffemodel',
                              caffe.TEST)


# In[101]:

net_full_conv.blobs['data'].reshape(1,3,857,1523)


# In[ ]:



