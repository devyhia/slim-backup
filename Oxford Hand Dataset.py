
# coding: utf-8

# In[82]:

from glob import glob


# In[2]:

get_ipython().system(u'cd /home/devyhia/hand_dataset/')


# In[4]:

get_ipython().system(u'ls /home/devyhia/hand_dataset/')


# In[5]:

get_ipython().system(u'cat /home/devyhia/hand_dataset/README.txt')


# In[7]:

PWD = "/home/devyhia/hand_dataset/"


# In[11]:

get_ipython().system(u'cat $PWD/training_dataset/README.txt')


# In[83]:

import numpy as np


# In[84]:

from scipy.io import loadmat


# In[85]:

from PIL import Image, ImageDraw


# In[158]:

import sys


# In[159]:

sys.path.append('/home/devyhia/models/slim/')


# In[161]:

import Sharedd


# In[167]:

img = Image.open('/home/devyhia/hand_dataset/training_dataset/training_data/images/Buffy_100.jpg')


# In[87]:

from pprint import pprint


# In[88]:

pprint(data['boxes'])


# In[135]:

draw = ImageDraw.Draw(img)


# In[125]:

from collections import namedtuple


# In[126]:

Point = namedtuple('Point', ['x', 'y'])
FourPointBox = namedtuple('FourPointBox', ['a', 'b', 'c', 'd'])
TwoPointBox = namedtuple('TwoPointBox', ['x1', 'y1', 'x2', 'y2'])


# In[156]:

annots = glob('/home/devyhia/hand_dataset/training_dataset/training_data/annotations/*.mat') + glob('/home/devyhia/hand_dataset/validation_dataset/validation_data/annotations/*.mat') + glob('/home/devyhia/hand_dataset/test_dataset/test_data/annotations/*.mat')


# In[168]:

img.size


# In[163]:

N = len(annots)
for i, annot_file in enumerate(annots):
    # annot_file: '/home/devyhia/hand_dataset/training_dataset/training_data/annotations/Buffy_1.mat'
    data = loadmat(annot_file)

    fourPointBoxes = []
    twoPointBoxes = []

    for box_idx in range(data['boxes'].shape[1]):
        box = data['boxes'][0,box_idx][0,0]

        box4points = FourPointBox(
            a=Point(x=box[0][0,1], y=box[0][0,0]),
            b=Point(x=box[1][0,1], y=box[1][0,0]),
            c=Point(x=box[2][0,1], y=box[2][0,0]),
            d=Point(x=box[3][0,1], y=box[3][0,0])
        )

        box2points = TwoPointBox(
            x1=min(box4points.a.x, box4points.b.x, box4points.c.x, box4points.d.x),
            y1=min(box4points.a.y, box4points.b.y, box4points.c.y, box4points.d.y),
            x2=max(box4points.a.x, box4points.b.x, box4points.c.x, box4points.d.x),
            y2=max(box4points.a.y, box4points.b.y, box4points.c.y, box4points.d.y),
        )

        fourPointBoxes.append(box4points)
        twoPointBoxes.append(box2points)

    with open(annot_file.replace('.mat', '.fourPointBoxes'), 'w') as f:
        pickle.dump(fourPointBoxes, f)
    
    with open(annot_file.replace('.mat', '.twoPointBoxes'), 'w') as f:
        pickle.dump(twoPointBoxes, f)
    
    Shared.update_screen('\r{} out of {}'.format(i+1, N))


# In[146]:

import pickle


# In[149]:

with open('test.picke', 'w') as f:
    pickle.dump(fourPointBoxes, f)


# In[151]:

with open('test.picke', 'r') as f:
    print(pickle.load(f))


# In[115]:

draw.line((box[0][0,1], box[0][0,0], box[1][0,1], box[1][0,0]), fill='red')
draw.line((box[1][0,1], box[1][0,0], box[2][0,1], box[2][0,0]), fill='red')
draw.line((box[2][0,1], box[2][0,0], box[3][0,1], box[3][0,0]), fill='red')
draw.line((box[3][0,1], box[3][0,0], box[0][0,1], box[0][0,0]), fill='red')


# In[137]:

img


# In[80]:

data['boxes'][0,0][0][0][0]


# In[56]:

draw = ImageDraw.Draw(img)
for i in range(len(data['boxes'][0])):
    box = data['boxes'][0][i][0][0]
    draw.rectangle((box[0], box[1], box[2], box[3]))


# In[51]:



