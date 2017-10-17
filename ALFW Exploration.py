
# coding: utf-8

# In[19]:

import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from pprint import pprint
import helpers
get_ipython().magic(u'matplotlib inline')


# In[78]:

get_ipython().magic(u'matplotlib inline')


# In[93]:

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


# In[94]:

con = sqlite3.connect('/home/devyhia/face_detection/aflw/data/aflw.sqlite')
con.row_factory = dict_factory


# In[95]:

cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
pprint(cur.fetchall())


# In[51]:

pprint(cur.execute('SELECT * from FacesMetaData LIMIT 2').fetchall())


# In[54]:

from glob import glob


# In[97]:

pprint(cur.execute('SELECT * from FaceImages where filepath LIKE "3%" LIMIT 5').fetchall())


# In[ ]:

Image.open('/home/devyhia/face_detection/aflw/data/flickr/3/image04573.jpg')


# In[5]:

import sys
sys.path.append('/home/deeplearners/caffe/python/')


# In[6]:

import caffe


# In[8]:

from scipy.ndimage import imread


# In[9]:

from glob import glob


# In[189]:

img0 = glob('/home/devyhia/aflw/data/flickr/3/*')[100]


# In[190]:

img0_content = imread(img0)


# In[191]:

plt.imshow(img0_content)


# In[20]:

caffe.set_device(0)
caffe.set_mode_gpu()


# In[25]:

net = caffe.Net('/home/devyhia/vgg_face_caffe/VGG_FACE_deploy.prototxt',
                '/home/devyhia/vgg_face_caffe/VGG_FACE.caffemodel', caffe.TEST)


# In[27]:

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})


# In[28]:

transformer.set_transpose('data', (2,0,1))


# In[29]:

transformer.set_channel_swap('data', (2,1,0))


# In[36]:

transformer.set_raw_scale('data', 224)


# In[37]:

im = caffe.io.load_image(img0)


# In[42]:

net.blobs['data'].data[...] = transformer.preprocess('data', im)


# In[43]:

out = net.forward()


# In[44]:

print out['prop']


# In[48]:

out['prob'].argmax()


# In[49]:

out['prob'].shape


# In[50]:

import dlib


# In[55]:

get_ipython().system(u'pip install dlib')


# In[57]:

import dlib


# In[157]:

plt.imshow(imread('/home/devyhia/aflw/data/flickr/0/image10000.jpg'))


# In[159]:

plt.imshow(imread('/home/devyhia/aflw/data/flickr/0/image10000.jpg'))


# In[96]:

import json


# In[97]:

coco_val = json.load(open('/home/devyhia/annotations/instances_val2014.json'))


# In[102]:

coco_val.keys()


# In[104]:

coco_val['annotations'][0]


# In[124]:

coco_val['images'][2]


# In[106]:

coco_val_annot = {}


# In[ ]:

N = len(coco_val['annotations'])
for i, a in enumerate(coco_val['annotations']):
    image_id = a['image_id']
    category_id = a['category_id']
    if image_id not in coco_val_annot:
        coco_val_annot[image_id] = []
    
    coco_val_annot[image_id] += [ category_id ]
    helpers.update_screen("\r{} out of {}".format(i+1, N))


# In[110]:

len(coco_val_annot.keys())


# In[116]:

coco_val_no_persons = []


# In[118]:

for image_id, categories in coco_val_annot.iteritemste():
    if min(categories) > 1:
        coco_val_no_persons += [ image_id ]
    
    helpers.update_screen("{} -- {}".format(image_id, len(coco_val_no_persons)))


# In[121]:

len(coco_val_no_persons)


# In[122]:

no_persons_images = []


# In[125]:

coco_val_no_persons[0]


# In[129]:

with open('/home/devyhia/annotations/no_persons.txt', 'w') as f:
    f.write("\n".join(map(str, coco_val_no_persons)))


# In[155]:

len(glob('/home/devyhia/coco-no-persons/*.jpg'))


# In[138]:

import pandas as pd


# In[141]:

res = pd.read_csv('/home/devyhia/annotations/no_persons.txt', header=None)


# In[152]:

res.iloc[0, 0]


# In[153]:

import urllib


# In[154]:




# In[164]:

len(glob('/home/devyhia/aflw/data/flickr/0/*.jpg'))


# In[192]:

import Image


# In[193]:

get_ipython().system(u'pip install Image')


# In[194]:

get_ipython().system(u'ls /home/devyhia/.local/lib/python2.7/site-packages')


# In[196]:

import image


# In[199]:

image.Image


# In[211]:

import tensorflow as tf


# In[212]:

tf.__version__


# ~/face_detection/coco-no-persons/neg_hard/24/

# In[213]:

res = np.fromfile('/home/devyhia/face_detection/coco-no-persons/neg_hard/24/12_0.npy')


# In[216]:

get_ipython().system(u'cd /home/devyhia/A-Convolutional-Neural-Network-Cascade-for-Face-Detection/')


# In[220]:

sys.path.append('/home/devyhia/A-Convolutional-Neural-Network-Cascade-for-Face-Detection/')


# In[224]:

import param
import os


# In[225]:

neg_file_list = [f for f in os.listdir(param.neg_dir) if f.endswith(".jpg")]


# In[276]:

neg_db_24 = np.empty((0,param.img_size_24,param.img_size_24,param.input_channel),np.float32)


# In[277]:

neg_file_list = [f for f in os.listdir(param.neg_dir + "neg_hard/24/") if f.startswith("24_") and f.endswith(".npy")]


# In[278]:

neg_arrays = [np.load(param.neg_dir + "neg_hard/24/" + db_name, mmap_mode='r') for nid,db_name in enumerate(neg_file_list)]


# In[279]:

total_size = reduce(lambda prev, curr: prev + curr.shape[0], neg_arrays, 0)


# In[280]:

total_size


# In[285]:

neg_24_array = np.memmap(param.neg_dir + "neg_hard/24/24_all.npy", dtype='float32', mode='w+', shape=(total_size,24,24,3))


# In[ ]:

cursor = 0
for i, arr in enumerate(neg_arrays):
    print("Processing array {}".format(i))
    total = arr.shape[0]
    neg_24_array[cursor:(cursor+total), :, :, :] = arr
    cursor += total


# In[289]:

neg_12_array.flush()


# In[290]:

neg_24_array.flush()


# In[292]:

np.load(param.neg_dir + "neg_hard/24/12_all.npy", mmap_mode='r')


# In[2]:

import Shared


# In[3]:

Shared.select_gpu(1)


# In[4]:

import sys
sys.path.append('/home/devyhia/A-Convolutional-Neural-Network-Cascade-for-Face-Detection/')


# In[5]:

import numpy as np
import tensorflow as tf
from PIL import Image
from compiler.ast import flatten
import os
import sys
import math

import param
import util
import model

import random

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

#12net
input_12_node = tf.placeholder("float")
target_12_node = tf.placeholder("float", [None,1])
inputs_12 = np.zeros((param.mini_batch,param.img_size_12,param.img_size_12,param.input_channel), np.float32)

net_12 = model.detect_12Net(input_12_node,target_12_node)
net_12_calib = model.calib_12Net(input_12_node,target_12_node)
restorer_12 = tf.train.Saver([v for v in tf.all_variables() if "12det_" in v.name])
restorer_12.restore(sess, param.model_dir + "12-net.ckpt")
restorer_12_calib = tf.train.Saver([v for v in tf.all_variables() if "12calib_" in v.name])
restorer_12_calib.restore(sess, param.model_dir + "12-calib-net.ckpt")

#24net
input_24_node = tf.placeholder("float", [None, param.img_size_24, param.img_size_24, param.input_channel])
from_12_node = tf.placeholder("float",[None,16])
target_24_node = tf.placeholder("float", [None,1])
inputs_24 = np.zeros((param.mini_batch,param.img_size_24,param.img_size_24,param.input_channel), np.float32)

net_24 = model.detect_24Net(input_24_node,target_24_node,from_12_node)
net_24_calib = model.calib_24Net(input_24_node,target_24_node)
restorer_24 = tf.train.Saver([v for v in tf.all_variables() if "24det_" in v.name])
restorer_24.restore(sess, param.model_dir + "24-net.ckpt")
restorer_24_calib = tf.train.Saver([v for v in tf.all_variables() if "24calib_" in v.name])
restorer_24_calib.restore(sess, param.model_dir + "24-calib-net.ckpt")

#48net
input_48_node = tf.placeholder("float", [None, param.img_size_48, param.img_size_48, param.input_channel])
from_24_node = tf.placeholder("float",[None,128+16])
target_48_node = tf.placeholder("float", [None,1])
inputs_48 = np.zeros((param.mini_batch,param.img_size_48,param.img_size_48,param.input_channel), np.float32)

net_48 = model.detect_48Net(input_48_node,target_48_node,from_24_node)
net_48_calib = model.calib_48Net(input_48_node,target_48_node)
restorer_48 = tf.train.Saver([v for v in tf.all_variables() if "48det_" in v.name])
restorer_48.restore(sess, param.model_dir + "48-net.ckpt")
restorer_48_calib = tf.train.Saver([v for v in tf.all_variables() if "48calib_" in v.name])
restorer_48_calib.restore(sess, param.model_dir + "48-calib-net.ckpt")


# In[6]:

import matplotlib.patches as patches


# In[7]:

neg_file_list = [f for f in os.listdir(param.pos_dir+'/0/') if f.endswith(".jpg")]


# In[50]:

neg_file_list = neg_file_list[:1000]


# In[137]:

get_ipython().system(u'wget http://s.afl.com.au/staticfile/AFL%20Tenant/Media/Images/396463-tlsnewslandscape.jpg')


# In[8]:

img_name = neg_file_list[1200]
img = Image.open(param.pos_dir + '/0/' + img_name)


# In[138]:

img = Image.open('396463-tlsnewslandscape.jpg')


# In[139]:

#check if gray
if len(np.shape(img)) != param.input_channel:
    img = np.asarray(img)
    img = np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1))
    img = np.concatenate((img,img,img),axis=2)
    img = Image.fromarray(img)


# In[ ]:

#12-net
#xmin, ymin, xmax, ymax, score, cropped_img, scale
result_box = util.sliding_window(img, param.thr_12, net_12, input_12_node)

#12-calib
result_db_tmp = np.zeros((len(result_box),param.img_size_12,param.img_size_12,param.input_channel),np.float32)
for id_,box in enumerate(result_box):
    result_db_tmp[id_,:] = util.img2array(box[5],param.img_size_12)

calib_result = net_12_calib.prediction.eval(feed_dict={input_12_node: result_db_tmp})
result_box = util.calib_box(result_box,calib_result,img)

#NMS for each scale
scale_cur = 0
scale_box = []
suppressed = []
for id_,box in enumerate(result_box):
    if box[6] == scale_cur:
        scale_box.append(box)
    if box[6] != scale_cur or id_ == len(result_box)-1:
        suppressed += util.NMS(scale_box)
        scale_cur += 1
        scale_box = [box]

result_box = suppressed
result_box = [f for f in result_box if f[4] > 0.5]
suppressed = []          


# In[ ]:

len(result_box)


# In[12]:

#24-net
result_db_12 = np.zeros((len(result_box),param.img_size_12,param.img_size_12,param.input_channel),np.float32)
result_db_24 = np.zeros((len(result_box),param.img_size_24,param.img_size_24,param.input_channel),np.float32)
for bid,box in enumerate(result_box):
    resized_img_12 = util.img2array(box[5],param.img_size_12)
    resized_img_24 = util.img2array(box[5],param.img_size_24)

    result_db_12[bid,:] = resized_img_12
    result_db_24[bid,:] = resized_img_24

from_12 = net_12.from_12.eval(feed_dict={input_12_node: result_db_12})
result = net_24.prediction.eval(feed_dict={input_24_node: result_db_24, from_12_node: from_12})
result_id = np.where(result > param.thr_24)[0]
result_box = [result_box[i] for i in result_id]

#24-calib
result_db_tmp = np.zeros((len(result_box),param.img_size_24,param.img_size_24,param.input_channel),np.float32)
for id_,box in enumerate(result_box):
    result_db_tmp[id_,:] = util.img2array(box[5],param.img_size_24)

calib_result = net_24_calib.prediction.eval(feed_dict={input_24_node: result_db_tmp})
result_box = util.calib_box(result_box,calib_result,img)

#NMS for each scale
scale_cur = 0
scale_box = []
suppressed = []
for id_,box in enumerate(result_box):
    if box[6] == scale_cur:
        scale_box.append(box)
    if box[6] != scale_cur or id_ == len(result_box)-1:
        suppressed += util.NMS(scale_box)
        scale_cur += 1
        scale_box = [box]

result_box = suppressed
result_box = [f for f in result_box if f[4] > 0.5]
suppressed = []


# In[ ]:

len(result_box)


# In[394]:

#48-net
result_db_12 = np.zeros((len(result_box),param.img_size_12,param.img_size_12,param.input_channel),np.float32)
result_db_24 = np.zeros((len(result_box),param.img_size_24,param.img_size_24,param.input_channel),np.float32)
result_db_48 = np.zeros((len(result_box),param.img_size_48,param.img_size_48,param.input_channel),np.float32)

for bid,box in enumerate(result_box):
    resized_img_12 = util.img2array(box[5],param.img_size_12)
    resized_img_24 = util.img2array(box[5],param.img_size_24)
    resized_img_48 = util.img2array(box[5],param.img_size_48)

    result_db_12[bid,:] = resized_img_12
    result_db_24[bid,:] = resized_img_24
    result_db_48[bid,:] = resized_img_48

from_12 = net_12.from_12.eval(feed_dict={input_12_node: result_db_12})
from_24 = net_24.from_24.eval(feed_dict={input_24_node: result_db_24, from_12_node:from_12})

result = net_48.prediction.eval(feed_dict={input_48_node: result_db_48, from_24_node: from_24})
result_id = np.where(result > param.thr_48)[0]
result_box = [result_box[i] for i in result_id]


#global NMS
result_box = util.NMS(result_box)

#48-calib
result_db_tmp = np.zeros((len(result_box),param.img_size_48,param.img_size_48,param.input_channel),np.float32)
for id_,box in enumerate(result_box):
    result_db_tmp[id_,:] = util.img2array(box[5],param.img_size_48)

calib_result = net_48_calib.prediction.eval(feed_dict={input_48_node: result_db_tmp})
result_box = util.calib_box(result_box,calib_result,img)


# In[ ]:

len(result_box)


# In[134]:

img_best = reduce(lambda prev, curr: curr if curr[4] > prev[4] else prev, neg_box[1:], neg_box[0])


# In[400]:

img_best = [ i for i in result_box if i[4] > 0.9999 ]


# In[401]:

len(img_best)


# In[197]:

plt.imshow(img)


# In[ ]:

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, aspect='equal')
ax2.imshow(img)
for i in range(len(img_best)):
    ax2.add_patch(
        patches.Rectangle(
            (img_best[i][0], img_best[i][1]),
            img_best[i][2] - img_best[i][0],
            img_best[i][3] - img_best[i][1],
            fill=False      # remove background
        )
    )


# In[19]:

neg_db_tmp = np.zeros((len(neg_box),param.img_size_12,param.img_size_12,param.input_channel),np.float32)
for id_,box in enumerate(neg_box):
    neg_db_tmp[id_,:] = util.img2array(box[5],param.img_size_12)


# In[20]:

calib_result = net_12_calib.prediction.eval(feed_dict={input_12_node: neg_db_tmp})


# In[21]:

neg_box = util.calib_box(neg_box,calib_result,img)


# In[22]:

len(neg_box)


# In[23]:

scale_cur = 0
scale_box = []
suppressed = []
for id_,box in enumerate(neg_box):
    if box[6] == scale_cur:
        scale_box.append(box)
    if box[6] != scale_cur or id_ == len(neg_box)-1:
        suppressed += util.NMS(scale_box)
        scale_cur += 1
        scale_box = [box]

neg_box = suppressed


# In[25]:

len(neg_box)


# In[29]:

import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from pprint import pprint
import helpers
get_ipython().magic(u'matplotlib inline')


# In[40]:

plt.imshow(neg_box[5][5])


# In[177]:

sess.close()
tf.reset_default_graph()


# In[254]:

res = np.load(param.neg_dir + "neg_hard/48/48_0.npy", mmap_mode='r')


# In[257]:

plt.imshow(res[0])


# In[265]:

res48 = [ np.load(param.neg_dir + "neg_hard/48/48_{}.npy".format(i), mmap_mode='r') for i in range(6) ]


# In[266]:

res24 = [ np.load(param.neg_dir + "neg_hard/48/24_{}.npy".format(i), mmap_mode='r') for i in range(6) ]


# In[267]:

res12 = [ np.load(param.neg_dir + "neg_hard/48/12_{}.npy".format(i), mmap_mode='r') for i in range(6) ]


# In[268]:

reduce(lambda prev, curr: prev + curr.shape[0], res48, 0)


# In[269]:

reduce(lambda prev, curr: prev + curr.shape[0], res24, 0)


# In[270]:

reduce(lambda prev, curr: prev + curr.shape[0], res24, 0)


# In[271]:

import data


# In[273]:

pos_db_12, pos_db_24, pos_db_48 = data.load_db_detect_train_from_cache(48)


# In[278]:

pos_db_48.shape


# In[286]:

plt.imshow(pos_db_48[8])


# In[293]:

neg_db_12 = np.empty((0,param.img_size_12,param.img_size_12,param.input_channel),np.float32)


# In[294]:

neg_db_24 = np.empty((0,param.img_size_24,param.img_size_24,param.input_channel),np.float32)


# In[302]:

[f for f in sorted(os.listdir(param.neg_dir + "neg_hard/48/")) if f.startswith("48_") and f.endswith(".npy")][:3]


# In[307]:

[f for f in os.listdir(param.neg_dir + "neg_hard/24/") if f.startswith("24_") and f.endswith(".npy")]


# In[384]:

sess.close()
tf.reset_default_graph()


# In[83]:

parsed_line = line.split(',')

filename = parsed_line[0][3:-1]
xmin = int(parsed_line[1])
ymin = int(parsed_line[2])
xmax = xmin + int(parsed_line[3])
ymax = ymin + int(parsed_line[4][:-2])


# In[ ]:



