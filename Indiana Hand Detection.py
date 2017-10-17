
# coding: utf-8

# In[1]:

import sys
sys.path.append('/home/devyhia/LendingAHand/')
sys.path.append('/home/deeplearners/caffe/python/')

import numpy as np
#import Image
import sys
import os
import PIL
import operator
from  math import pow
from PIL import Image, ImageDraw, ImageFont
import caffe

caffe.set_device(0)


# In[20]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[3]:

sys.path.append('/home/devyhia/FaceDetection_CNN/')
from nms import nms


# In[4]:

from scipy.ndimage import imread


# In[5]:

caffe_root = '/home/deeplearners/caffe/'


# In[6]:

net_full_conv = caffe.Net(
    '/home/devyhia/LendingAHand/hand_classifier_conv.prototxt',
    '/home/devyhia/LendingAHand/hand_classifier_conv.caffemodel',
    caffe.TEST
)


# In[54]:

net_full = caffe.Net(
    '/home/devyhia/LendingAHand/hand_classifier.prototxt',
    '/home/devyhia/LendingAHand/hand_classifier.caffemodel',
    caffe.TEST
)


# In[13]:

def generateBoundingBox(featureMap, scale):
    boundingBox = []
    stride = 32
    cellSize = 227
    #227 x 227 cell, stride=32
    for (x,y), prob in np.ndenumerate(featureMap):
        if(prob >= 0.1):
            boundingBox.append([float(stride * y)/ scale, float(x * stride)/scale, float(stride * y + cellSize - 1)/scale, float(stride * x + cellSize - 1)/scale, prob])
    #sort by prob, from max to min.
    #boxes = np.array(boundingBox)
    return boundingBox


# In[55]:

net_full_conv.blobs


# In[86]:

get_ipython().system(u'wget http://a.scpr.org/i/0fb039f9119d1310ee112897996da0a9/133472-full.jpg')


# In[ ]:

glob


# In[47]:

Image.open('/home/devyhia/hand_dataset_indiana/train/pos/5L_0038145_I_3_0_3.png').size


# In[84]:

get_ipython().system(u'ls *.jpg')


# In[9]:

img_url = '/home/devyhia/distracted.driver/Talk Left/600.original.jpg'


# In[10]:

img = Image.open(img_url)


# In[58]:

img_arr = imread(img_url)


# In[11]:

scales = [1, 0.793700526, 0.6299605249726766, 0.5000000000300495, 0.3968502630238503, 0.31498026250526834, 0.2500000000300495]


# In[55]:

total_boxes = []
for scale in scales[1:2]:
    print("Image: {} - Scale: {}".format(0, scale))
    
    scale_img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
    scale_img.save("tmp.jpg")
    
    net_full_conv.blobs['data'].reshape(1,3,scale_img.size[1], scale_img.size[0])
    
    im = caffe.io.load_image("tmp.jpg")
    transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    
    out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    
    boxes = generateBoundingBox(out['prob'][0,1], scale)
    
    if(boxes):
        total_boxes.extend(boxes)


# In[56]:

len(total_boxes)


# In[57]:

print('NMS - MAX')
#nms
boxes_nms = np.array(total_boxes)
# true_boxes1 = boxes_nms
# true_boxes1 = nms_max(boxes_nms, overlapThresh=0.3)
# Shared.update_screen('NMS - AVG - Boxes: {}\n'.format(true_boxes1.shape[0]))
# true_boxes = nms_average(np.array(true_boxes1), overlapThresh=0.07)
# true_boxes = non_max_suppression_fast(true_boxes1, overlapThresh=0.07)
# Shared.update_screen('{}\n'.format(true_boxes1.shape))
# np.sort(true_boxes1.reshape(-1, 1,1,1,1,1), axis=4)[::-1][:1].reshape(-1,5)
true_boxes = nms(total_boxes, 0.3)


# In[58]:

len(true_boxes)


# In[161]:

out['prob'].shape


# In[120]:

true_boxes


# In[353]:

sorted([b for b in true_boxes if b[0] >= 0.5 * 1920], reverse=True, key=lambda x: x[4])


# In[84]:

img.size


# In[61]:

img = Image.open(img_url)
draw = ImageDraw.Draw(img)
# sorted(true_boxes, reverse=True, key=lambda x: x[4])[:2]
# sorted([b for b in true_boxes if b[2] < 1920 and b[3] < 1080], reverse=True, key=lambda x: x[4])[:2]
for box in sorted([b for b in true_boxes if b[0] >= 0.5 * 1920], reverse=True, key=lambda x: x[4])[:2]:
    draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255,0,0) )
    font_path=os.environ.get("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    ttFont = ImageFont.truetype(font_path, 20)
    draw.text((box[0], box[1]), "{0:.2f}".format(box[4]), font=ttFont, fill='red')


# In[62]:

img.resize((int(img.size[0]), int(img.size[1])))


# In[324]:

def get_box_probs(boxes):
    for i, box in enumerate(boxes):
        print("{} out of {}".format(i+1, len(boxes)))
        img.crop(map(int, box[:4])).resize((227, 227)).save('tmp.jpg')
    
        transformer = caffe.io.Transformer({'data': net_full.blobs['data'].data.shape})
        transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)

        im = caffe.io.load_image("tmp.jpg")
        out = net_full.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
        box += (out['prob'][0][1],) # probability it is a hand


# In[368]:

get_box_probs(true_boxes)


# In[292]:

true_boxes


# In[370]:

img = Image.open(img_url)
draw = ImageDraw.Draw(img)
# sorted(true_boxes, reverse=True, key=lambda x: x[4])[:2]
# sorted([b for b in true_boxes if b[2] < 1920 and b[3] < 1080], reverse=True, key=lambda x: x[4])[:2]
for box in sorted([b for b in true_boxes if b[0] >= 0.25 * 1920], reverse=True, key=lambda x: x[5]):
    draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255,0,0) )
    font_path=os.environ.get("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    ttFont = ImageFont.truetype(font_path, 20)
    draw.text((box[0], box[1]), "{0:.2f}".format(box[5]), font=ttFont, fill='red')


# In[371]:

img


# In[261]:

boxes_nms = np.array(boxes)


# In[262]:

true_boxes = nms(boxes, 0.1)


# In[263]:

len(true_boxes)


# In[265]:

draw = ImageDraw.Draw(img)
for box in true_boxes:
    draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255,0,0) )
    font_path=os.environ.get("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    ttFont = ImageFont.truetype(font_path, 20)
    draw.text((box[0], box[1]), "{0:.2f}".format(box[4]), font=ttFont)


# In[266]:

img.resize((int(img.size[0]/2), int(img.size[1]/2)))


# In[313]:

img.crop((1250, 230, 1450, 400)).resize((227,227)).save('tmp.jpg')


# In[241]:

img.crop(np.array(true_boxes[1][:4], dtype=np.int)).save('tmp.jpg')


# In[314]:

im = caffe.io.load_image("tmp.jpg")
transformer = caffe.io.Transformer({'data': net_full.blobs['data'].data.shape})
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)


# In[315]:

out = net_full.forward_all(data=np.asarray([transformer.preprocess('data', im)]))


# In[316]:

out


# In[183]:

img.resize((int(img.size[0]/2), int(img.size[1]/2)))


# In[145]:

img.crop(box=(1000, 375, 1300,575)).resize((227, 227)).save('tmp.jpg')


# In[151]:

def convert_full_conv():
    # Load the original network and extract the fully connected layers' parameters.
    net = caffe.Net('/home/devyhia/LendingAHand/hand_classifier.prototxt',
                    '/home/devyhia/LendingAHand/hand_classifier.caffemodel',
                    caffe.TEST)
    params = ['fc6', 'fc7', 'fc8_flickr']
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net('/home/devyhia/LendingAHand/hand_classifier_conv.prototxt',
                              '/home/devyhia/LendingAHand/hand_classifier.caffemodel',
                              caffe.TEST)
    params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]
    net_full_conv.save('/home/devyhia/LendingAHand/hand_classifier_conv.caffemodel')


# In[152]:

convert_full_conv()


# In[318]:

get_ipython().system(u'pip install selectivesearch --user')


# In[13]:

import selectivesearch


# In[14]:

img_lbl, regions = selectivesearch.selective_search(img_arr, scale=0.25, sigma=0.9, min_size=10)


# In[25]:

regions_dict = {}
for r in regions:
    regions_dict[r['rect']] = r


# In[28]:

regions = []
for k in regions_dict:
    regions.append(regions_dict[k])


# In[21]:

draw = ImageDraw.Draw(img)
for r in sorted([ r for r in regions if r['size'] < 0.5*1920*1080], reverse=True, key=lambda r: r['prob'])[:10]:
    box = r['rect']
    prob = r['prob']
    draw.rectangle(box, outline=(255,0,0))
    font_path=os.environ.get("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    ttFont = ImageFont.truetype(font_path, 20)
    draw.text((box[0], box[1]), "{0:.2f}".format(prob), font=ttFont)


# In[40]:

sorted(regions, reverse=True, key=lambda r: r['prob'])[:5]


# In[22]:

img.resize((int(img.size[0]/2), int(img.size[1]/2)))


# In[15]:




# In[ ]:



