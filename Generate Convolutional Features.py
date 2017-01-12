
# coding: utf-8

# In[ ]:

import argparse

parser = argparse.ArgumentParser(description='RNN-CNN Network.')
parser.add_argument('--models', nargs='+', default=2, help='List of models')
parser.add_argument('--gpu', default=3, help='GPU to use for train')
parser.add_argument('--data', default="X", help='Data to be transformed')
args, unknown_args = parser.parse_known_args()


# In[8]:

import pickle


# In[20]:

import os

os.environ["gpu"] = str(args.gpu)

from InceptionV4 import *

vgg = InceptionV4("test")

config = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 1})

sess = tf.Session(config=config)
saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())


# In[33]:

tinyImageNetDir = "/home/devyhia/vgg"
Xt, yt = np.load("{}/{}.npy".format(tinyImageNetDir, args.data)), np.load("{}/{}.npy".format(tinyImageNetDir, "y" if args.data == "X" else "yt"))


# In[3]:

# Xt = vgg.resize_images(Xt)


# In[11]:

def update_screen(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


# In[12]:

# models = ["model{}".format(i) for i in range(1,5)]


# In[ ]:

sample_idx = range(0, Xt.shape[0])

for model in args.models:
    print("Loading model ({}) ...".format(model))
    saver.restore(sess, "{}.tfmodel".format(model))
    fc3ls, preds = [], []
    size = Xt.shape[0]
    step = 50
    for i in range(size / step):
        [fc3l, pred] = sess.run([vgg.end_points["PreLogitsFlatten"], vgg.end_points["Predictions"]], feed_dict={vgg.X: vgg.resize_images(Xt[sample_idx[i*step:(i+1)*step]]) })
        update_screen("\r{} out of {}".format((i+1)*step, size))
        fc3ls.append(fc3l)
        preds.append(pred)
    
    update_screen("\n")
    fc3ls = np.vstack(fc3ls)
    preds = np.vstack(preds)
    np.save("features/{}.PreLogitsFlatten.{}".format(model, args.data), fc3ls)
    np.save("features/{}.Predictions.{}".format(model, args.data), preds)


# In[ ]:



