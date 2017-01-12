
# coding: utf-8

# In[1]:

import os

os.environ["gpu"] = "1"

from Inception_V4 import *

vgg = InceptionV4("test")

config = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 1})

sess = tf.Session(config=config)
saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())


# In[3]:

tinyImageNetDir = "/home/devyhia/vgg"
Xt, yt = np.load("{}/X.npy".format(tinyImageNetDir)), np.load("{}/y.npy".format(tinyImageNetDir))


# In[ ]:

# Xt = vgg.resize_images(Xt)


# In[4]:

def update_screen(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


# In[5]:

models = ["model{}".format(i) for i in range(1,5)]


# In[6]:

sample_idx = range(0, Xt.shape[0])

for model in models:
    print("Loading model ({}) ...".format(model))
    saver.restore(sess, "{}.tfmodel".format(model))
    fc3ls = None
    size = Xt.shape[0]
    step = 10
    for i in range(size / step):
        [fc3l] = sess.run([vgg.end_points["Mixed_6h"]], feed_dict={vgg.X: vgg.resize_images(Xt[sample_idx[i*step:(i+1)*step]]) })
        update_screen("\r{} out of {}".format((i+1)*step, size))
        if i == 0:
            fc3ls = fc3l
        else:
            fc3ls = np.vstack((fc3ls, fc3l))
    update_screen("\n")

    np.save("features/{}.Mixed_6h.X".format(model), fc3ls)


# In[ ]:



