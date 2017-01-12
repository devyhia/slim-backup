
# coding: utf-8

# In[2]:

import tensorflow as tf, os, numpy as np, cv2
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)


# In[32]:

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])


# In[33]:

summaries = []
A = tf.placeholder(tf.float32, shape=[None])
B = tf.placeholder(tf.float32, shape=[None])
sel = tf.equal(A, B)
cond = tf.where(sel)
images = tf.reshape(tf.gather(X, cond), [-1, 224, 224, 3])
# summaries += [ tf.image_summary('Image', images) ]


# In[34]:

sess = tf.Session()

tf_logs = tf.train.SummaryWriter("logs/{}".format("Test"), sess.graph, max_queue=0)
# tf_summary = tf.merge_summary(summaries)


# In[19]:

TRAIN_DIR = '/home/devyhia/cats.vs.dogs/train/'
TEST_DIR = '/home/devyhia/cats.vs.dogs/test/'

ROWS = 224
COLS = 224
CHANNELS = 3
SLICE = 10000

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)][:10]

#     Bagging Code
# test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
#     bag = int(args.bag)
#     bag_start = bag * 3125
#     bag_end = (bag+1) * 3125

#     # slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
#     train_images = train_dogs[:bag_start] + train_dogs[bag_end:] + train_cats[:bag_start] + train_cats[bag_end:]
#     valid_images = train_dogs[bag_start:bag_end] + train_cats[bag_start:bag_end]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        data[i] = read_image(image_file).astype(np.float32)

    return data

def get_label(path):
    return 1 if re.search("(cat|dog)\.(\d+)\.", path).group(1) == 'cat' else 0

data = prep_data(train_images)


# In[35]:

_sel, _cond, _images = sess.run([sel, cond, images], feed_dict={ A: np.array([1,1,1,0,1]), B: np.array([0,1,1,1,1]), X: data[:5] })
# tf_logs.add_summary(s, 6)


# In[ ]:



