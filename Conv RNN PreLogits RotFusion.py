
# coding: utf-8

# In[2]:

# from sklearn.decomposition import PCA
from matplotlib.mlab import PCA


# In[3]:

import argparse

parser = argparse.ArgumentParser(description='RNN-CNN Network.')
parser.add_argument('--depth', default=4, help='Depth of the RNN network')
parser.add_argument('--hidden', default=128, help='Hidden units of the RNN network')
parser.add_argument('--gpu', default=3, help='GPU to use for train')
parser.add_argument('--rot', default=0, help='RNN Rotation')
parser.add_argument('--name', default="rnn_augmented_with_4_rotations", help='Name of the RNN model to use for train')
parser.add_argument('--predict', default="no", help='RNN Rotation')
args, unknown_args = parser.parse_known_args()

import logging
logging.basicConfig(filename="logs/{}.log".format(args.name), format='%(message)s', level=logging.DEBUG)
logging.info("Hello from logging ...")


# In[4]:

import os, random, sys
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# In[5]:

# VGG16 Features
# tinyImageNetDir = "/home/devyhia/vgg"
# X, Y = np.load("{}/features/vgg16_12_Adagrad.fc2.X.npy".format(tinyImageNetDir)), np.load("{}/y.npy".format(tinyImageNetDir))
# Xt, Yt = np.load("{}/features/vgg16_12_Adagrad.fc2.Xt.npy".format(tinyImageNetDir)), np.load("{}/yt.npy".format(tinyImageNetDir))


# In[5]:

# Inception V4 Features
tinyImageNetDir = "/home/devyhia/vgg"
X, Y = np.load("features/model2.PreLogitsFlatten.X.npy".format(tinyImageNetDir)), np.load("{}/y.npy".format(tinyImageNetDir))
Xt, Yt = np.load("features/model2.PreLogitsFlatten.Xt.npy".format(tinyImageNetDir)), np.load("{}/yt.npy".format(tinyImageNetDir))


# In[6]:

# Tiny Images Raw Data
tinyImageNetDir = "/home/devyhia/vgg"
rawX = np.load("{}/X.npy".format(tinyImageNetDir))
rawXt = np.load("{}/Xt.npy".format(tinyImageNetDir))


# In[7]:

# X = np.array([np.hstack([prelogX[0].reshape(24, 64), rawX[0].reshape(24, 512)]) for i in range(prelogX.shape[0])])
# Xt = np.array([np.hstack([prelogXt[0].reshape(24, 64), rawXt[0].reshape(24, 512)]) for i in range(prelogXt.shape[0])])


# In[8]:

# Reverse Sequence
# reverse_idx = list(reversed(range(X.shape[1])))
# X = X[:, reverse_idx]
# Xt = Xt[:, reverse_idx]


# In[58]:

tf.reset_default_graph()

# Parameters
learning_rate = 0.001
batch_size = 50
display_step = 25
epochs = 10
depth = 4

# Network Parameters
n_input = 64 # MNIST data input (img shape: 28*28)
n_steps = 64 # timesteps
n_hidden = 128
n_classes = 100 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float32", [None, X.shape[1]])
y = tf.placeholder("float32", [None, n_classes])

# Images to be passed
x_raw = tf.placeholder("float32", [None, 64, 64, 3])
x_raw_gray_0 = tf.image.rgb_to_grayscale(x_raw)
x_raw_gray_90 = tf.map_fn(lambda _img: tf.image.rot90(_img, 1), x_raw_gray_0)
x_raw_gray_180 = tf.map_fn(lambda _img: tf.image.rot90(_img, 2), x_raw_gray_0)
x_raw_gray_270 = tf.map_fn(lambda _img: tf.image.rot90(_img, 3), x_raw_gray_0)

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases, scope="RNN"):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    x = tf.reshape(x, (-1, n_steps, n_input))
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    #     , forget_bias=1.0
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    multi_cells = rnn_cell.MultiRNNCell([lstm_cell] * depth, state_is_tuple=True)

    # Get lstm cell output
    outputs, states = rnn.rnn(multi_cells, x, dtype=tf.float32, scope=scope)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


if args.rot == 0:
    pred_rnn_x_raw = RNN(x_raw_gray_0, weights, biases, scope="RNN_0")
elif args.rot == 1:
    pred_rnn_x_raw = RNN(x_raw_gray_90, weights, biases, scope="RNN_90")
elif args.rot == 2:
    pred_rnn_x_raw = RNN(x_raw_gray_180, weights, biases, scope="RNN_180")
elif args.rot == 3:
    pred_rnn_x_raw = RNN(x_raw_gray_270, weights, biases, scope="RNN_270")
else:
    pred_rnn_x_raw = RNN(x_raw_gray_0, weights, biases, scope="RNN_0")
    
pred_fc = slim.fully_connected(x, n_classes, activation_fn=None)

pred = pred_fc + pred_rnn_x_raw # + pred_rnn_x_raw_90 # + pred_rnn_x_raw_180 + pred_rnn_x_raw_270
prob = tf.nn.softmax(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
top5_correct_pred = tf.nn.in_top_k(prob, tf.argmax(y,1), 5)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
top5accuracy = tf.reduce_mean(tf.cast(top5_correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# In[6]:

def __iterate_minibatches(_X, _rawX, _y, size):
    if _X.shape[0] % size > 0:
        raise "The minibatch size should be a divisor of the batch size."

    idx = np.arange(_X.shape[0]).astype(np.int32)
    np.random.shuffle(idx) # in-place shuffling
    for i in range(_X.shape[0] / size):
        # To randomize the minibatches every time
        _idx = idx[i*size:(i+1)*size]
        yield _X[_idx], _rawX[_idx], _y[_idx]


# In[7]:

def update_screen(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


# In[8]:

def predict_proba(sess, Xt, rawXt, size=1000, step=10, randomize=True):
    preds, probs = [], []
    idx = range(0, Xt.shape[0])
    sample_idx = random.sample(idx, size) if randomize else idx
    for i in range(size / step):
        itr = sample_idx[i*step:(i+1)*step]
        _pred, _prob = sess.run([pred, prob], feed_dict={x: Xt[itr], x_raw: rawXt[itr]})
        preds.append(_pred)
        probs.append(_prob)
#         update_screen("\r{} of {}".format(i, size / step))
    
#     update_screen("\n")
    preds = np.vstack(preds)
    probs = np.vstack(probs)
    
    return preds, probs, sample_idx


# In[9]:

def calculate_loss(sess, Xt, rawXt, yt, size=1000, step=10):
    preds, probs, sample_idx = predict_proba(sess, Xt, rawXt, size=size, step=step)

    loss, acc, top5acc = sess.run([cost, accuracy, top5accuracy], feed_dict={pred: preds, y: yt[sample_idx]})

    return loss, acc, top5acc


# In[10]:

def train():
    # Launch the graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(init)
        prev_acc = 0.0
        for ep in range(epochs):
            logging.info("==== EPOCH {} ====".format(ep))
            step = 1
            for _X, _rawX, _Y in __iterate_minibatches(X, rawX, Y, batch_size):
                sess.run(optimizer, feed_dict={x: _X, x_raw: _rawX, y: _Y})
                if step % display_step == 0:
                    loss, acc, top5acc = calculate_loss(sess, Xt, rawXt, Yt)
                    logging.info("Iter " + str(step) + ", Loss= " +                           "{:.4f}".format(loss) + ", Acc= " +                           "{:.4f}".format(acc) + ", Top-5 Acc= " +                           "{:.4f}".format(top5acc))
                step += 1

            loss, acc, top5acc = calculate_loss(sess, Xt, rawXt, Yt, size=Xt.shape[0])
            logging.info("====================================")
            logging.info("Epoch {}: Loss={} Acc={} Top-5 Acc={}".format(ep, loss, acc, top5acc))
            logging.info("====================================")
            if acc > prev_acc:
                prev_acc = acc
                saver.save(sess, "conv_rnn_prelogits/{}.tfmodel".format(args.name))
                logging.info("++++ Saved BEST ACC") 


# In[11]:

def gen_predictions():
    # Launch the graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(init)
        saver.restore(sess, "conv_rnn_prelogits/{}.tfmodel".format(args.name))
        res = predict_proba(sess, Xt, rawXt, size=Xt.shape[0], randomize=False)
        np.save("probs/{}.probs.npy".format(args.name), res[1])


# In[63]:

if args.predict == "no":
    train()
else:
    print("Predicting ...")
    gen_predictions()

