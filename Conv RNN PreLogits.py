
# coding: utf-8

# In[1]:

# from sklearn.decomposition import PCA
from matplotlib.mlab import PCA


# In[2]:

import argparse

parser = argparse.ArgumentParser(description='RNN-CNN Network.')
parser.add_argument('--depth', default=1, help='Depth of the RNN network')
parser.add_argument('--hidden', default=128, help='Hidden units of the RNN network')
parser.add_argument('--gpu', default=2, help='GPU to use for train')
parser.add_argument('--name', default="cnn_rnn_softmax", help='Name of the RNN model to use for train')
args, unknown_args = parser.parse_known_args()


# In[3]:

import os, random, sys
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# In[8]:

# VGG16 Features
tinyImageNetDir = "/home/devyhia/vgg"
X, Y = np.load("{}/features/vgg16_12_Adagrad.fc2.X.npy".format(tinyImageNetDir)), np.load("{}/y.npy".format(tinyImageNetDir))
Xt, Yt = np.load("{}/features/vgg16_12_Adagrad.fc2.Xt.npy".format(tinyImageNetDir)), np.load("{}/yt.npy".format(tinyImageNetDir))


# In[97]:

# Inception V4 Features
tinyImageNetDir = "/home/devyhia/vgg"
X, Y = np.load("features/model2.PreLogitsFlatten.X.npy".format(tinyImageNetDir)), np.load("{}/y.npy".format(tinyImageNetDir))
Xt, Yt = np.load("features/model2.PreLogitsFlatten.Xt.npy".format(tinyImageNetDir)), np.load("{}/yt.npy".format(tinyImageNetDir))


# In[5]:

# Tiny Images Raw Data
tinyImageNetDir = "/home/devyhia/vgg"
rawX = np.load("{}/X.npy".format(tinyImageNetDir))
rawXt = np.load("{}/Xt.npy".format(tinyImageNetDir))


# In[5]:

# X = np.array([np.hstack([prelogX[0].reshape(24, 64), rawX[0].reshape(24, 512)]) for i in range(prelogX.shape[0])])
# Xt = np.array([np.hstack([prelogXt[0].reshape(24, 64), rawXt[0].reshape(24, 512)]) for i in range(prelogXt.shape[0])])


# In[217]:

# Reverse Sequence
# reverse_idx = list(reversed(range(X.shape[1])))
# X = X[:, reverse_idx]
# Xt = Xt[:, reverse_idx]


# In[9]:

tf.reset_default_graph()

# Parameters
learning_rate = 0.001
batch_size = 50
display_step = 25
epochs = 100
depth = int(args.depth)

# Network Parameters
n_input = 128 # MNIST data input (img shape: 28*28)
n_steps = 12 # timesteps
n_hidden = int(args.hidden) # hidden layer num of features
n_classes = 100 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float32", [None, n_steps, n_input])
y = tf.placeholder("float32", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[10]:

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

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
    outputs, states = rnn.rnn(multi_cells, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
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


# In[11]:

def __iterate_minibatches(_X,_y, size):
    if _X.shape[0] % size > 0:
        raise "The minibatch size should be a divisor of the batch size."

    idx = np.arange(_X.shape[0]).astype(np.int32)
    np.random.shuffle(idx) # in-place shuffling
    for i in range(_X.shape[0] / size):
        # To randomize the minibatches every time
        _idx = idx[i*size:(i+1)*size]
        _X_small = _X[_idx]
        _y_small = _y[_idx]
        yield _X_small, _y_small


# In[12]:

def update_screen(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


# In[13]:

def predict_proba(sess, Xt, size=1000, step=10, randomize=True):
    preds, probs = [], []
    idx = range(0, Xt.shape[0])
    sample_idx = random.sample(idx, size) if randomize else idx
    for i in range(size / step):
        _pred, _prob = sess.run([pred, prob], feed_dict={x: Xt[sample_idx[i*step:(i+1)*step]]})
        preds.append(_pred)
        probs.append(_prob)
#         update_screen("\r{} of {}".format(i, size / step))
    
#     update_screen("\n")
    preds = np.vstack(preds)
    probs = np.vstack(probs)
    
    return preds, probs, sample_idx


# In[14]:

def calculate_loss(sess, Xt, yt, size=1000, step=10):
    preds, probs, sample_idx = predict_proba(sess, Xt, size=size, step=step)

    loss, acc, top5acc = sess.run([cost, accuracy, top5accuracy], feed_dict={pred: preds, y: yt[sample_idx]})

    return loss, acc, top5acc


# In[15]:

rnn_shape = (-1, n_steps, n_input)
rnn_resize = lambda X: X.reshape(rnn_shape)


# In[16]:

# Launch the graph
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config = tf.ConfigProto(gpu_options=gpu_options)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(init)
    prev_acc = 0.0
    for ep in range(epochs):
        print("==== EPOCH {} ====".format(ep))
        step = 1
        for _X, _Y in __iterate_minibatches(X, Y, batch_size):
            _X = rnn_resize(_X)
            sess.run(optimizer, feed_dict={x: _X, y: _Y})
            if step % display_step == 0:
                loss, acc, top5acc = calculate_loss(sess, rnn_resize(Xt), Yt)
                print("Iter " + str(step) + ", Loss= " +                       "{:.4f}".format(loss) + ", Acc= " +                       "{:.4f}".format(acc) + ", Top-5 Acc= " +                       "{:.4f}".format(top5acc))
            step += 1

        loss, acc, top5acc = calculate_loss(sess, rnn_resize(Xt), Yt, size=Xt.shape[0])
        print("====================================")
        print("Epoch {}: Loss={} Acc={} Top-5 Acc={}".format(ep, loss, acc, top5acc))
        print("====================================")
        if acc > prev_acc:
            prev_acc = acc
            saver.save(sess, "conv_rnn_prelogits/{}.tfmodel".format(args.name))
            print("++++ Saved BEST ACC") 


# ## CNN-RNN Ensemble

# In[27]:

probas_training = []
probas_testing = []
preds_training = []
preds_testing = []


# In[44]:

# Load RNN Predictions
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(init)
    for i in range(3,5):
        X, Y = np.load("features/model{}.PreLogitsFlatten.X.npy".format(i)), np.load("{}/y.npy".format(tinyImageNetDir))
        Xt, Yt = np.load("features/model{}.PreLogitsFlatten.Xt.npy".format(i)), np.load("{}/yt.npy".format(tinyImageNetDir))
        saver.restore(sess, "conv_rnn_prelogits/cnn_prelogits_rnn_model_{}.tfmodel".format(i))
    #     print(calculate_loss(sess, rnn_resize(Xt), Yt, size=Xt.shape[0]))
    #     calculate_loss()
        res_training = predict_proba(sess, rnn_resize(X), size=X.shape[0], randomize=False, step=250)
        res_testing = predict_proba(sess, rnn_resize(Xt), size=Xt.shape[0], randomize=False, step=250)
        preds_training += [res_training[0]]
        preds_testing += [res_testing[0]]
        probas_training += [res_training[1]]
        probas_testing += [res_testing[1]]


# In[28]:

# Load CNN Predictions
# probas_training += [ np.load("features/{}.Predictions.X.npy".format(model)) for model in ["model{}".format(i) for i in range(2,5)] ]
probas_testing += [ np.load("features/{}.Predictions.Xt.npy".format(model)) for model in ["model{}".format(i) for i in range(2,5)] ]


# In[29]:

probas_testing += [ np.load("probs/{}.probs.npy".format("model1")) ]


# In[35]:

probas_testing += [ np.load("probs/cnn_rnn_rot_{}_1.probs.npy".format(i)) for i in range(4) ]


# In[312]:

g = tf.reset_default_graph()
ens = EnsembleNetwork("test", 1636, 100)
    
# Load FC(RNN+CNN_Logits) Predictions
with tf.Session() as sess:    
    for i in range(2,5):
        X, Y = np.load("features/model{}.PreLogitsFlatten.X.npy".format(i)), np.load("{}/y.npy".format(tinyImageNetDir))
        Xt, Yt = np.load("features/model{}.PreLogitsFlatten.Xt.npy".format(i)), np.load("{}/yt.npy".format(tinyImageNetDir))

        ensX = np.hstack((probas_training[i-2], X))
        ensXt = np.hstack((probas_testing[i-2], Xt))

        print(ensX.shape, ensXt.shape)

        ens.saver.restore(sess, "conv_rnn_prelogits/{}.tfmodel".format("inception_rnn_{}".format(i)))
        print(ens.calculate_loss(sess, ensXt, Yt, size=Xt.shape[0]))
        probas_training.append(ens.predict_proba(sess, ensX, size=ensX.shape[0], randomize=False)[1])
        probas_testing.append(ens.predict_proba(sess, ensXt, size=ensXt.shape[0], randomize=False)[1])


# In[313]:

g = tf.reset_default_graph()
ens = EnsembleNetwork("test", 1536, 100)
    
# Load FC(CNN_Logits) Predictions
with tf.Session() as sess:    
    for i in range(2,5):
        X, Y = np.load("features/model{}.PreLogitsFlatten.X.npy".format(i)), np.load("{}/y.npy".format(tinyImageNetDir))
        Xt, Yt = np.load("features/model{}.PreLogitsFlatten.Xt.npy".format(i)), np.load("{}/yt.npy".format(tinyImageNetDir))

        ens.saver.restore(sess, "conv_rnn_prelogits/{}.tfmodel".format("fc_inception_{}_logits".format(i)))
        print(ens.calculate_loss(sess, Xt, Yt, size=Xt.shape[0]))
        probas_training.append(ens.predict_proba(sess, X, size=X.shape[0], randomize=False)[1])
        probas_testing.append(ens.predict_proba(sess, Xt, size=Xt.shape[0], randomize=False)[1])


# In[30]:

probs_test = tf.placeholder(tf.float32, shape=[None, 100])
y_test = tf.placeholder(tf.float32, shape=[None, 100])
correct_prediction_test = tf.equal(tf.argmax(probs_test,1), tf.argmax(y_test,1))
top_5_correct_prediction_test = tf.nn.in_top_k(probs_test, tf.argmax(y_test,1), 5)
accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
top_5_accuracy_test = tf.reduce_mean(tf.cast(top_5_correct_prediction_test, tf.float32))


# In[31]:

ensemble_models = lambda enslist: reduce(lambda p1, p2: p1 + p2, enslist) / len(enslist)


# In[139]:

with tf.Session() as sess:
#     idx = range(3,6)+[12]
#     idx = range(4) #range(len(probas_testing))
    idx = [1,2,3,5] #,2,3,5
    enslist = [probas_testing[i] for i in idx]
#     enslist = probas_testing
    ens = ensemble_models(enslist)
    corpred, top5corpred, acc, top5acc = sess.run([correct_prediction_test, top_5_correct_prediction_test, accuracy_test, top_5_accuracy_test], feed_dict={probs_test: ens, y_test: Yt})
    print(acc, top5acc)


# In[113]:

import pandas as pd


# In[123]:

top1_dist = np.array([ 
         100*corpred[Yt.argmax(axis=1) == label].astype(np.float32).sum() / corpred[Yt.argmax(axis=1) == label].shape[0] for label in range(0,100) 
    ])


# In[125]:

top5_dist = np.array([ 
         100*top5corpred[Yt.argmax(axis=1) == label].astype(np.float32).sum() / corpred[Yt.argmax(axis=1) == label].shape[0] for label in range(0,100) 
    ])


# In[130]:

np.save("cnn_rnn_top1_dist.npy", top1_dist)
np.save("cnn_rnn_top5_dist.npy", top5_dist)


# In[ ]:

import time
with tf.Session() as sess:
    top_weights = []
    top_acc_x = 0.0
    top_acc_xt = 0.0
    for x in np.arange(0.1, 10, 0.1):
        for y in np.arange(0.1, 10, 0.1):
            weights = [x,y]
            if x == y: continue
            probas_ensemble_x = reduce(lambda (w1, p1), (w2, p2): w1 * p1 + w2 * p2, zip(weights, probas_training)) / (x+y)
            probas_ensemble_xt = reduce(lambda (w1, p1), (w2, p2): w1 * p1 + w2 * p2, zip(weights, probas_testing)) / (x+y)
            acc, top5acc_x = sess.run([accuracy_test, top_5_accuracy_test], feed_dict={probs_test: probas_ensemble_x, y_test: Y})
            acc, top5acc_xt = sess.run([accuracy_test, top_5_accuracy_test], feed_dict={probs_test: probas_ensemble_xt, y_test: Yt})
            if top5acc_x > top_acc_x:
                top_acc_x = top5acc_x
                top_acc_xt = top5acc_xt
                top_weights = weights

            update_screen("\rX Acc={:.4f} Top Acc={:.4f} Xt: Acc={:.4f} Top Acc={:.4f} weights={}".format(top5acc_x, top_acc_x, top5acc_xt, top_acc_xt, top_weights))


# In[ ]:

probas_training[1][12].min()


# In[ ]:

print(acc, top5acc)


# In[314]:

class EnsembleNetwork():
    def __init__(self, name, n_in, n_out, start_learning_rate=0.1, end_learning_rate=0.0001):
        self.name = name
        with tf.name_scope("Ensemble") as scope:
            self.X = tf.placeholder(tf.float32, shape=[None, n_in], name="X")
            self.Y = tf.placeholder(tf.float32, shape=[None, n_out], name="Y")
#             self.PreLogits = slim.fully_connected(self.X, 512, activation_fn=None, scope='PreLogits')
            self.Logits = slim.fully_connected(self.X, 100, activation_fn=None, scope='Logits')
            self.Probs = tf.nn.softmax(self.Logits)

            # Define loss and optimizer
#             self.GlobalStep = tf.Variable(0, trainable=False)
#             self.LearningRate = tf.train.polynomial_decay(start_learning_rate, self.GlobalStep, 100000, end_learning_rate, power=0.5)
# #              = tf.train.exponential_decay(init_learning_rate, self.GlobalStep, 100000, 0.96, staircase=True)
            self.Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.Logits, self.Y))
            self.Optimizer = tf.train.AdamOptimizer(epsilon=.1, learning_rate=0.01).minimize(self.Cost)

            # Evaluate model
            self.CorrectPred = tf.equal(tf.argmax(self.Logits,1), tf.argmax(self.Y,1))
            self.Top5CorrectPred = tf.nn.in_top_k(self.Probs, tf.argmax(self.Y,1), 5)

            self.Accuracy = tf.reduce_mean(tf.cast(self.CorrectPred, tf.float32))
            self.Top5Accuracy = tf.reduce_mean(tf.cast(self.Top5CorrectPred, tf.float32))
            
            self.saver = tf.train.Saver()
    
    def __iterate_minibatches(self, _X,_y, size):
        if _X.shape[0] % size > 0:
            raise "The minibatch size should be a divisor of the batch size."

        idx = np.arange(_X.shape[0]).astype(np.int32)
        np.random.shuffle(idx) # in-place shuffling
        for i in range(_X.shape[0] / size):
            # To randomize the minibatches every time
            _idx = idx[i*size:(i+1)*size]
            _X_small = _X[_idx]
            _y_small = _y[_idx]
            yield _X_small, _y_small
    
    def predict_proba(self, sess, Xt, size=1000, step=10, randomize=True):
        preds, probs = [], []
        idx = range(0, Xt.shape[0])
        sample_idx = random.sample(idx, size) if randomize else idx
        for i in range(size / step):
            _pred, _prob = sess.run([self.Logits, self.Probs], feed_dict={self.X: Xt[sample_idx[i*step:(i+1)*step]]})
            preds.append(_pred)
            probs.append(_prob)

        preds = np.vstack(preds)
        probs = np.vstack(probs)

        return preds, probs, sample_idx
    
    def calculate_loss(self, sess, Xt, yt, size=1000, step=10):
        preds, probs, sample_idx = self.predict_proba(sess, Xt, size=size, step=step)

        loss, acc, top5acc = sess.run([self.Cost, self.Accuracy, self.Top5Accuracy], feed_dict={self.Logits: preds, self.Y: yt[sample_idx]})

        return loss, acc, top5acc
        
    def train(self, sess, X, Y, Xt, Yt, epochs=100, batch_size=100):
        sess.run(tf.initialize_all_variables())
        self.prev_acc = 0.0
        for ep in range(epochs):
            print("==== EPOCH {} ====".format(ep))
            step = 1
            for _X, _Y in self.__iterate_minibatches(X, Y, batch_size):
                sess.run(self.Optimizer, feed_dict={self.X: _X, self.Y: _Y})
                if step % display_step == 0:
                    loss, acc, top5acc = self.calculate_loss(sess, Xt, Yt)
                    print("Iter " + str(step) + ", Loss= " +                           "{:.4f}".format(loss) + ", Acc= " +                           "{:.4f}".format(acc) + ", Top-5 Acc= " +                           "{:.4f}".format(top5acc))
                step += 1

            loss, acc, top5acc = self.calculate_loss(sess, Xt, Yt, size=Xt.shape[0])
            print("====================================")
            print("Epoch {}: Loss={} Acc={} Top-5 Acc={}".format(ep, loss, acc, top5acc))
            print("====================================")
            if top5acc > self.prev_acc:
                self.prev_acc = top5acc
                self.saver.save(sess, "conv_rnn_prelogits/{}.tfmodel".format(self.name))
                print("++++ Saved BEST ACC")


# In[361]:

ensX = ensemble_models(probas_training[3:9])
ensXt = ensemble_models(probas_testing[3:9])


# In[362]:

ensX = np.hstack((ensX / ensX.max(), X))
ensXt = np.hstack((ensXt / ensXt.max(), Xt))


# In[363]:

g = tf.reset_default_graph()
ens = EnsembleNetwork("fc_ensemble_all_augmented", ensX.shape[1], 100)


# In[364]:

sess = tf.Session()


# In[365]:

ens.train(sess, ensX, Y, ensXt, Yt, epochs=30, batch_size=50)


# In[366]:

ens.saver.restore(sess, "conv_rnn_prelogits/{}.tfmodel".format("fc_ensemble_all_augmented"))


# In[368]:

print(ens.calculate_loss(sess, ensXt, Yt, size=Xt.shape[0]))


# In[369]:

predXt = ens.predict_proba(sess, ensXt, size=ensXt.shape[0], randomize=False)


# In[370]:

probas2 = [ ensemble_models(probas_testing[3:9]), predXt]


# In[378]:

probas2_ensemble = ensemble_models(probas2)


# In[381]:

ensemble_models = lambda enslist: reduce(lambda p1, p2: p1 + p2, enslist) / len(enslist)
with tf.Session() as sess:
    idx = range(3,6)+range(6,9)
    acc, top5acc = sess.run([accuracy_test, top_5_accuracy_test], feed_dict={probs_test: probas2_ensemble, y_test: Yt})
    print(acc, top5acc)


# In[ ]:



