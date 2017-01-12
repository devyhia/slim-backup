
# coding: utf-8

# In[1]:

import argparse

parser = argparse.ArgumentParser(description='Simple Network.')
parser.add_argument('--gpu', default=2, help='GPU to use for train')
parser.add_argument('--name', default="simple_network_1", help='Name of the model to use for train')
args, unknown_args = parser.parse_known_args()


# In[2]:

import os, random, sys
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
slim = tf.contrib.slim

from sklearn.metrics import log_loss as nll

import Shared

Shared.select_gpu(args.gpu)


# In[3]:

class SimpleNetwork():
    def __init__(self, name, n_in, n_out, start_learning_rate=0.1, end_learning_rate=0.0001):
        tf.reset_default_graph()
        self.name = name
        with tf.name_scope("SimpleNetwork") as scope:
            self.X = tf.placeholder(tf.float32, shape=[None, n_in], name="X")
            self.Y = tf.placeholder(tf.float32, shape=[None, n_out], name="Y")
#             self.PreLogits = slim.fully_connected(self.X, 512, activation_fn=None, scope='PreLogits')
            self.Logits = slim.fully_connected(self.X, n_out, activation_fn=None, scope='Logits')
            self.Probs = tf.nn.softmax(self.Logits)

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
        
    def train(self, sess, X, Y, Xt, Yt, epochs=100, batch_size=100, display_step=25):
        sess.run(tf.initialize_all_variables())
        self.prev_acc = 10
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
            if loss < self.prev_acc:
                self.prev_acc = loss
                self.saver.save(sess, "inception_catdogs/{}.tfmodel".format(self.name))
                print("++++ Saved BEST LOSS")


# In[4]:

y = np.array([[1,0]]*12500 + [[0,1]]*12500)


# In[23]:

models = [
"cvd_inception_resnet_v2_depth_100_16",
"cvd_inception_resnet_v2_depth_150",   
"cvd_inception_v3_aux_logits",         
"cvd_inception_v3_deep_logits_aux_logits",
"cvd_inception_v3_deep_logits",        
"cvd_inception_v3_depth_150_50",       
"cvd_inception_v3_depth_200",          
"cvd_inception_v3_depth_256_64",       
"cvd_inception_v3_depth_256",          
"cvd_inception_v4_depth_150",          
"cvd_model_inception_1",               
"cvd_model_inception_deep_fc_1",       
"cvd_model_inception_deep_fc_fulldata_1.Epoch.0",
"cvd_model_inception_deep_fc_fulldata_1.Epoch.1",
"cvd_model_inception_deep_fc_fulldata_2.Epoch.0",
"cvd_model_inception_deep_fc_fulldata_2.Epoch.1",
"cvd_model_inception_deep_logits_1",   
"cvd_model_inception_deep_logits_mul_1",
"cvd_model_inception_resnet_v2_fulldata_2",
"cvd_model_inception_resnet_v2_fulldata",
"cvd_model_inception_v3_fulldata_2",   
"cvd_model_inception_v3_fulldata",     
"cvd_model_resnet_101_fulldata",       
"cvd_model_resnet_12",                 
"cvd_model_resnet_14",                 
"cvd_model_resnet_152_fulldata",       
"cvd_model_resnet_15",                 
"cvd_model_resnet_16",                 
"cvd_model_resnet_17",                 
"cvd_model_resnet_18",                 
"cvd_model_resnet_19",                 
"cvd_model_resnet_20",                 
"cvd_model_resnet_21",                 
"cvd_model_resnet_bag_0",              
"cvd_model_resnet_bag_1",              
"cvd_model_resnet_bag_2",              
"cvd_model_resnet_bag_3"
]


# In[5]:

models = ['cvd_model_inception_1', 'cvd_model_inception_deep_logits_1','cvd_model_inception_deep_logits_mul_1','cvd_model_resnet_12','cvd_model_resnet_14','cvd_model_resnet_15','cvd_model_resnet_16','cvd_model_resnet_17','cvd_model_resnet_18','cvd_model_resnet_19','cvd_model_resnet_20','cvd_model_inception_deep_fc_1','cvd_model_inception_deep_fc_fulldata_1.Epoch.0','cvd_model_inception_deep_fc_fulldata_2.Epoch.0','cvd_inception_v3_deep_logits','cvd_inception_v3_depth_256','cvd_inception_v3_depth_256_64']


# In[6]:

X_models = [ np.load("CatVsDogs.X.{}.npy".format(model)) for model in models ]
X = np.hstack([ np.load("CatVsDogs.X.{}.npy".format(model))[:,0].reshape(-1,1) for model in models ])


# ### Make Sure All Good

# In[7]:

for m in X_models:
    print(nll(y, m))


# #### Neural Network

# In[8]:

C = 10000
Xt, yt = X[C:], y[C:]
# X, y = X[:C], y[:C]


# In[9]:

net = SimpleNetwork(args.name, Xt.shape[1], y.shape[1])


# In[10]:

sess = tf.Session()


# In[11]:

net.train(sess, X, y, Xt, yt, batch_size=100, epochs=15)


# #### Make Submission

# In[12]:

import pandas as pd


# In[13]:

_Xt = np.hstack([ np.load("CatVsDogs.Xt.{}.npy".format(model))[:,0].reshape(-1,1) for model in models ])


# In[14]:

logits, probs_ens, idx = net.predict_proba(sess, _Xt, randomize=False, size=_Xt.shape[0])


# In[15]:

submission = np.hstack((np.arange(1, probs_ens.shape[0]+1, 1).reshape(-1, 1), probs_ens[:, 0].reshape(-1,1)))
df_submission = pd.DataFrame(submission, columns=["id", "label"])
df_submission["id"] = df_submission["id"].astype(np.int)
df_submission.to_csv("submission_{}.csv".format("deep_ensemble_nn_7_clfs_15_epochs"), index=False)

