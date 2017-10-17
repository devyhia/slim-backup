
# coding: utf-8

# In[1]:

import numpy as np
import random


# In[2]:

from sklearn import metrics
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
slim = tf.contrib.slim

from sklearn.metrics import log_loss as nll


# In[46]:

alexNetPred_X_Original = np.load('Prediction.X.alexnet.distracted_driver.original.1.npy')
alexNetPred_Xt_Original = np.load('Prediction.Xt.alexnet.distracted_driver.original.1.npy')
alexNetPred_X_Segmented = np.load('Prediction.X.alexnet.distracted_driver.segmented.1.npy')
alexNetPred_Xt_Segmented = np.load('Prediction.Xt.alexnet.distracted_driver.segmented.1.npy')
alexNetPred_X_Face = np.load('Prediction.X.alexnet.distracted_driver.face.1.npy')
alexNetPred_Xt_Face = np.load('Prediction.Xt.alexnet.distracted_driver.face.1.npy')
alexNetPred_X_Hands = np.load('Prediction.X.alexnet.distracted_driver.hands.1.npy')
alexNetPred_Xt_Hands = np.load('Prediction.Xt.alexnet.distracted_driver.hands.1.npy')
alexNetPred_X_HandsAndFace = np.load('Prediction.X.alexnet.distracted_driver.hands_and_face.1.npy')
alexNetPred_Xt_HandsAndFace = np.load('Prediction.Xt.alexnet.distracted_driver.hands_and_face.1.npy')
inceptionV3Pred_X_Original = np.load('Prediction.X.inceptionV3.distracted_driver.original.1.npy')
inceptionV3Pred_Xt_Original = np.load('Prediction.Xt.inceptionV3.distracted_driver.original.1.npy')
inceptionV3Pred_X_Segmented = np.load('Prediction.X.inceptionV3.distracted_driver.segmented.1.npy')
inceptionV3Pred_Xt_Segmented = np.load('Prediction.Xt.inceptionV3.distracted_driver.segmented.1.npy')


# In[45]:

y = np.load('cache/y.original.npy')
yt = np.load('cache/yt.original.npy')


# In[47]:

ensemble_train = [alexNetPred_X_Original, alexNetPred_X_Segmented, inceptionV3Pred_X_Original, inceptionV3Pred_X_Segmented, alexNetPred_X_Face, alexNetPred_X_Hands, alexNetPred_X_HandsAndFace] # 
ensemble_test = [alexNetPred_Xt_Original, alexNetPred_Xt_Segmented, inceptionV3Pred_Xt_Original, inceptionV3Pred_Xt_Segmented, alexNetPred_Xt_Face, alexNetPred_Xt_Hands, alexNetPred_Xt_HandsAndFace] #
ensemble = reduce(lambda prev, curr: prev + curr, ensemble_test[1:], ensemble_test[0]) / len(ensemble_test)


# In[48]:

metrics.accuracy_score(y_pred=alexNetPred_Xt_Original.argmax(axis=1), y_true=yt.argmax(axis=1))


# In[49]:

metrics.accuracy_score(y_pred=ensemble.argmax(axis=1), y_true=yt.argmax(axis=1))


# In[50]:

metrics.log_loss(y_pred=ensemble, y_true=yt)


# In[23]:

ensemble_original = (alexNetPred_Xt_Original + inceptionV3Pred_Xt_Original)/2
ensemble_segmented = (alexNetPred_Xt_Segmented + inceptionV3Pred_Xt_Segmented)/2
ensemble_alexNet = (alexNetPred_Xt_Original + alexNetPred_Xt_Segmented)/2
ensemble_inceptionV3 = (inceptionV3Pred_Xt_Original + inceptionV3Pred_Xt_Segmented)/2


# In[24]:

print metrics.accuracy_score(y_pred=ensemble_original.argmax(axis=1), y_true=yt.argmax(axis=1))
print metrics.log_loss(y_pred=ensemble_original, y_true=yt)

print metrics.accuracy_score(y_pred=ensemble_segmented.argmax(axis=1), y_true=yt.argmax(axis=1))
print metrics.log_loss(y_pred=ensemble_segmented, y_true=yt)

print metrics.accuracy_score(y_pred=ensemble_alexNet.argmax(axis=1), y_true=yt.argmax(axis=1))
print metrics.log_loss(y_pred=ensemble_alexNet, y_true=yt)

print metrics.accuracy_score(y_pred=ensemble_inceptionV3.argmax(axis=1), y_true=yt.argmax(axis=1))
print metrics.log_loss(y_pred=ensemble_inceptionV3, y_true=yt)


# In[42]:

metrics.log_loss(y_pred=ensemble, y_true=yt)


# In[19]:

yt_seg = np.load('cache/yt.segmented.npy')


# In[90]:

import argparse

parser = argparse.ArgumentParser(description='Simple Network.')
parser.add_argument('--gpu', default=3, help='GPU to use for train')
parser.add_argument('--name', default="ensemble_network_distracted_driver.1", help='Name of the model to use for train')
args, unknown_args = parser.parse_known_args()


# In[91]:

import Shared

Shared.select_gpu(args.gpu)


# In[128]:

class SimpleNetwork():
    def __init__(self, name, n_in, n_out, start_learning_rate=0.1, end_learning_rate=0.0001):
        tf.reset_default_graph()
        self.name = name
        with tf.name_scope("SimpleNetwork") as scope:
            self.X = []
            for i in range(n_in):
                self.X += [tf.placeholder(tf.float32, shape=[None, n_out], name="X{}".format(i))]

            self.Y = tf.placeholder(tf.float32, shape=[None, n_out], name="Y")
            
            self.W = [
                slim.variable('W{}'.format(i),
                             shape=[1],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
                for i in range(n_in)
            ]
            
            self.Logits = self.W[0] * self.X[0]
            for i in range(n_in)[1:]:
                self.Logits = self.Logits + self.weights[i] * self.X[i]
            
            self.Total = self.W[0]
            for w in self.W[1:]:
                self.Total += w
                
            self.Probs = tf.nn.softmax(self.Logits)

            self.Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.Logits, self.Y))
            self.Optimizer = tf.train.AdamOptimizer(epsilon=.1, learning_rate=0.001).minimize(self.Cost)

            # Evaluate model
            self.CorrectPred = tf.equal(tf.argmax(self.Logits,1), tf.argmax(self.Y,1))
            self.Top5CorrectPred = tf.nn.in_top_k(self.Probs, tf.argmax(self.Y,1), 5)

            self.Accuracy = tf.reduce_mean(tf.cast(self.CorrectPred, tf.float32))
            self.Top5Accuracy = tf.reduce_mean(tf.cast(self.Top5CorrectPred, tf.float32))
            
            self.saver = tf.train.Saver()
    
    def __iterate_minibatches(self, X,y, size):
        '''Iterates over X and y in batches of a certain size'''
        # if X.shape[0] % size > 0:
        #     raise "The minibatch size should be a divisor of the batch size."
        
        total = X[0].shape[0]
        idx = np.arange(total)
        np.random.shuffle(idx) # in-place shuffling
        for i in range(total / size):
            # To randomize the minibatches every time
            _idx = idx[i*size:(i+1)*size]
            yield [_X[_idx] for _X in X], y[_idx]
    
    def predict_proba(model, sess, X, step=10):
        fc3ls = []

        size = X[0].shape[0]
        sample_idx = random.sample(range(0, size), size)
        reverse_idx = list(map(sample_idx.index, range(0,size)))
        for i in range(int(np.ceil(float(size) / step))):
            feed_dict={}
            for ModelX, BatchX in zip(model.X, X):
                feed_dict[ModelX] = BatchX
                
            fc3l = sess.run(model.Logits, feed_dict=feed_dict)
            fc3ls.append(fc3l)
        
        preds = np.vstack(fc3ls)
        probs = sess.run(model.Probs, feed_dict={model.Logits: preds})

        return preds[reverse_idx], probs[reverse_idx]
    
    def calculate_loss(self, sess, Xt, yt, size=1000, step=10):
        preds, probs = self.predict_proba(sess, Xt, step=step)

        loss, acc, top5acc = sess.run([self.Cost, self.Accuracy, self.Top5Accuracy], feed_dict={self.Logits: preds, self.Y: yt})

        return loss, acc, top5acc
        
    def train(self, sess, X, Y, Xt, Yt, epochs=100, batch_size=100, display_step=25):
        sess.run(tf.initialize_all_variables())
        self.prev_acc = 10
        for ep in range(epochs):
            print("==== EPOCH {} ====".format(ep))
            step = 1
            for _X, _Y in self.__iterate_minibatches(X, Y, batch_size):
                feed_dict={self.Y: _Y}
                for ModelX, BatchX in zip(self.X, _X):
                    feed_dict[ModelX] = BatchX

                sess.run(self.Optimizer, feed_dict=feed_dict)
                if step % display_step == 0:
                    loss, acc, top5acc = self.calculate_loss(sess, Xt, Yt)
                    print("Iter " + str(step) + ", Loss= " + "{:.4f}".format(loss) + ", Acc= " + "{:.4f}".format(acc) + ", Top-5 Acc= " + "{:.4f}".format(top5acc))
                step += 1

            loss, acc, top5acc = self.calculate_loss(sess, Xt, Yt, size=Xt[0].shape[0])
            print("====================================")
            print("Epoch {}: Loss={} Acc={} Top-5 Acc={}".format(ep, loss, acc, top5acc))
            print("====================================")
            if loss < self.prev_acc:
                self.prev_acc = loss
                self.saver.save(sess, "{}.tfmodel".format(self.name))
                print("++++ Saved BEST LOSS")


# In[125]:

X = np.hstack(ensemble_train)
Xt = np.hstack(ensemble_test)


# In[129]:

net = SimpleNetwork(args.name, len(ensemble_train), yt.shape[1])


# In[130]:

with tf.Session() as sess:
    net.train(sess, ensemble_train, y, ensemble_test, yt, batch_size=50, epochs=30)


# In[ ]:



