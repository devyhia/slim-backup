
# coding: utf-8

# In[9]:

import argparse
import cv2
import re

parser = argparse.ArgumentParser(description='InceptionV4 Cats-Vs-Dogs Network.')
parser.add_argument('--gpu', default=1, help='GPU to use for train')
parser.add_argument('--name', default="cats_vs_dogs_model", help='Name of the RNN model to use for train')
parser.add_argument('--resume', default="no", help='Resume from previous checkpoint')
parser.add_argument('--test', default="no", help='Generate test predictions')
parser.add_argument('--bag', default="no", help='Bagging split')
args, unknown_args = parser.parse_known_args()


# In[10]:

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from os import listdir, environ
import pandas as pd
from IPython import embed
import sys
import random
import time
import os

from datasets import flowers
# from nets import resnet_v1

from nets import inception
from preprocessing import inception_preprocessing


from datasets import dataset_utils
# from helpers import *

slim = tf.contrib.slim


# In[11]:

from datasets import dataset_utils

slim = tf.contrib.slim

BEST_ACC = 0
CHECKPOINTS_DIR = "checkpoints/"
IMAGE_DIMENSION = 299

class InceptionV4:
    def __init__(self, model_name, isTesting=False):
        tf.reset_default_graph()
        
        self.X = tf.placeholder(tf.float32, shape=[None, IMAGE_DIMENSION,IMAGE_DIMENSION,3], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        self.name = model_name
        self.ImageNetMean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        self.X_Norm = self.X - self.ImageNetMean

        self.summaries = []
        
        self.__model()
        
        self.saver = tf.train.Saver()
        
        self.best_accuracy = 0.0

        # Just in case a previous run was closed using a halt file.
        if os.path.isfile("{}.halt".format(self.name)):
            os.remove("{}.halt".format(self.name))

        # if weights is not None and sess is not None:
        #     self.load_weights(sess, weights)
    
    def __get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training."""
#         checkpoint_exclude_scopes=["resnet_v1_101/logits"]
        checkpoint_exclude_scopes=["InceptionV4/Logits", "InceptionV4/AuxLogits"]

        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        return slim.assign_from_checkpoint_fn(
          os.path.join(CHECKPOINTS_DIR, 'inception_v4.ckpt'),
          variables_to_restore)
        
    def __model(self):
        # Create the model, use the default arg scope to configure the batch norm parameters.
#         with slim.arg_scope(resnet_v1.resnet_arg_scope()):
#             self.logits, self.end_points = resnet_v1.resnet_v1_101(self.X_Norm, 2, is_training=True)
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            self.logits, self.end_points = inception.inception_v4(self.X_Norm, num_classes=2)


        # Specify the loss function:
#         slim.losses.softmax_cross_entropy(self.logits, self.y)
#         self.total_loss = slim.losses.get_total_loss()
        self.total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))
        
        self.probs = self.end_points["Predictions"]

        # Calculate Accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.y, 1))
        self.wrong_prediction = tf.not_equal(tf.argmax(self.probs, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.summaries += [tf.histogram_summary('Correct Predictions', tf.cast(self.correct_prediction, tf.int32))]
        self.summaries += [tf.histogram_summary('Predictions', tf.argmax(self.probs, 1))]
        self.summaries += [tf.histogram_summary('Training Labels', tf.argmax(self.y, 1))]
        self.summaries += [tf.histogram_summary('Activations', self.probs)]

        self.summaries += [tf.scalar_summary('Loss', self.total_loss)]
        self.summaries += [tf.scalar_summary('Accuracy', self.accuracy)]
        
        # Confusing Images
        self.confusing_images = tf.where(self.wrong_prediction)
        self.confusing_images = tf.gather(self.X, self.confusing_images)
        self.confusing_images = tf.reshape(self.confusing_images, [-1, IMAGE_DIMENSION, IMAGE_DIMENSION, 3])
        
        self.summaries += [tf.image_summary('Confusing Images', self.confusing_images, max_images=25)]


    def __iterate_minibatches(self, X,y, size):
        if X.shape[0] % size > 0:
            raise "The minibatch size should be a divisor of the batch size."

        idx = np.arange(X.shape[0])
        np.random.shuffle(idx) # in-place shuffling
        for i in range(X.shape[0] / size):
            # To randomize the minibatches every time
            _idx = idx[i*size:(i+1)*size]
            yield X[_idx], y[_idx]

    def __update_screen(self, msg):
        sys.stdout.write(msg)
        sys.stdout.flush()

    def resize_images(self, X):
        return np.array([ imresize(X[i], (IMAGE_DIMENSION,IMAGE_DIMENSION)) for i in range(X.shape[0])])

    def calculate_loss(self, sess, Xt, yt, size=1000, step=10):
        fc3ls = None
        sample_idx = random.sample(range(0, Xt.shape[0]), size)
        for i in range(size / step):
            [fc3l] = sess.run([self.logits], feed_dict={self.X: Xt[sample_idx[i*step:(i+1)*step]], self.y: yt[sample_idx[i*step:(i+1)*step]]})
            if i == 0:
                fc3ls = fc3l
            else:
                fc3ls = np.vstack((fc3ls, fc3l))

        loss, accuracy, summary = sess.run([self.total_loss, self.accuracy, self.tf_summary], feed_dict={self.X: Xt[sample_idx], self.logits: fc3ls, self.y: yt[sample_idx]})

        return loss, accuracy, summary

    def __graceful_halt(self, t_start_training):
        # Save Trained Model
        p = self.saver.save(sess, "{}.tfmodel".format(self.name))
        t = time.time() - t_start_training
        print("++ Training: END -- Exec. Time={0:.0f}m {1:.0f}s Model Path={2:s}".format(t / 60, t % 60, p))


    def train(self, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
        print("++ Training with {} epochs and {} minibatch size.".format(epochs, minibatch_size))
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.tf_summary = tf.merge_summary(self.summaries)
        self.tf_logs = tf.train.SummaryWriter("logs/{}".format(self.name), sess.graph, flush_secs=30)
        
        # Specify the optimizer and create the train op:
        self.train_step = slim.learning.create_train_op(self.total_loss, self.optimizer)
        
        self.init_fn = self.__get_init_fn()
        
        # Initialize Model ...
        sess.run(tf.initialize_all_variables())
        self.init_fn(sess)
        
        # Resize Validation Once ...
        val_X_res = self.resize_images(val_X)
        
        lr = 0.01
        lr_step = (lr - 0.0001) / epochs
        lr += lr_step
        
        step = 0

        t_start_training = time.time()
        for i in range(epochs):
            t_start_epoch = time.time()
            t_start_minibatch = time.time()
            print("+++ Epoch 1: START")
            cnt = 0
            
            lr -= lr_step
            
            for _X, _y in self.__iterate_minibatches(X, y, minibatch_size):
                # Resize Training Batches (to avoid consuming much memeory)
                cnt += 1
                _X_res = self.resize_images(_X)
                self.__update_screen("\r++++ Mini Batch ({} out of {}): ".format(cnt, X.shape[0]/minibatch_size))
                sess.run([self.train_step], feed_dict={self.X: _X_res, self.y: _y, self.learning_rate: lr})
                if cnt % 25 == 0:
                    # loss, accuracy = sess.run([self.cross_entropy, self.accuracy], feed_dict={self.X: val_X_res[test_sample], self.y: val_y[test_sample]})
                    val_loss, val_accuracy, summary = self.calculate_loss(sess, val_X_res, val_y, 1000)
                    t = time.time() - t_start_minibatch
                    self.__update_screen(" Loss={0:.4f} Accuracy={1:.4f} Exec. Time={2:.0f}m {3:.0f}s\n".format(val_loss, val_accuracy, t / 60, t % 60))
                    t_start_minibatch = time.time()
                    
                    self.tf_logs.add_summary(summary, i * (X.shape[0] / minibatch_size) + cnt)

                    # Handle Close Signals
                    if os.path.isfile("{}.halt".format(self.name)):
                        self.__graceful_halt(t_start_training)
                        os.remove("{}.halt".format(self.name))
                        exit(0)

            self.__update_screen("\n")
            val_loss, val_accuracy, summary = self.calculate_loss(sess, val_X_res, val_y, val_X.shape[0])
#             self.tf_logs.add_summary(summary, epoch * (X.shape[0] / minibatch_size) + cnt)
            t = time.time() - t_start_epoch
            print("+++ Epoch {0:.0f}: END -- Loss={1:.4f} Accuracy={2:.4f} Exec. Time={3:.0f}m {4:.0f}s".format(i, val_loss, val_accuracy, t / 60, t % 60))

            # Always Save Best Accuracy
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.saver.save(sess, "{}.tfmodel".format(self.name))
                print("+++ Epoch {0:.0f}: END -- SAVED BEST ACC".format(i))

        self.__graceful_halt(t_start_training)
    
    def load_model(self, sess):
        sess.run(tf.initialize_all_variables())
        self.saver.restore(sess, "{}.tfmodel".format(self.name))

    def predict(self, sess, X):
        prob = sess.run(vgg.probs, feed_dict={vgg.X: [X]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print p, prob[p]
    
    def predict_proba(self, sess, X, step=10):
        fc3ls = []

        y = np.zeros((step, self.y.get_shape()[1]))

        size = X.shape[0]
        sample_idx = range(0, size)

        for i in range(size / step):
            self.__update_screen("\rpredict_proba: {} / {}".format(i*step, size))
            [fc3l] = sess.run([self.logits], feed_dict={self.X: X[sample_idx[i*step:(i+1)*step]]})
            fc3ls.append(fc3l)

        probs = sess.run(self.probs, feed_dict={self.logits: np.vstack(fc3ls)})

        return probs

    def load_weights(self, sess, weight_file):
        # This includes optimizer variables as well.
        init = tf.initialize_all_variables()
        sess.run(init)

        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i == 30:
                break
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

        # Transfer Learning
        # Initialize the last fully connected layer
        for par in self.parameters[-2:]:
            print("Param: {}".format(par.name))


# In[12]:

def update_screen(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    
def prepare_data():
    TRAIN_DIR = '/home/devyhia/cats.vs.dogs/train/'
    TEST_DIR = '/home/devyhia/cats.vs.dogs/test/'

    ROWS = IMAGE_DIMENSION
    COLS = IMAGE_DIMENSION
    CHANNELS = 3
    SLICE = 10000

    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
    train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
    
    train_images = train_dogs[:SLICE] + train_cats[:SLICE]
    valid_images = train_dogs[SLICE:] + train_cats[SLICE:]

    np.random.shuffle(train_images)
    np.random.shuffle(valid_images)

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
            if i%250 == 0: update_screen('\rProcessed {} of {}'.format(i, count))
        
        update_screen('\n')

        return data

    def get_label(path):
        return 1 if re.search("(cat|dog)\.(\d+)\.", path).group(1) == 'cat' else 0
    
    print("Prep data ...")
    X = prep_data(train_images)
    Xt = prep_data(valid_images)
    # test = prep_data(test_images)
    
#     pre_processing = tf.placeholder(tf.float32, shape=[None, IMAGE_DIMENSION,IMAGE_DIMENSION,3])
#     after_processing = tf.sub(tf.mul(pre_processing, 2.0/255.), 1.0)
    
#     def preprocess(sess, X, i, batch_size):
#         update_screen("\rPreprocessing {} ...".format(i))
#         return sess.run(after_processing, feed_dict={pre_processing: X[i*batch_size:(i+1)*batch_size]})
    
#     with tf.Session() as sess:
#         Xs, Xts = [], []
#         batch_size = 200
        
#         for i in range(X.shape[0] / batch_size):
#             Xs.append(preprocess(sess, X, i, batch_size))
#         update_screen("\n")
        
#         for i in range(Xt.shape[0] / batch_size):
#             Xts.append(preprocess(sess, Xt, i, batch_size))
#         update_screen("\n")
        
#         X = np.vstack(Xs)
#         Xt = np.vstack(Xts)
        
    print("Train shape: {}".format(X.shape))
    print("Valid shape: {}".format(Xt.shape))

    labels_train = [get_label(i) for i in train_images]
    labels_valid = [get_label(i) for i in valid_images]

    print(pd.DataFrame(labels_train, columns=["label"])["label"].value_counts())
    print(pd.DataFrame(labels_valid, columns=["label"])["label"].value_counts())

    y = np.zeros((X.shape[0], 2))
    yt = np.zeros((Xt.shape[0], 2))

    for i in range(y.shape[0]):
        y[i, labels_train[i]] = 1

    for i in range(yt.shape[0]):
        yt[i, labels_valid[i]] = 1

    # print(labels_train)
    # print(labels_valid)

    print(y)
    print(yt)

    print("X=", X.shape, "y=", y.shape)
    print("Xt=", Xt.shape, "yt=", yt.shape)

    return X, y, Xt, yt

def prepare_test_data():
    CACHE_PATH = "cache/Xt.npy"
    if os.path.isfile(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return range(1, data.shape[0]+1), data

    TEST_DIR = '../cats.vs.dogs/test/'

    ROWS = IMAGE_DIMENSION
    COLS = IMAGE_DIMENSION
    CHANNELS = 3

    file_key = lambda f: int(re.match("(\d+)\.jpg", f).group(1))
    test_images =  [TEST_DIR+i for i in sorted(os.listdir(TEST_DIR), key=file_key)]

    def read_image(file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
        return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
        ids = []

        for i, image_file in enumerate(images):
            data[i] = read_image(image_file)
            ids.append(re.search("(\d+)\.", image_file).group(1))

            if i%250 == 0: print('Processed {} of {}'.format(i, count))

        return ids, data

    ids, data = prep_data(test_images)

    np.save(CACHE_PATH, data)

    return ids, data


# In[ ]:

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

DATA_DIR = "/home/devyhia/vgg/"

model_name = args.name
model = InceptionV4(model_name)

sess = tf.Session()


# In[ ]:

def test():
    print("+++++ TESTING +++++")
    model.load_model(sess)
    ids, Xt = prepare_test_data()
    ids = np.array(ids).astype(np.int)

    prob = model.predict_proba(sess, Xt, step=50)
    
    np.save("CatVsDogs.Xt.{}.npy".format(model_name), prob)


# In[ ]:

def train():    
    print("+++++ TRAINING +++++")
    X, y, Xt, yt = prepare_data()
    model.train(sess, X, y, Xt, yt, minibatch_size=10)


# In[ ]:

if args.test == "no":
    train()
else:
    test()

