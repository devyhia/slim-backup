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
from nets import resnet_v1
# from preprocessing import inception_preprocessing

from datasets import dataset_utils
# from helpers import *

slim = tf.contrib.slim

import argparse
import cv2
import re

import CatVsDogs

DIM= 224
CHECKPOINTS_DIR= "checkpoints/"

def update_screen(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

# 'resnet_v1_{}.ckpt'.format(101 if args.version == '101' else '152')
# ["resnet_v1_101/logits"]

def get_init_fn(checkpoint_file, exclude):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes=exclude

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

    return slim.assign_from_checkpoint_fn(os.path.join(CHECKPOINTS_DIR, checkpoint_file), variables_to_restore)

def define_model(model, model_name, model_func):
    """Provides the outline of the defined model"""
    # Reset the current graph
    tf.reset_default_graph()

    # Define the model inputs / placeholders
    model.X = tf.placeholder(tf.float32, shape=[None, DIM,DIM,3], name='X')
    model.y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
    model.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    model.name = model_name
    # Just in case a previous run was closed using a halt file.
    if os.path.isfile("{}.halt".format(model.name)):
        os.remove("{}.halt".format(model.name))


    model.ImageNetMean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    model.X_Norm = model.X - model.ImageNetMean

    model.summaries = []

    model_func()

    model.total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.logits, model.y))
    model.probs = tf.nn.softmax(model.logits)

    # Calculate Accuracy
    model.correct_prediction = tf.equal(tf.argmax(model.probs, 1), tf.argmax(model.y, 1))
    model.wrong_prediction = tf.not_equal(tf.argmax(model.probs, 1), tf.argmax(model.y, 1))
    model.accuracy = tf.reduce_mean(tf.cast(model.correct_prediction, tf.float32))

    # Define Tensorboard Summaries
    model.summaries += [tf.histogram_summary('Correct Predictions', tf.cast(model.correct_prediction, tf.int32))]
    model.summaries += [tf.histogram_summary('Predictions', tf.argmax(model.probs, 1))]
    model.summaries += [tf.histogram_summary('Training Labels', tf.argmax(model.y, 1))]
    model.summaries += [tf.histogram_summary('Activations', model.probs)]

    model.summaries += [tf.scalar_summary('Loss', model.total_loss)]
    model.summaries += [tf.scalar_summary('Accuracy', model.accuracy)]

    # Confusing Images
    model.confusing_images = tf.where(model.wrong_prediction)
    model.confusing_images = tf.gather(model.X, model.confusing_images)
    model.confusing_images = tf.reshape(model.confusing_images, [-1, DIM, DIM, 3])

    model.summaries += [tf.image_summary('Confusing Images', model.confusing_images, max_images=25)]

    # Saving Functionality
    model.saver = tf.train.Saver()
    model.best_accuracy = 0.0

# def resize_images(model, X):
#     return np.array([ imresize(X[i], (224,224)) for i in range(X.shape[0])])

def iterate_minibatches(X,y, size):
    '''Iterates over X and y in batches of a certain size'''
    if X.shape[0] % size > 0:
        raise "The minibatch size should be a divisor of the batch size."

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx) # in-place shuffling
    for i in range(X.shape[0] / size):
        # To randomize the minibatches every time
        _idx = idx[i*size:(i+1)*size]
        yield X[_idx], y[_idx]

def calculate_loss(model, sess, Xt, yt, size=1000, step=10):
    '''Calculate the loss of the model on a random sample of a given size'''
    fc3ls = []
    sample_idx = random.sample(range(0, Xt.shape[0]), size)
    for i in range(size / step):
        # model.y: yt[sample_idx[i*step:(i+1)*step]]
        fc3l = sess.run(model.logits, feed_dict={model.X: Xt[sample_idx[i*step:(i+1)*step]]})
        fc3ls.append(fc3l)

    fc3ls = np.vstack(fc3ls)
    loss, accuracy, summary = sess.run([model.total_loss, model.accuracy, model.tf_summary], feed_dict={model.X: Xt[sample_idx], model.logits: fc3ls, model.y: yt[sample_idx]})

    return loss, accuracy, summary

def graceful_halt(model, t_start_training):
    # Save Trained Model
    p = model.saver.save(sess, "{}.tfmodel".format(model.name))
    t = time.time() - t_start_training
    print("++ Training: END -- Exec. Time={0:.0f}m {1:.0f}s Model Path={2:s}".format(t / 60, t % 60, p))

def train_model(model, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
    print("++ Training with {} epochs and {} minibatch size.".format(epochs, minibatch_size))

    if optimizer is not None:
        model.optimizer = optimizer
    else:
        # model.optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate, epsilon=0.1)
        model.optimizer = tf.train.GradientDescentOptimizer(model.learning_rate)

    model.tf_summary = tf.merge_summary(model.summaries)
    model.tf_logs = tf.train.SummaryWriter("logs/{}".format(model.name), sess.graph, flush_secs=30)

    # Specify the optimizer and create the train op:
    model.train_step = slim.learning.create_train_op(model.total_loss, model.optimizer)

    # Initialize Model ...
    sess.run(tf.initialize_all_variables())
    model.init_fn(sess)

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

        for _X, _y in iterate_minibatches(X, y, minibatch_size):
            # Resize Training Batches (to avoid consuming much memeory)
            cnt += 1
            update_screen("\r++++ Mini Batch ({} out of {}): ".format(cnt, X.shape[0]/minibatch_size))
            sess.run([model.train_step], feed_dict={model.X: _X, model.y: _y, model.learning_rate: lr})
            if cnt % 25 == 0:
                val_loss, val_accuracy, summary = calculate_loss(model, sess, val_X, val_y, 1000)
                t = time.time() - t_start_minibatch
                update_screen(" Loss={0:.4f} Accuracy={1:.4f} Exec. Time={2:.0f}m {3:.0f}s\n".format(val_loss, val_accuracy, t / 60, t % 60))
                t_start_minibatch = time.time()

                model.tf_logs.add_summary(summary, i * (X.shape[0] / minibatch_size) + cnt)

                # Handle Close Signals
                if os.path.isfile("{}.halt".format(model.name)):
                    graceful_halt(model, t_start_training)
                    os.remove("{}.halt".format(model.name))
                    exit(0)

        update_screen("\n")
        val_loss, val_accuracy, summary = calculate_loss(model, sess, val_X, val_y, val_X.shape[0])
        # model.tf_logs.add_summary(summary, epoch * (X.shape[0] / minibatch_size) + cnt)
        t = time.time() - t_start_epoch
        print("+++ Epoch {0:.0f}: END -- Loss={1:.4f} Accuracy={2:.4f} Exec. Time={3:.0f}m {4:.0f}s".format(i, val_loss, val_accuracy, t / 60, t % 60))

        # Always Save Best Accuracy
        if val_accuracy > model.best_accuracy:
            model.best_accuracy = val_accuracy
            model.saver.save(sess, "{}.tfmodel".format(model.name))
            print("+++ Epoch {0:.0f}: END -- SAVED BEST ACC".format(i))

def load_model(model, sess):
    sess.run(tf.initialize_all_variables())
    model.saver.restore(sess, "{}.tfmodel".format(model.name))

def predict_proba(model, sess, X, step=10):
    fc3ls = []

    size = X.shape[0]
    sample_idx = random.sample(range(0, size), size)
    reverse_idx = list(map(sample_idx.index, range(0,size)))

    for i in range(size / step):
        update_screen("\rpredict_proba: {} / {}".format(i*step, size))
        fc3l = sess.run(model.logits, feed_dict={model.X: X[sample_idx[i*step:(i+1)*step]]})
        fc3ls.append(fc3l)

    probs = sess.run(model.probs, feed_dict={model.logits: np.vstack(fc3ls)})

    return probs[reverse_idx]

def select_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# def load_weights(self, sess, weight_file):
#     # This includes optimizer variables as well.
#     init = tf.initialize_all_variables()
#     sess.run(init)
#
#     weights = np.load(weight_file)
#     keys = sorted(weights.keys())
#     for i, k in enumerate(keys):
#         if i == 30:
#             break
#         print i, k, np.shape(weights[k])
#         sess.run(self.parameters[i].assign(weights[k]))
#
#     # Transfer Learning
#     # Initialize the last fully connected layer
#     for par in self.parameters[-2:]:
#         print("Param: {}".format(par.name))

def define_parser(klass='Model'):
    parser = argparse.ArgumentParser(description='{} Network Specifications.'.format(klass))
    parser.add_argument('--gpu', default=1, help='GPU to use for train')
    parser.add_argument('--name', default=klass, help='Name of the model to use for train')
    parser.add_argument('--epochs', default='30', help='Number of epochs')
    parser.add_argument('--resume', default="no", help='Resume from previous checkpoint')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='Generate test predictions')
    parser.add_argument('--on-training', dest='on_training', default=False, action='store_true', help='Test on training')
    parser.add_argument('--bag', default="no", help='Bagging split')
    parser.add_argument('--fulldata', dest='fulldata', default=False, action='store_true', help='Train on full data?')
    return parser

def test(sess, model, args):
    print("+++++ TESTING +++++")
    model.load_model(sess)
    ids, Xt = CatVsDogs.prepare_test_data()
    ids = np.array(ids).astype(np.int)

    prob = model.predict_proba(sess, Xt, step=50)

    np.save("CatVsDogs.Xt.{}.npy".format(args.name), prob)

def test_on_training(sess, model, args):
    print("+++++ TESTING ON TRAINING +++++")
    model.load_model(sess)
    Xt = CatVsDogs.prepare_training_data_for_testing()

    prob = model.predict_proba(sess, Xt, step=50)

    np.save("CatVsDogs.X.{}.npy".format(args.name), prob)

def train(sess, model, args):
    print("+++++ TRAINING +++++")
    X, y, Xt, yt = CatVsDogs.prepare_data(fulldata=args.fulldata)
    model.train(sess, X, y, Xt, yt, epochs=int(args.epochs), minibatch_size=10)

def main(ModelClass, args):
    select_gpu(args.gpu)
    model = ModelClass(args.name)
    sess = tf.Session()
    if not args.test:
        train(sess, model, args)
    else:
        if args.on_training:
            test_on_training(sess, model, args)
        else:
            test(sess, model, args)
