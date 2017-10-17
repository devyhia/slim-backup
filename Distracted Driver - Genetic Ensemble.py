
# coding: utf-8

# In[84]:

import argparse

parser = argparse.ArgumentParser(description='Simple Network.')
args, unknown_args = parser.parse_known_args()


# In[85]:

import os, random, sys
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
slim = tf.contrib.slim


# In[86]:

import sklearn.metrics as metrics
import Shared


# In[87]:

y = np.load('cache/y.original.npy')
yt = np.load('cache/yt.original.npy')


# In[188]:

models = [
    'alexnet.distracted_driver.original.1',
    'alexnet.distracted_driver.segmented.1',
#     'alexnet.distracted_driver.face.1',
#     'alexnet.distracted_driver.hands.1',
#     'alexnet.distracted_driver.hands_and_face.1',
#     'inceptionV3.distracted_driver.original.1',
#     'inceptionV3.distracted_driver.segmented.1',
#     'inceptionV3.distracted_driver.face.1',
#     'inceptionV3.distracted_driver.hands.1',
#     'inceptionV3.distracted_driver.hands_and_face.1',
]


# In[189]:

X = [ np.load("Prediction.X.{}.npy".format(model)) for model in models ]
Xt = [ np.load("Prediction.Xt.{}.npy".format(model)) for model in models ]


# In[190]:

for model in Xt:
    print("{:.4f}\t{:.2f}".format(metrics.log_loss(yt, model), 100 * metrics.accuracy_score(yt.argmax(axis=1), model.argmax(axis=1))))


# In[191]:

ens = lambda x: reduce(lambda prev, curr: prev+curr, x) / len(x)


# In[192]:

print(metrics.log_loss(y, ens(X)))
print(metrics.log_loss(yt, ens(Xt)), metrics.accuracy_score(yt.argmax(axis=1), ens(Xt).argmax(axis=1)))


# In[178]:

from random import randint, random, sample
from operator import add

def individual(length):
    'Create a member of the population.'
    return [ random() for x in xrange(length) ]

def population(count, length):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual

    """
    return [ individual(length) for x in xrange(count) ]

def fitness(individual, portion=0.5):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    size = X[0].shape[0]
    idx = sample(range(0, size), int(portion*size))
    total = reduce(add, individual, 0)
    ens = reduce(lambda prev, curr: prev + curr[0] * curr[1][idx], zip(individual, X), np.zeros(X[0][idx].shape)) / total
    return metrics.log_loss(y[idx], ens)

def grade(pop):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x) for x in pop))
    return summed / (len(pop) * 1.0)

def evolve(pop, retain=0.2, random_select=0.1, mutate=0.05):
    graded = [ (fitness(x), x) for x in pop]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = random()
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)
    parents.extend(children)
    return parents


# In[182]:

# Example usage
p_count = 50
p = population(p_count, len(X))
fitness_history = [grade(p),]
for i in xrange(10):
    p = evolve(p)
    fitness_history.append(grade(p))
    Shared.update_screen("\rStep {}: Min Loss={}".format(i, np.min(fitness_history)))


# In[183]:

graded = [ (fitness(x, portion=1.0), x) for x in p]
graded = [ x for x in sorted(graded)]


# In[184]:

graded[-1]


# In[185]:

weights = graded[-1][1]
total = reduce(add, weights, 0)
probs_ens = reduce(lambda prev, curr: prev + curr[0] * curr[1], zip(weights, Xt), np.zeros(Xt[0].shape)) / total


# In[186]:

print(metrics.log_loss(yt, probs_ens))


# In[187]:

print(metrics.accuracy_score(yt.argmax(axis=1), probs_ens.argmax(axis=1)))


# In[119]:

graded[-1]


# In[158]:

weights = graded[-1][1]
weights = [0.18309921362793558,
  0.4326389772065735,
  0.10991134113843704,
  0.01194613485192364,
  0.00971146230693376,
  0.9320650338477107,
  0.8376795455837012,
  0.08035432265243636,
  0.39423251988088026,
  0.7441139851401929]
total = reduce(add, weights, 0)
probs_ens = reduce(lambda prev, curr: prev + curr[0] * curr[1], zip(weights, Xt), np.zeros(Xt[0].shape)) / total


# In[159]:

print(metrics.log_loss(yt, probs_ens))


# In[160]:

print(metrics.accuracy_score(yt.argmax(axis=1), probs_ens.argmax(axis=1)))


# #### Make Submission

# In[ ]:

import pandas as pd


# In[ ]:

weights = graded[-1][1]
total = reduce(add, weights, 0)
probs_ens = reduce(lambda prev, curr: prev + curr[0] * curr[1], zip(weights, Xt), np.zeros(Xt[0].shape)) / total


# In[38]:

submission = np.hstack((np.arange(1, probs_ens.shape[0]+1, 1).reshape(-1, 1), probs_ens[:, 0].reshape(-1,1)))
df_submission = pd.DataFrame(submission, columns=["id", "label"])
df_submission["id"] = df_submission["id"].astype(np.int)
df_submission.to_csv("submission_{}.csv".format("deep_ensemble_ga_weights_least_fit"), index=False)


# ### Statistics

# In[149]:

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=False)
plt.rc('font', family='serif')


# In[150]:

import seaborn as sn


# In[151]:

import itertools


# In[162]:

metrics.confusion_matrix(y_pred=Xt[0].argmax(axis=1), y_true=yt.argmax(axis=1))


# In[163]:

metrics.confusion_matrix(y_pred=Xt[1].argmax(axis=1), y_true=yt.argmax(axis=1))


# In[165]:

metrics.confusion_matrix(y_pred=Xt[2].argmax(axis=1), y_true=yt.argmax(axis=1))


# In[166]:

metrics.confusion_matrix(y_pred=Xt[3].argmax(axis=1), y_true=yt.argmax(axis=1))


# In[167]:

metrics.confusion_matrix(y_pred=Xt[4].argmax(axis=1), y_true=yt.argmax(axis=1))


# In[161]:

metrics.confusion_matrix(y_pred=probs_ens.argmax(axis=1), y_true=yt.argmax(axis=1))


# In[154]:

CLASSES = [
    "Drive Safe",
    "Text Left",
    "Talk Left",
    "Text Right",
    "Talk Right",
    "Adjust Radio",
    "Drink",
    "Hair & Makeup",
    "Reach Behind",
    "Talk Passenger"
]


# In[155]:

plot_confusion_matrix(cm, CLASSES)


# In[ ]:



