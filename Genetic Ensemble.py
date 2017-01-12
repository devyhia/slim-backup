
# coding: utf-8

# In[1]:

import argparse

parser = argparse.ArgumentParser(description='Simple Network.')
args, unknown_args = parser.parse_known_args()


# In[7]:

import os, random, sys
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
slim = tf.contrib.slim


# In[8]:

import sklearn.metrics as metrics
import Shared


# In[9]:

y = np.array([[1,0]]*12500 + [[0,1]]*12500)


# In[4]:

models = [
    "cvd_inception_v3_depth_150_50",
    "cvd_inception_v3_depth_200",
    "cvd_inception_resnet_v2_depth_100_16",
    "cvd_inception_resnet_v2_depth_100_16",
    "cvd_inception_v4_depth_150",
    "cvd_inception_v3_depth_256",
    "cvd_inception_v3_depth_256_64"
]


# In[25]:

X = [ np.load("CatVsDogs.X.{}.npy".format(model)) for model in models ]
Xt = [ np.load("CatVsDogs.Xt.{}.npy".format(model)) for model in models ]


# In[14]:

for model in X:
    print(metrics.log_loss(y, model))


# In[15]:

ens = lambda x: reduce(lambda prev, curr: prev+curr, x) / len(x)


# In[16]:

print(metrics.log_loss(y, ens(X)))


# In[18]:

from random import randint, random
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

def fitness(individual):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    total = reduce(add, individual, 0)
    ens = reduce(lambda prev, curr: prev + curr[0] * curr[1], zip(individual, X), np.zeros(X[0].shape)) / total
    return metrics.log_loss(y, ens)

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


# In[19]:

# Example usage
p_count = 50
p = population(p_count, len(X))
# p += [[0.01, 0.99] + [0.01]*14]
fitness_history = [grade(p),]
for i in xrange(1000):
    p = evolve(p)
    fitness_history.append(grade(p))
    Shared.update_screen("\rStep {}: Min Loss={}".format(i, np.min(fitness_history)))


# In[27]:

graded = [ (fitness(x), x) for x in p]
graded = [ x for x in sorted(graded)]


# In[30]:

graded[-1]


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


# In[ ]:



