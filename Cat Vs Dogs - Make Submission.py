
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd


# In[1]:

import argparse

parser = argparse.ArgumentParser(description='Make Ensemble Submission')
parser.add_argument('--name', help='Name of Kaggle Submission')
parser.add_argument('--models', nargs='+', help='List of models')
args, unknown_args = parser.parse_known_args()


# In[ ]:

probs = []
probs_ens = None

for model in args.models:
    prob = np.load("CatVsDogs.Xt.{}.npy".format(model))
    probs.append(prob)

probs_ens = probs[0]
for prob in probs[1:]:
    probs_ens += prob

probs_ens /= len(probs)

submission = np.hstack((np.arange(1, probs_ens.shape[0]+1, 1).reshape(-1, 1), probs_ens[:, 0].reshape(-1,1)))
df_submission = pd.DataFrame(submission, columns=["id", "label"])
df_submission["id"] = df_submission["id"].astype(np.int)
df_submission.to_csv("submission_{}.csv".format(args.name), index=False)

