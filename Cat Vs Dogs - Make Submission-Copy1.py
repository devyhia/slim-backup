
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


# In[7]:

df_submission = pd.read_csv('submission_inception_resnet_ensemble_deep_fc_fulldata_inception_v3_256.csv')


# In[24]:

df_uncertain = df_submission[(df_submission.label <0.55) & (df_submission.label > 0.45)].reset_index(drop=True)


# In[13]:

import CatVsDogs


# In[14]:

ids, Xt = CatVsDogs.prepare_test_data()


# In[15]:

import matplotlib.pyplot as plt


# In[16]:

get_ipython().magic(u'matplotlib inline')


# In[44]:

flatten = lambda l: [item for sublist in l for item in sublist]


# In[56]:

fig, axes = plt.subplots(np.ceil(df_uncertain.shape[0] / 4).astype(np.int), 4, figsize=(15,15))
axes = flatten(axes)

for idx, ax in enumerate(axes):
    if idx >= df_uncertain.shape[0]:
        ax.axis('off')
        continue

    image_idx = df_uncertain.loc[idx, 'id'] - 1
    ax.imshow(Xt[image_idx])
    ax.set_title("LABEL: {}".format(df_uncertain.loc[idx, 'label']))

# plt.tight_layout()


# In[ ]:



