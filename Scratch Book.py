
# coding: utf-8

# In[6]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

Xt = np.load("features/model2.PreLogitsFlatten.Xt.npy")


# In[12]:

plt.imshow(Xt[5].reshape(48, 32))


# In[13]:

Xt2 = np.load("features/model1.Mixed_6h.X.0.npy")


# In[26]:

f, axarr = plt.subplots(7,7)
for i in range(7*7):
    axarr[i / 7, i % 7].imshow(Xt2[0, :, :, i])

# plt.tight_layout()


# In[ ]:



