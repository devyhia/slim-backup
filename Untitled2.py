
# coding: utf-8

# In[ ]:

get_ipython().system(u'tail -100f /home/devyhia/A-Convolutional-Neural-Network-Cascade-for-Face-Detection/log/train_48net_1.log')


# In[28]:

get_ipython().system(u'export DIR=/home/devyhia/A-Convolutional-Neural-Network-Cascade-for-Face-Detection/')


# In[29]:

get_ipython().system(u'python "$DIR/train_48net.py" &> "$DIR/log/train_48net_1.log" &')


# In[9]:

neg_db_48 = 364454


# In[4]:

import param


# In[5]:

import sys


# In[6]:

sys.path.append('/home/devyhia/A-Convolutional-Neural-Network-Cascade-for-Face-Detection/')


# In[7]:

import param


# In[13]:

xrange(0,neg_db_48,param.mini_batch)[-1] + param.mini_batch


# In[14]:

neg_db_48


# In[ ]:



