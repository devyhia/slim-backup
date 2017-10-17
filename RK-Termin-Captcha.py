
# coding: utf-8

# In[1]:

from PIL import Image


# In[2]:

img = Image.open('rk-termin-captcha.jpeg')


# In[14]:

img.crop((75, 0, 225, 50))


# In[ ]:



