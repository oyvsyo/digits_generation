
# coding: utf-8

# In[7]:


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import save_model


# In[2]:


model = keras.models.load_model('mnist_gan')


# In[3]:


get_image = K.function([model.layers[5].input, K.learning_phase()],
                           [model.layers[-1].output])


# In[4]:


images = [0]*10
vectors = [0]*10
for i in range(10):
    images[i] = [0]*10
    vectors[i] = [0]*10
for f in range(10):
    for j in range(10):
        vectors[f][j] = np.array([f/10, j/10])
        images[f][j] = get_image([vectors[f][j].reshape(-1,2)])[0].reshape(28, 28)


# In[16]:


nrow = 10
ncol = 10

fig = plt.figure(figsize=(ncol+1, nrow+1)) 

gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

for i in range(nrow):
    for j in range(ncol):
        im = images[i][j]
        ax= plt.subplot(gs[i,j])
        ax.imshow(im, cmap='cool')
        plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])

plt.show()

