
# coding: utf-8

# In[5]:

import theano
import theano.tensor as T
import theano.tensor.nnet as conv
import numpy as np


# In[83]:

## A randomly initialized convolutional layer with 2 feature maps
# Input : (RGB) 3x639x516
# Receptive Field size : 9x9

# Random element
rng = np.random.RandomState(23455)

# Input 
input = T.tensor4('input')

# weight shape, w [depth_of_layer_m,depth_of_layer_m-1,filter_height,filter_width]
wshape = (2,3,9,9)

# bound on w values
wbound = np.sqrt(3 * 9 * 9)

# weight initialize with random numbers
wval = np.asarray(rng.uniform(low=-1/wbound,high=1/wbound,size=wshape),dtype=input.dtype)

# build shared variable w
w = theano.shared(wval,name='w')


# In[84]:

# setup bias
b = theano.shared(np.asarray(rng.uniform(-0.5,0.5,(2,)),dtype=input.dtype),name='b')


# In[85]:

# build symbolic expression for convolution layer
conv_out = conv.conv2d(input,w)


# In[86]:

# associate the bias term with the output of conv_layer
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))


# In[87]:

# compile function
convf = theano.function([input],output)


# In[89]:

# import an inmage and pass it to the random convnet
import pylab
from PIL import Image

img = Image.open(open('../../data/3wolfmoon.jpg'))

# convert to numpy array and normalize (%256)
img = np.asarray(img, dtype=theano.config.floatX)/256.

# reshape the image from 639x516x3 to 3,639,516 to 1,3,639,516
#  Format : [ batch_size, depth, height, width]
img_ = img.transpose(2,0,1).reshape(1,3,639,516)
print img.shape,' reshaped to ', img_.shape

# apply convol function
omg = convf(img_)

omg_gray1 = omg[0,0,:,:]
omg_gray2 = omg[0,1,:,:]

pylab.gray()
pylab.subplot(1, 2, 1); pylab.imshow(omg_gray1)
pylab.subplot(1, 2, 2); pylab.imshow(omg_gray2)

pylab.show()

