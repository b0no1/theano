
# coding: utf-8

# In[ ]:

## >> http://deeplearning.net/tutorial/lenet.html << ##
import numpy as np

import theano.tensor as T
import theano
import theano.tensor.nnet as conv

from theano.tensor.signal import downsample

from logistic import load_data,LogisticRegression
from mlp import HiddenLayer


# In[ ]:

class ConvPoolLayer(object):
    def __init__(self,rng,input,filter_shape,image_shape, pool_size=(2,2) ):
        # check if filter shape matches the input shape
        assert image_shape[1] == filter_shape[1]
        # >>>> self.input = input
        # number of inputs to each hidden unit: fan_in
        fan_in = np.prod(filter_shape[1:]) # 1x5x5 : receptive field 5x5 of grayscale image
        # fan_out to lower layer (left)
        #  for gradient propagation
        #   20 * 5x5 connections
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) # 20 * (5x5) : nkern[0] * (5x5)
        # random initialization of weights
        wbound = np.sqrt(6. / (fan_in + fan_out))
        wval = np.asarray(rng.uniform(low = -wbound, high = wbound, size=filter_shape),
                          dtype = theano.config.floatX)
        self.w = theano.shared(wval,name='w',borrow = True)
        # bias term 
        self.b = theano.shared(np.zeros((filter_shape[0],),dtype=theano.config.floatX),name='b', borrow=True)
        # convol operation
        conv_out = conv.conv2d(input,self.w,filter_shape=filter_shape,image_shape=image_shape)
        # pooling : downsampling
        pooled = downsample.max_pool_2d(input=conv_out,ds=pool_size,ignore_border=True)
        # apply non-linearity and bias to pooled output
        #  dimshuffle : convert shape of bias from (filter_shape[0],) to (1, n_filters, 1, 1)
        self.output = T.tanh(pooled + self.b.dimshuffle('x',0,'x','x'))
        # store params
        self.params = [self.w,self.b]
        self.input = input


# In[ ]:

# Load MNIST data
datasets = load_data('mnist.pkl.gz')

batch_size = 500

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


# In[ ]:

x = T.matrix('x')
y = T.ivector('y')

learning_rate = 0.01

# convert input x to form (batch_size,1,28,28)
layer0_input = x.reshape((batch_size,1,28,28))

# setup random stream
rng = np.random.RandomState(123455)

# build layer0
layer0 = ConvPoolLayer(rng=rng,input=layer0_input,
                      image_shape=(batch_size,1,28,28),
                      filter_shape=(20,1,5,5))
                      


# In[ ]:

## Layer 1 setup ##
layer1 = ConvPoolLayer(rng=rng,input=layer0.output,
                      image_shape=(batch_size,20,12,12),
                      filter_shape=(50,20,5,5))


# In[ ]:

## Layer 2 : Hidden Layer setup ##
# layer1 output shape : batch_sizex50x4x4
# layer2_h input shape req : batch_size x (50*4*4)
layer2_h_input = layer1.output.flatten(2)
# n_in = 50x4x4 pixels; n_out = 500 hidden nodes
layer2_h = HiddenLayer(rng=rng,input=layer2_h_input,n_in=50*4*4,n_out=500)


# In[ ]:

# Layer 3 : Output layer : LogisticRegression
layer3_o = LogisticRegression(input=layer2_h.output,n_in=500,n_out=10)


# In[ ]:

# cost 
cost = layer3_o.neg_log_likelihood(y)
# >> setup gradient expression <<
### Need :parameters
params = layer3_o.params + layer2_h.params + layer1.params + layer0.params
gparams = T.grad(cost,params)


# In[ ]:

## Updates ##
updates = [(param, param - gparam*learning_rate) 
              for param,gparam in zip(params,gparams)]


index = T.lscalar('index')
# compile train
train = theano.function(inputs=[index],
                        outputs=cost,
                        updates=updates,
                        givens = { x : train_set_x[index*batch_size : (index +1)*batch_size],
                                   y : train_set_y[index*batch_size : (index +1)*batch_size]}
                       )


# In[ ]:

# Actual training #
# Actual training begins here
minibatch_avg_cost = 0
for j in xrange(300):
    for i in xrange(n_train_batches):
        minibatch_avg_cost = train(i)        
    print 'iteration ',j,' : cost : ', minibatch_avg_cost


# In[ ]:

# testing
test = theano.function(inputs = [index],
                      outputs = layer3_o.errors(y),
                      givens = { x : test_set_x[index*batch_size : (index +1)*batch_size],
                                 y : test_set_y[index*batch_size : (index +1)*batch_size]
                               }
                      )
error_sum = 0.0
for i in xrange(n_test_batches):
    error_sum += test(i)
print 'avg_error : ',error_sum/n_test_batches


# In[ ]:

# visualize feature maps in convolnet
visual = theano.function(inputs=[index],
                        outputs = [layer3_o.errors(y),layer0.output],
                        givens = { x : valid_set_x[index*batch_size : (index+1)*batch_size],
                                   y : valid_set_y[index*batch_size : (index+1)*batch_size]
                                 }
                        )

import pylab


# In[ ]:

er,imcluster0 = visual(16)


# In[ ]:

pylab.gray()

pylab.imshow(imcluster0[3,0,:,:])
#pylab.savefig('im01.png')

for i in xrange(20):
    pylab.imshow(imcluster0[3,i,:,:])
    #pylab.show()
    pylab.savefig('im%d.png'%(i))
#pylab.show()

