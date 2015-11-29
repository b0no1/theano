
# coding: utf-8

# In[1]:

### >> Multilayer Perceptron << ###
## http://deeplearning.net/tutorial/mlp.html ##

import os
import sys

import theano
import theano.tensor as T
import numpy as np

from logistic import load_data,LogisticRegression


# In[4]:

# Define the hidden layer class
class HiddenLayer(object):
    def __init__(self,rng,input,n_in,n_out,w=None,b=None):
        self.input = input
        if w is None:
            wval = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)),
                              dtype = theano.config.floatX)
            w = theano.shared(wval,name='w',borrow=True)
        #bias
        if b is None:
            b = theano.shared(
                value = np.zeros( (n_out,),dtype=theano.config.floatX),
                name='b',
                borrow=True
            )
        
        #b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        #b = theano.shared(value=b_values, name='b', borrow=True)
        
        self.w = w
        self.b = b
        self.output = T.tanh(T.dot(input,self.w)+self.b)
        self.params = [self.w,self.b]    
        


# In[5]:

datasets = load_data('mnist.pkl.gz')

batch_size = 20

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


# In[6]:

### : Testing the hidden layer ###
#------------------------------##

x = T.matrix('x')

rng = np.random.RandomState(1234)
hl = HiddenLayer(rng, input=x, n_in=28*28, n_out=500)


# In[7]:

### Define MLP class ###
class MLP(object):
    def __init__(self,rng,input,n_in,n_h,n_out):
        self.hidden_layer = HiddenLayer(rng,input=input,n_in=n_in,n_out=n_h)
        self.output_layer = LogisticRegression(input=self.hidden_layer.output, n_in=n_h,n_out=n_out)
        #regularization
        self.L1 = abs(self.hidden_layer.w).sum() + abs(self.output_layer.w).sum()
        self.L2 = (self.hidden_layer.w**2).sum() + (self.output_layer.w**2).sum()
        # Negative Log Likelihood
        self.neg_log_likelihood = (self.output_layer.neg_log_likelihood)
        # errors function
        self.errors = (self.output_layer.errors)
        # params
        self.params = self.hidden_layer.params + self.output_layer.params
        
        self.input = input


# In[8]:

index = T.lscalar('index')
x = T.matrix('x')
y = T.ivector('y')
rng = np.random.RandomState(1234)

# instantiate MLP classifier
cl = MLP(rng, input = x, n_in = 28*28, n_h = 500, n_out = 10)


# In[9]:

# setup cost
cost = cl.neg_log_likelihood(y) + (cl.L1 * 0.00) + (cl.L2 * 0.0001)

# setup gradient
gparams = [ T.grad(cost,param) for param in cl.params ]

# setup updates 
updates = [ (param, param - 0.01*gparam) for param,gparam in zip(cl.params,gparams)]


# In[10]:

# compile training function
train = theano.function(inputs=[index],
                       outputs=cost,
                       updates=updates,
                       givens = { x : train_set_x[index * batch_size : (index+1)*batch_size],
                                  y : train_set_y[index * batch_size : (index+1)*batch_size]
                                }
                       )


# In[12]:

# Actual training begins here
minibatch_avg_cost = 0
for j in xrange(100):
    for i in xrange(n_train_batches):
        minibatch_avg_cost = train(i)
    if j % 10 == 0:
        print 'iteration ',j,' : cost : ', minibatch_avg_cost
    


# In[14]:

# compile the test function
test = theano.function(inputs=[index],
                      outputs=cl.errors(y),
                      givens = { x : test_set_x[index*batch_size : (index+1)*batch_size],
                                 y : test_set_y[index*batch_size : (index+1)*batch_size]
                               }
                      )


# In[18]:

# testing
error_sum = 0.0
for i in xrange(n_test_batches):
    error_sum += test(i)
print 'avg_error : ',error_sum/n_test_batches

