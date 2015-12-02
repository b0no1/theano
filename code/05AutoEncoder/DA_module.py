
# coding: utf-8

# In[ ]:

## Autoencoder : http://deeplearning.net/tutorial/dA.html ##
import numpy as np
import theano
import theano.tensor as T

from logistic import load_data

from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images


# In[ ]:

class DenoisingAutoencoder(object):
    def __init__(self,input,rng,num_v,num_h,theano_rng=None,w=None,bh=None):
        self.num_h = num_h
        self.num_v = num_v
        # setup random stream
        if not theano_rng:
            self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        # init weight
        wval = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (num_h + num_v)),
                                      high=4 * np.sqrt(6. / (num_h + num_v)),
                                      size=(num_v,num_h)),dtype=theano.config.floatX)
        if w is None:
            self.w = theano.shared(value=wval,name='w',borrow = True)
        # init visible layer bias
        self.bv = theano.shared(value = np.zeros(num_v,dtype=theano.config.floatX),name='bv',borrow=True)
        # init hidden layer bias
        if bh is None:
            self.bh = theano.shared(value = np.zeros(num_h,dtype=theano.config.floatX),name='bh',borrow=True)
        # setup weight hidden-output layer connections
        #  -> tied weights
        self.w_ = self.w.T
        self.x = input
        self.params = [self.w,self.bv,self.bh]
        
    def encode(self,x_):
        return T.nnet.sigmoid(T.dot(x_,self.w) + self.bh) ### Notice the use of bh here ###
    
    def decode(self,code):
        return T.nnet.sigmoid(T.dot(code,self.w_) + self.bv) ### Notice the use of bv here ###
    
    def loss(self,y,z):
        return - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

    def cost(self,y,z):
        return T.mean(self.loss(y,z))
    
    def corrupt_x(self,corruption_level):
        return self.theano_rng.binomial(size=self.x.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * self.x
   
    def step(self,corruption_level=0.,learning_rate=0.1):
        x_ = self.corrupt_x(corruption_level=corruption_level)
        y  = self.encode(x_)
        z  = self.decode(y)
        # setup cost, gradients and updates
        cost = self.cost(y,z)

        # gradients
        gparams = T.grad(cost,self.params)
        
        # updates
        updates = [ (param, param - (learning_rate*gparam) )
              for param,gparam in zip(self.params,gparams)]
        
        return (cost,updates)
