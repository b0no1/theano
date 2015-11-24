import theano
from theano import tensor as T
import numpy as np

rng = np.random

trX = np.linspace(-1, 1, 101)[np.newaxis]
trX = np.append(trX.T,np.ones((101,1)),axis=1)
trY = (2.035 * trX[:,0]) + (1.345 * trX[:,1])

#print trY

X = T.vector('x')
Y = T.scalar('y')

'''
we will be using a weight and a bias for fitting a model here
w[0] : slope
w[1] : intercept
'''

w = theano.shared(np.random.randn(2),name='w')

y = (X*w).sum()

cost = T.mean(T.sqr(y - Y))

gradient = T.grad(cost=cost, wrt=w)

updates = [[w, w - gradient * 0.01]]
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

for i in range(100):
    for x, y in zip(trX, trY):
        cost_value = train(x, y)
        print 'Cost : %f' %cost_value
        print w.get_value()

