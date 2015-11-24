import theano
from theano import tensor as T
import numpy as np

rng = np.random

trX = np.linspace(-1, 1, 101)
trY = (2.035 * trX) + 1.345

#print trX
#print trY

X = T.scalar()
Y = T.scalar()

'''
we will be using a weight and a bias for fitting a model here
m : slope
c : intercept
'''

m_value = rng.randn()
c_value = rng.randn()

m = theano.shared(m_value,name='m')
c = theano.shared(c_value,name='c')

y = (X*m) + c

cost = T.mean(T.sqr(y - Y))

gradient_m = T.grad(cost=cost, wrt=m)
gradient_c = T.grad(cost=cost, wrt=c)

updates = [[m, m - gradient_m * 0.01], (c, c - gradient_c * 0.01)]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

for i in range(100):
    for x, y in zip(trX, trY):
        cost_value = train(x, y)
        print 'Cost : %f' %cost_value
        print 'y = %f x + %f' %(m.get_value(),c.get_value())

        
