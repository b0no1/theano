import theano
from theano import tensor as T
import numpy as np

rng = np.random

trX1 = np.linspace(-1, 1, 101)[np.newaxis]
trX = np.ones((101,1))
for i in range(5):
    trX = np.append(trX,trX1.T,axis=1)
trY = (1 * trX[:,0]) + (0.5 * trX[:,1]) + (1 * trX[:,2]) + (1.5 * trX[:,3]) + (2 * trX[:,4]) + (2.5 * trX[:,5]) 
print trY

#print (np.asarray([1, 0.7680387,  1.55897067, 2.05684484,  1.03026025,  2.08588553]) * trX).sum(axis=1)
#print trX
#print trY
'''

X = T.vector('x')
Y = T.scalar('y')

# we will be using a weight and a bias for fitting a model here
#   w[0] : slope
#   w[1] : intercept

w = theano.shared(np.random.randn(6),name='w')

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
'''
