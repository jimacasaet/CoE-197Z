# -*- coding: utf-8 -*-
"""
John Rufino Macasaet
2013-18722
CoE 197 Z
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
        alpha = alpha - theta * gradient
    return theta
    
def genData(count, x):
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    y = None
    if count == 4:
        c = float(sys.argv[3]) 
        d = float(sys.argv[4])
        y = np.polyval([a,b,c,d],x)
    elif count == 3:
        c = float(sys.argv[3])
        y = np.polyval([a,b,c],x)
    elif count == 2:
        y = np.polyval([a,b],x)
    return y

# python3 linreg.py a b c d
# where: ax^3+bx^2+cx+d=0
arg_count = len(sys.argv)-1

# generate data points
points = np.reshape(np.linspace(-10,10,num=21),(21,1))
one = np.ones((21,1));
print(points.shape)
x = np.hstack((one,points))
p = genData(arg_count, points)
noise = np.random.normal(0,1,len(p)) #generate noise to be added
p = p + noise #add the noise
'''
m, n = np.shape(x)
numIterations= 100000
alpha = 0.0001
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
'''
plt.plot(points,p,'bo')
plt.show()
print(x)

'''
References:
http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/

'''