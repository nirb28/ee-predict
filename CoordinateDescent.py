

#https://stackoverflow.com/questions/51977418/coordinate-descent-in-python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(10, 100, 1000)
y = np.linspace(10, 100, 1000)

def func(x, y, param):
    return param[0] * x + param[1] * y

def costf(X, y, param):
    return np.sum((X.dot(param) - y) ** 2)/(2 * len(y))

z = func(x, y, [5, 8]) + np.random.normal(0., 10.)
z = z.reshape(-1, 1)

interc = np.ones(1000)
X = np.concatenate([interc.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

#param = np.random.randn(3).T
param = np.array([2, 2, 2])

def gradient_descent(X, y, param, eta=0.01, iter=100):
    cost_history = [0] * iter

    for iteration in range(iter):
        h = X.dot(param)
        loss = h - y
        gradient = X.T.dot(loss)/(2 * len(y))
        param = param - eta * gradient
        cost = costf(X, y, param)
        #print(cost)
        cost_history[iteration] = cost

    return param, cost_history


def coordinate_descent(X, y, param, iter=100):
    cost_history = [0] * iter

    for iteration in range(iter):
        for i in range(len(param)):
            dele = np.dot(np.delete(X, i, axis=1), np.delete(param, i, axis=0))
            param[i] = np.dot(X[:,i].T, (y - dele))/np.sum(np.square(X[:,i]))
            cost = costf(X, y, param)
            #print(cost)
            cost_history[iteration] = cost

    return param, cost_history


ret, xret = gradient_descent(X, z, param)
cret, cxret = coordinate_descent(X, z, param)

plt.plot(range(len(xret)), xret, label="GD")
plt.plot(range(len(cxret)), cxret, label="CD")
plt.legend()
plt.show()