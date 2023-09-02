

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

# Nirbaan
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Sample data: Replace with your actual data
catalyst_data = np.random.rand(300, 3)  # Catalyst properties (x, y, z)
original_ee = np.random.rand(300)# Original enantiomeric excess values

# Function to optimize catalyst properties using coordinate descent
def optimize_catalysts(catalysts, ee_values, iterations=100, step_size=0.01):
    optimized_catalysts = np.copy(catalysts)
    for _ in range(iterations):
        for i in range(len(optimized_catalysts)):
            original_ee = predict_ee(optimized_catalysts[i])  # Use your regression model
            for x in range(len(optimized_catalysts[i])):
                old_value = optimized_catalysts[i, x]
                optimized_catalysts[i, x] = old_value + step_size
                new_ee = predict_ee(optimized_catalysts[i])
                if new_ee < original_ee:
                    optimized_catalysts[i, x] = old_value - step_size
                    new_ee = predict_ee(optimized_catalysts[i])
                if new_ee < original_ee:
                    optimized_catalysts[i, x] = old_value
    return optimized_catalysts

# Function to predict enantiomeric excess using regression model
def predict_ee(properties):
    # Use your regression model to predict EE based on properties
    # Replace this with your actual prediction code
    return np.random.rand()

# Perform optimization on the first 150 catalysts
optimized_catalysts = optimize_catalysts(catalyst_data[:150], original_ee[:150])

# Perform k-nearest neighbors analysis
num_neighbors = 5  # Number of neighbors to consider
knn = NearestNeighbors(n_neighbors=num_neighbors)
knn.fit(catalyst_data[150:])  # Using the remaining 150 catalysts for validation

# Validate optimized catalysts using KNN
for i in range(len(optimized_catalysts)):
    distances, indices = knn.kneighbors([optimized_catalysts[i]])
    neighbors_ee = [original_ee[150 + idx] for idx in indices[0]]
    optimized_ee = predict_ee(optimized_catalysts[i])
    average_neighbors_ee = np.mean(neighbors_ee)
    print(f"Optimized EE: {optimized_ee:.4f} vs. Average Neighbors' EE: {average_neighbors_ee:.4f}")

# Further analysis and interpretation can be added here