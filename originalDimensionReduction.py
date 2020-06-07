import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn import random_projection

d = 1000  # number of dimensions

my_data = pd.read_csv('dataset.txt')

# setting the matrixes

# Setting up X-Values
X = my_data.iloc[:, 0:d]
ones = np.ones([X.shape[0], 1]) # adding constant column of ones
X = np.concatenate((ones, X), axis=1)

# Setting up costs (y values)
y = my_data.iloc[:, d:d + 1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray

# projection of input vectors (x values)
transformer = random_projection.GaussianRandomProjection(eps=0.5)

# generating random gaussian matrix
transformer.fit(np.array(X))
projection_matrix = transformer.components_

X = transformer.transform(X)

# new number of dimension
d_new = len(X[0])

# parameter array
theta = np.zeros([1, d_new])


# compute cost
def computeCost(X, y, theta):
    tobesummed = np.power(((X @ theta.T) - y), 2)
    return np.sum(tobesummed) / (2 * len(X))


def gradientDescent(X, y, theta, iters):
    cost = [0]
    while True:
        iters += 1
        alpha = 0.1*1/iters

        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost.append(computeCost(X, y, theta))

        if iters % 100 == 0:
            print("Cost at iteration " + str(iters) + ": " + str(cost[iters]))

        if iters != 1:
            if abs(cost[iters] - cost[iters - 1]) < precision:
                break

    return theta, cost


# set hyper parameters
iters = 0
precision = 0.000000001

g, cost = gradientDescent(X, y, theta, iters)

finalCost = computeCost(X, y, g)
print("Converges, Final Cost: " + str(finalCost) + "\n")

# Projection Recovery
one_matrix = np.ones([d+1, 1])

g = linprog(c=one_matrix, A_eq=projection_matrix, b_eq=g[0])


for i in range(d+1):
    if i==0:
        print("Constant: " + str(round(g.x[i],5)))
    else:
        print("Variable x" + str(i) + " Coefficient: " + str(round(g.x[i],5)))