import numpy as np
import pandas as pd
from sklearn import random_projection

d = 1000  # number of dimensions

# my_data = pd.read_csv('home.txt', names=["size", "bedroom", "price"])
my_data = pd.read_csv('dataset.txt')  # , names=nm)

# setting the matrixes

# Setting up costs (y values)
y = my_data.iloc[:, d:d + 1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray

# projection of input vectors (x values)
transformer = random_projection.GaussianRandomProjection(eps=0.5)

# generating random gaussian matrix
transformer.fit(np.array(my_data))
projection_matrix = transformer.components_

my_data = transformer.transform(my_data)

# new number of dimension
d = len(my_data[0])

# parameter array
theta = np.zeros([1, d + 1])

my_data = pd.DataFrame(my_data)

# Setting up X-Values
X = my_data.iloc[:, 0:d]
ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)


# compute cost
def computeCost(X, y, theta):
    tobesummed = np.power(((X @ theta.T) - y), 2)
    return np.sum(tobesummed) / (2 * len(X))


def gradientDescent(X, y, theta, iters, alpha):
    cost = [0]
    while True:
        iters += 1

        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost.append(computeCost(X, y, theta))

        if iters % 1000 == 0:
            print("Cost at iteration " + str(iters) + ": " + str(cost[iters]))

        if iters != 1:
            if abs(cost[iters] - cost[iters - 1]) < precision:
                break

    return theta, cost


# set hyper parameters
alpha = 0.0000015
iters = 0
precision = 0.00001

g, cost = gradientDescent(X, y, theta, iters, alpha)

finalCost = computeCost(X, y, g)
print("Converges, Final Cost: " + str(finalCost) + "\n")

g = g.tolist()[0]

for i in range(1, len(g)):
    print("Variable: x" + str(i) + ", Coefficient: " + str(g[i]))

print("Constant: " + str(g[0]))
