import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial.distance import cdist, euclidean
from sklearn import random_projection

# HYPERPARAMETERS
d = 1000  # number of dimensions
m = 10  # number of nodes
n = 50  # total number of data points

byzantine_set = []
X_set = []  # list of all x matrices
Y_set = []  # list of all y vectors
gradient_set = []

my_data = pd.read_csv('dataset.txt')

# SETTING UP INPUT MATRICES

# Setting up X-Values
X = my_data.iloc[:, 0:d]
ones = np.ones([X.shape[0], 1])  # adding constant column of ones
X = np.concatenate((ones, X), axis=1)

# Setting up costs (y values)
y = my_data.iloc[:, d:d + 1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray

# projection of input vectors (x values)
transformer = random_projection.GaussianRandomProjection(eps=0.2)

# generating random gaussian matrix
transformer.fit(np.array(X))
projection_matrix = transformer.components_

X = transformer.transform(X)

# new number of dimension
d_new = len(X[0])

for i in range(m):
    X_set.append(X[i * int(n / m):(i + 1) * int(n / m), :])
    Y_set.append(y[i * int(n / m):(i + 1) * int(n / m), :])

    gradient_set.append(np.zeros([1, d_new]))

# parameter array
theta = np.zeros([1, d_new])

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def trimmed_mean(X, neighbor_count):
    trim_count = neighbor_count // 2
    total = neighbor_count + 1
    sorted_X = np.sort(X, axis=0)

    # trimming
    sorted_X = sorted_X[trim_count + 1: total - trim_count + 1, :]
    size = len(sorted_X)
    # trimmed mean
    return np.sum(sorted_X, axis=0) / size


# compute cost
def computeCost(X, y, theta):
    tobesummed = np.power(((X @ theta.T) - y), 2)
    return np.sum(tobesummed) / (2 * len(X))


def gradientDescent(iters, theta, gradient_set):
    cost = [0]

    while True:
        iters += 1
        alpha = 0.01 * 1 / iters

        # calculating gradients
        for i in range(m):
            gradient_set[i] = (1 / len(X_set[i])) * np.sum(X_set[i] * (X_set[i] @ theta.T - Y_set[i]),
                                                           axis=0)

        gradient_set = np.array(gradient_set)
        theta = theta - alpha * geometric_median(gradient_set)
        cost.append(computeCost(X_set[i], Y_set[i], theta))

        if iters % 1 == 0:
            print("Cost at iteration " + str(iters) + ": " + str(cost[iters]))

        if iters != 1:
            if abs(cost[iters] - cost[iters - 1]) < precision:
                break

    return theta, cost


# set hyper parameters
iters = 0
precision = 0.000000001

g, cost = gradientDescent(iters, theta, gradient_set)

finalCost = computeCost(X, y, g)
print("Converges, Final Cost: " + str(finalCost) + "\n")

# Projection Recovery
one_matrix = np.ones([d + 1, 1])

g = linprog(c=one_matrix, A_eq=projection_matrix, b_eq=g[0])

for i in range(d + 1):
    if i == 0:
        print("Constant: " + str(round(g.x[i], 5)))
    else:
        print("Variable x" + str(i) + " Coefficient: " + str(round(g.x[i], 5)))

# for i in range(1, len(g)):
#    print("Variable: x" + str(i) + ", Coefficient: " + str(g[i]))
#
# print("Constant: " + str(g[0]))
