import numpy as np
from scipy.spatial.distance import cdist, euclidean
import pandas as pd
import sys
import os
from Graph_Generation import graphGenerator
sys.path.append('../')


### HYPERPARAMETERS
d = 10  # number of dimensions
n = 1000  # total number of datapoints

# byzantine_set = [4, 5, 9]  # set of byzantine nodes
# adjList = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2, 4, 5],
#           4: [3, 6], 5: [3, 6], 6: [4, 5, 7, 8, 9], 7: [6], 8: [6], 9: [6]}

# adjList = {0: [4, 5], 1: [4, 5], 2: [4, 5], 3: [4, 5],
#           4: [0, 1, 2, 3, 6], 5: [0, 1, 2, 3, 6], 6: [4, 5, 7], 7: [6]}

adjList, byzantine_set, m = graphGenerator.get_adj_list()

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('..\\Dataset_Generation\\dataset.txt', cur_path)
my_data = pd.read_csv(new_path)

X_set = []  # list of all x matrices
Y_set = []  # list of all y vectors
theta_set = []  # list of all parameter vectors
gradient_set = []
cost_set = []

for i in range(m):
    # Setting up X-Values
    X = my_data.iloc[i * int(n / m):(i + 1) * int(n / m), 0:d]
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    X_set.append(X)

    # Setting up costs (y values), .values converts it from pandas.core.frame.DataFrame to numpy.ndarray
    y = my_data.iloc[i * int(n / m):(i + 1) * int(n / m), d:d + 1].values
    Y_set.append(y)

    # parameter arrays
    theta_set.append(np.zeros([1, d + 1]))
    gradient_set.append(np.zeros([1, d + 1]))

    cost_set.append([])


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


def gradientDescent(iters, alpha, target):
    while True:
        for i in range(m):
            calculateGradient(X_set[i], Y_set[i], theta_set[i], i)

        for i in range(m):
            currentNeighborGrad = []
            for j in range(m):
                if j == i or j in adjList[i]:
                    currentNeighborGrad.append(gradient_set[j])

            currentNeighborGrad = np.array(currentNeighborGrad)
            gradient = geometric_median(currentNeighborGrad)
            # gradient = trimmed_mean(currentNeighborGrad, len(adjList[i]))

            alpha = 0.01 / (iters + 1)
            theta_set[i] = theta_set[i] - alpha * gradient

            cost_set[i].append(computeCost(X_set[i], Y_set[i], theta_set[i]))

        if iters % 100 == 0:
            print("Cost at iteration " + str(iters + 1) + ": " + str(cost_set[target][iters]))

        if iters != 0:
            if abs(cost_set[target][iters] - cost_set[target][iters - 1]) < precision:
                break

            # if cost_set[target][iters] - cost_set[target][iters - 1] > 0:
            #   break

        iters += 1

    return theta_set[target], cost_set[target]


def calculateGradient(X, y, theta, i):
    gradient = (1.0 / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)

    if i in byzantine_set:
        gradient *= -1.0

    gradient_set[i] = gradient


# set hyper parameters
iters = 0
alpha = 0.001
precision = 0.001
target = 3

g, cost = gradientDescent(iters, alpha, target)

finalCost = computeCost(X_set[target], Y_set[target], g)
print("Converges, Final Cost: " + str(finalCost) + "\n")

g = g.tolist()[0]

for i in range(1, len(g)):
    print("Variable: x" + str(i) + ", Coefficient: " + str(g[i]))

print("Constant: " + str(g[0]))
