import numpy as np
from scipy.spatial.distance import cdist, euclidean
import pandas as pd
import sys
import os
from Graph_Generation import graphGenerator
sys.path.append('../')


### HYPERPARAMETERS
# byzantine_set = [4, 5, 9]  # set of byzantine nodes
# adjList = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2, 4, 5],
#           4: [3, 6], 5: [3, 6], 6: [4, 5, 7, 8, 9], 7: [6], 8: [6], 9: [6]}

# adjList = {0: [4, 5], 1: [4, 5], 2: [4, 5], 3: [4, 5],
#           4: [0, 1, 2, 3, 6], 5: [0, 1, 2, 3, 6], 6: [4, 5, 7], 7: [6]}

d = 10  # number of dimensions
n = 1000  # number of datapoints
adjList, byzantine_set, m = graphGenerator.get_adj_list()
print("Graph Generated Successfully")


cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('..\\Dataset_Generation\\dataset.txt', cur_path)
my_data = pd.read_csv(new_path)

X_set = []  # list of all x matrices
Y_set = []  # list of all y vectors
theta_set = []  # list of all parameter vectors
temp_theta_set = []
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
    temp_theta_set.append(np.zeros([1, d + 1]))

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


def gradientDescent(iters, target):
    while True:
        for i in range(m):
            # gradient calculation
            gradient = (1.0 / len(X_set[i])) * np.sum(X_set[i] * (X_set[i] @ theta_set[i].T - Y_set[i]), axis=0)

            if i in byzantine_set:
                gradient *= -1.0

            # gradient update
            alpha = 0.3 / (iters + 1)
            temp_theta_set[i] = theta_set[i] - alpha * gradient

        # neighbor aggregation
        for i in range(m):
            currentNeighborParam = []
            for j in range(m):
                if j == i or j in adjList[i]:
                    currentNeighborParam.append(temp_theta_set[j][0])
                    # print(temp_theta_set[j][0])


            theta_set[i] = np.reshape(geometric_median(np.array(currentNeighborParam)), (1,11))
            # theta_set[i] = trimmed_mean(currentNeighborGrad, len(adjList[i]))

            cost_set[i].append(computeCost(X_set[i], Y_set[i], theta_set[i]))

        if (iters+1) % 100 == 0:
            print("Cost at iteration " + str(iters + 1) + ": " + str(cost_set[target][iters]))

        if iters != 0:
            if abs(cost_set[target][iters] - cost_set[target][iters - 1]) < precision:
                break

        iters += 1

    return theta_set[target], cost_set[target], iters


# set hyper parameters
iters = 0
precision = 0.00001
target = 3

g, cost, iters = gradientDescent(iters, target)

finalCost = computeCost(X_set[target], Y_set[target], g)
print("Converges in " + str(iters) + " iterations")
print("Final Cost: " + str(finalCost) + "\n")

g = g.tolist()[0]

for i in range(1, len(g)):
    print("Variable: x" + str(i) + ", Coefficient: " + str(g[i]))

print("Constant: " + str(g[0]))
