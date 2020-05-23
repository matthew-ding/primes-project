from sympy import *
import numpy as np
from scipy.spatial.distance import cdist, euclidean

### HYPERPARAMETERS
n = 30  # number of dimensions
m = 8  # number of nodes
byzantine_set = [0, 1, 7]  # set of byzantine nodes
# adjList = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2, 4, 5],
#           4: [3, 6], 5: [3, 6], 6: [4, 5, 7, 8, 9], 7: [6], 8: [6], 9: [6]}

adjList = {0: [4, 5], 1: [4, 5], 2: [4, 5], 3: [4, 5],
           4: [0, 1, 2, 3, 6], 5: [0, 1, 2, 3, 6], 6: [4, 5, 7], 7: [6]}

dataset = []
for i in range(m):
    list = []
    for j in range(n):
        list.append(abs(float(np.random.normal(j, 0))))
    dataset.append(list)

# dictionary of variables
symbolDict = {}
for i in range(n):
    symbolDict["x" + str(i)] = Symbol("x" + str(i))

functionDict = []
for i in range(m):
    f = 0
    for j in range(n):
        expression = dataset[i][j] * (symbolDict["x" + str(j)] - dataset[i][j]) ** 2.0
        f += expression
    functionDict.append(f)


# output cost function for machine j
def function(j):
    output = functionDict[j]
    for i in range(n):
        output = output.subs(symbolDict["x" + str(i)], thetaDict[j][i])

    return output


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


# dictionary of partial derivatives (n x m dimensions)
gradientDict = []
for j in range(m):
    row = []
    for i in range(n):
        row.append(functionDict[j].diff(symbolDict["x" + str(i)]))
    gradientDict.append(row)

# Data
# dictionary of parameters
thetaDict = []

for j in range(m):
    row = []
    for i in range(n):
        row.append(i * 100)
    thetaDict.append(row)

thetaDict = np.array(thetaDict, dtype=float)

alpha = .005
iterations = 0
printData = True

while iterations < 150:
    calculatedGradients = {}

    # iterating through machines
    for z in range(m):
        calculatedGradients[z] = []
        # iterating through dimensions
        for i in range(n):
            gradient = gradientDict[z][i]

            for j in range(n):
                gradient = gradient.subs(symbolDict["x" + str(j)], thetaDict[z][j])

            gradient = float(gradient)

            if z in byzantine_set:
                gradient *= -1.0

            calculatedGradients[z].append(gradient)

    # Simultaneous update of gradients, based on neighbors
    for i in range(m):
        currentNeighborGrad = []

        for j in range(m):
            if j in adjList[i] or j == i:
                currentNeighborGrad.append(calculatedGradients[j])

        currentNeighborGrad = np.array(currentNeighborGrad, dtype=float)

        ### TODO: Fix geometric median update
        thetaDict[i] = thetaDict[i] - alpha * geometric_median(
            currentNeighborGrad)  # np.array(calculatedGradients[i], dtype=float)

    # if (i==5):
    #     print(geometric_median(currentNeighborGrad))
    #     print(np.array(calculatedGradients[i], dtype=float))

    iterations += 1
    # Cost function output
    if iterations % 10 == 0:
        print("Iteration " + str(iterations) + ", Cost: " + str(function(4)))

if printData:
    for i in range(n):
        print("theta (x" + str(i) + ") =", thetaDict[0][i], sep=" ")
