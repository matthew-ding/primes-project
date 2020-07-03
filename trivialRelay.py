import copy
import numpy as np
from scipy.spatial.distance import cdist, euclidean
import pandas as pd
from Graph_Generation import graphGenerator


class Node:
    def __init__(self, n, d):
        self.num = n
        self.parameterList = {n: Parameter(d)}

    def getParams(self):
        return self.parameterList


class Parameter:
    def __init__(self, d):
        self.params = np.zeros([1, d + 1])

    def getParams(self):
        return self.params


def main():
    d = 10  # number of dimensions
    n = 1000  # number of datapoints

    adjList, byzantine_set, m, diameter, target = graphGenerator.get_relay_graph()
    print("Graph Generated Successfully")

    my_data = pd.read_csv("Dataset_Generation\\dataset.txt")

    X_set = []  # list of all x matrices
    Y_set = []  # list of all y vectors
    nodeList = []  # list of all nodes
    cost_set = []  # list of each iterations cost
    byzantine_parameter = np.zeros([1, d + 1])
    for j in range(d):
        byzantine_parameter[0][j] = -1000


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
        nodeList.append(Node(i, d))

        if i in byzantine_set:
            nodeList[i].parameterList[i].params = byzantine_parameter

        cost_set.append({})


    # set hyper parameters
    iters = 0
    costTarget = 1
    costList = [computeCost(X_set[target], Y_set[target], np.zeros([1, d + 1]))]  # costs, main function output

    g, cost, iters = gradientDescent(iters, target, nodeList, diameter, adjList, m, byzantine_set, X_set, Y_set,
                                     byzantine_parameter, cost_set, costTarget, d)

    finalCost = computeCost(X_set[target], Y_set[target], g)
    print("Converges below cost of " + str(costTarget) + " in " + str(iters) + " iterations")
    print("Final Cost: " + str(finalCost) + "\n")

    g = g.tolist()[0]

    for i in range(1, len(g)):
        print("Variable: x" + str(i) + ", Coefficient: " + str(g[i]))

    print("Constant: " + str(g[0]))


    for key in cost:
        costList.append(cost[key])

    return costList, diameter


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


# compute cost
def computeCost(X, y, theta):
    tobesummed = np.power(((X @ theta.T) - y), 2)
    return np.sum(tobesummed) / (2 * len(X))


def broadcast(nodeList, adjList, m):
    tempNodeList = copy.deepcopy(nodeList)  # node list before broadcast starts
    for i in range(m):
        for j in range(m):
            if j in adjList[i]:
                tempNeighbor = tempNodeList[j].getParams()
                currentParamList = nodeList[i].parameterList

                # updating parameters to newest iteration
                for key in tempNeighbor:
                    currentParamList[key] = tempNeighbor[key]


# returns true if node is receiving majority honest parameters
def checkMajority(target, nodeList, byzantine_set):
    honestCounter = 0
    byzantineCounter = 0

    for i in nodeList[target].getParams():
        if i in byzantine_set:
            byzantineCounter += 1
        else:
            honestCounter += 1

    return honestCounter / (honestCounter + byzantineCounter) > 0.5


def gradientDescent(iters, target, nodeList, diameter, adjList, m, byzantine_set, X_set, Y_set,
                    byzantine_parameter, cost_set, costTarget, d):
    while True:
        # neighbor aggregation
        for i in range(diameter):
            iters += 1
            broadcast(nodeList, adjList, m)

        newParamList = {}

        for i in range(m):
            if i not in byzantine_set:
                theta = nodeList[i].parameterList

                # geometric median
                parameterMatrix = []
                for j in range(m):
                    parameterMatrix.append(theta[i].getParams()[0])

                theta[i].params = np.reshape(geometric_median(np.array(parameterMatrix)), (1, d + 1))

                # gradient calculation
                gradient = (1.0 / len(X_set[i])) * np.sum(X_set[i] * (X_set[i] @ theta[i].getParams().T - Y_set[i]),
                                                          axis=0)
                # gradient update
                # alpha = 0.3 / (iters/diameter + 1)
                alpha = 0.01
                newParamList[i] = (theta[i].params - alpha * gradient)
            else:
                newParamList[i] = byzantine_parameter

            # cost calculation
            cost_set[i][iters] = computeCost(X_set[i], Y_set[i], newParamList[i])

        # resetting parameter arrays
        nodeList = []
        for i in range(m):
            nodeList.append(Node(i, d))
            nodeList[i].parameterList[i].params = newParamList[i]

        if iters % (diameter*10) == 0:
            print("Cost at iteration " + str(iters) + ": " + str(cost_set[target][iters]))

        # if float(cost_set[target][iters]) < costTarget:
        #    break

        if iters > 100:
            break

    return nodeList[target].getParams()[target].getParams(), cost_set[target], iters


### FUNCTION CALL
# main()

