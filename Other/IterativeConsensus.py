import copy
import random
import sys
import numpy as np
from Graph_Generation import graphGenerator
from bisect import bisect_left, bisect_right

sys.path.append('../')


class Node:
    def __init__(self, n, m):
        self.num = n
        self.parameterList = {n: Parameter(0)}
        self.m = m

    def getParams(self):
        return self.parameterList

    def __str__(self):
        output = ""
        for i in range(self.m):
            if i in self.getParams():
                output += "Parameter " + str(i) + ": " + str(self.getParams()[i].getParams()) + "\n"
            else:
                output += "Parameter " + str(i) + ": NONE" + "\n"

        return output + "\n"


class Parameter:
    def __init__(self, it):
        self.iter = it
        self.params = 0

    def getIter(self):
        return self.iter

    def getParams(self):
        return self.params



def trimmed_mean(X, self_value, f):
    sorted_X = np.sort(X)
    index = bisect_right(sorted_X, self_value) - 1
    size = len(sorted_X)

    if size - index - 1 < f:
        sorted_X = sorted_X[:index + 1]
    else:
        sorted_X = sorted_X[:size - f]

    index = bisect_left(sorted_X, self_value)

    if index < f:
        sorted_X = sorted_X[index:]
    else:
        sorted_X = sorted_X[f:]

    return sum(sorted_X) / len(sorted_X)


def broadcast(nodeList, m, adjList, byzantine_set):
    tempNodeList = copy.deepcopy(nodeList)  # node list before broadcast starts
    for i in range(m):
        if i not in byzantine_set:  # byzantine nodes don't store other node's data
            for j in adjList[i]:
                tempNeighbor = tempNodeList[j].getParams()
                currentParamList = nodeList[i].parameterList

                # updating parameters to newest iteration
                for key in tempNeighbor:
                    if key not in currentParamList:
                        currentParamList[key] = tempNeighbor[key]
                    elif currentParamList[key].getIter() < tempNeighbor[key].getIter():
                        currentParamList[key] = tempNeighbor[key]


# Iterative Approximate Byzantine Consensus (relay version)
def IABC(iters, nodeList, m, adjList, diameter, maxIter, byzantine_set):
    while True:
        # neighbor aggregation
        broadcast(nodeList, m, adjList, byzantine_set)

        for i in range(m):
            if iters >= diameter - 1 and i not in byzantine_set:
                theta = nodeList[i].parameterList

                parameterArray = []
                for j in theta:
                    parameterArray.append(theta[j].getParams())

                # Trimmed mean
                theta[i].params = trimmed_mean(parameterArray, theta[i].getParams(), len(byzantine_set))

                theta[i].iter = iters

        if (iters + 1) % 10 == 0:
            print(nodeList[10])

        if iters > maxIter:
            break

        iters += 1

    return nodeList, iters


diameter = 10
maxIter = 1000

# Graph generation
adjList, byzantine_set, m, target = graphGenerator.get_relay_graph(diameter)
print("Graph Generated Successfully")

nodeList = []  # list of all nodes

for i in range(m):
    # parameter arrays
    nodeList.append(Node(i, m))

# setting initial parameters
for i in range(len(byzantine_set)):
    nodeList[i].parameterList[i].params = random.randrange(9, 150)

for i in range(len(byzantine_set), m):
    nodeList[i].parameterList[i].params = random.randrange(9, 110)

iters = 0

g, iters = IABC(iters, nodeList, m, adjList, diameter, maxIter, byzantine_set)

print("Converges in " + str(iters) + " iterations")

for j in range(diameter, m):
    print("NODE: " + str(j))
    print(g[j].getParams()[j].getParams())
