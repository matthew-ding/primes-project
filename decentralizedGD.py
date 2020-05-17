from sympy import *
import numpy as np
from scipy.spatial.distance import cdist, euclidean

### HYPERPARAMETERS
n = 30  # number of dimensions
m = 9  # number of nodes
byzantine_set = [4, 5]  # set of byzantine nodes
adjList = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2, 4, 5],
           4: [3, 6], 5: [3, 6], 6: [4, 5, 7, 8], 7: [6], 8: [6]}

dataset = {}
for i in range(m):
    if i not in byzantine_set:
        list = []
        for j in range(n):
            list.append(np.random.normal(i, 10))
        dataset[i] = list

# dictionary of variables
symbolDict = {}
for i in range(n):
    symbolDict["x" + str(i)] = Symbol("x" + str(i))

f = symbolDict["x0"] ** 2
for i in range(1, n):
    f = f + i * (symbolDict["x" + str(i)] - i) ** 2


def function():
    output = f
    for i in range(n):
        output = output.subs(symbolDict["x" + str(i)], thetaDict["x" + str(i)])

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


X = np.array([[1,1], [5,5]])
print(geometric_median(X))

# dictionary of partial derivatives
gradientDict = {}
for i in range(n):
    gradientDict["x" + str(i)] = f.diff(symbolDict["x" + str(i)])

# Data
# dictionary of parameters
thetaDict = {}
for i in range(n):
    thetaDict["x" + str(i)] = i * 100
alpha = .01
iterations = 0
check = 0
precision = 1 / 1000000
printData = True
maxIterations = 5000

tempThetaDict = {}
while True:
    for i in range(n):
        gradient = gradientDict["x" + str(i)]

        for j in range(n):
            gradient = gradient.subs(symbolDict["x" + str(j)], thetaDict["x" + str(j)])

        tempThetaDict["x" + str(i)] = thetaDict["x" + str(i)] - alpha * N(gradient).evalf()

    # If the number of iterations goes up too much, maybe the parameters
    # are diverging! Let's stop the loop and try to understand.
    iterations += 1
    if iterations > maxIterations:
        print("Too many iterations. Adjust alpha and make sure that the function is convex!")
        printData = False
        break

    # If the value of theta changes less of a certain amount, our goal is met.
    nextIteration = false
    for i in range(n):
        if abs(tempThetaDict["x" + str(i)] - thetaDict["x" + str(i)]) > precision:
            nextIteration = True
            continue

    if not nextIteration:
        break

    # Simultaneous update
    for i in range(n):
        thetaDict["x" + str(i)] = tempThetaDict["x" + str(i)]

    # Cost function output
    if iterations % 50 == 0:
        print("Iteration " + str(iterations) + ", Cost: " + str(function()))

if printData:
    print("The function " + str(f) + " converges to a minimum")
    print("Number of iterations:", iterations, sep=" ")
    for i in range(n):
        print("theta (x" + str(i) + ") =", thetaDict["x" + str(i)], sep=" ")
