import numpy as np


def trueOptimization(d, n, m, my_data):
    # d = dimensions
    # n = number of datapoints
    # m = number of nodes
    n = n - n % m

    # setting the matrixes
    # X = my_data.iloc[:, 0:d]
    X = my_data.iloc[:n//m*(m//2), 0:d]
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)

    # y = my_data.iloc[:, d:d + 1].values
    y = my_data.iloc[:n//m*(m//2), d:d + 1].values
    theta = np.zeros([1, d + 1])

    # set hyper parameters
    alpha = 0.01
    iters = 1000


    # compute cost
    def computeCost(X, y, theta):
        tobesummed = np.power(((X @ theta.T) - y), 2)
        return np.sum(tobesummed) / (2 * len(X))


    # gradient descent
    def gradientDescent(X, y, theta, iters, alpha):
        cost = np.zeros(iters)
        for i in range(iters):
            theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
            cost[i] = computeCost(X, y, theta)

            if i % 100 == 0:
                print("Cost at iteration: " + str(i) + ": " + str(cost[i]))

        return theta, cost

    # running the gd and cost function
    g, cost = gradientDescent(X, y, theta, iters, alpha)
    finalCost = computeCost(X, y, g)

    g = g.tolist()[0]

    print("\n")

    for i in range(1, len(g)):
        print("Variable: x" + str(i) + ", Coefficient: " + str(g[i]))

    print("Constant: " + str(g[0]))

    print("Final Cost " + str(finalCost))
