import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = 30  # number of dimensions

# my_data = pd.read_csv('home.txt', names=["size", "bedroom", "price"])
my_data = pd.read_csv('dataset.txt')  # , names=nm)

print(my_data)

# we need to normalize the features using mean normalization
# my_data = (my_data - my_data.mean()) / my_data.std()

# setting the matrixes
X = my_data.iloc[:, 0:d]
ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)

y = my_data.iloc[:, d:d + 1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1, d + 1])


# computecost
def computeCost(X, y, theta):
    tobesummed = np.power(((X @ theta.T) - y), 2)
    return np.sum(tobesummed) / (2 * len(X))


def gradientDescent(X, y, theta, iters, alpha):
    cost = [0]
    while True:
        iters += 1

        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost.append(computeCost(X, y, theta))

        if iters % 50 == 0:
            print("Cost at iteration " + str(iters) + ": " + str(cost[iters]))

        if iters != 1:
            if abs(cost[iters] - cost[iters - 1]) < precision:
                break

    return theta, cost


# set hyper parameters
alpha = 0.01
iters = 0
precision = 0.0000001

g, cost = gradientDescent(X, y, theta, iters, alpha)

finalCost = computeCost(X, y, g)
print("Converges, Final Cost: " + str(finalCost) + "\n")

g = g.tolist()[0]

for i in range(1, len(g)):
    print("Variable: x" + str(i+1) + ", Coefficient: " + str(g[i]))

print("Constant: " + str(g[0]))

# fig, ax = plt.subplots()
# ax.plot(np.arange(iters), cost, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
