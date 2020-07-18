# this code is adapted from a standard neural network (does not use inverses, etc.)
# link can be found here: https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/

import copy
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# import mnist
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]
print("MNIST imported")

# scale
X = X / 255

# one-hot encode labels
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

# split, reshape, shuffle
m = 10000
m_test = m
X_train_orig, X_test = X[:m], X[m:2 * m]
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:2 * m]
shuffle_index = np.random.permutation(m)
X_train_orig, Y_train = X_train_orig[shuffle_index, :], Y_train[:, shuffle_index]


# multiply by random gaussian matrix
random_gaussian = np.random.normal(0, 1, (784, m))
X_train_orig = np.matmul(X_train_orig, random_gaussian)
X_test = np.matmul(X_test, random_gaussian)

print("Random Gaussian Created")


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


X_train_orig = sigmoid(X_train_orig).T
X_test = sigmoid(X_test).T

# inverse
X_train_inv = np.linalg.inv(X_train_orig)
print("Inverse Created")

# identity
X_train = np.round(X_train_orig.dot(X_train_inv), decimals=8)
print("Identity Created")


def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1. / m) * L_sum

    return L


def feed_forward(X, params):
    cache = {}

    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache


def back_propagate(X, Y, params, cache):
    dZ2 = cache["A2"] - Y
    dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


# hyperparameters
n_x = X_train.shape[0]
n_h = 64
learning_rate = 4
beta = .9
batch_size = 128
batches = -(-m // batch_size)

# initialization
params = {"W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
          "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
          "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
          "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h)}

V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)

# train
for i in range(50):
    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):
        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache)

        V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])

        params["W1"] = params["W1"] - learning_rate * V_dW1
        params["b1"] = params["b1"] - learning_rate * V_db1
        params["W2"] = params["W2"] - learning_rate * V_dW2
        params["b2"] = params["b2"] - learning_rate * V_db2

    # training cost
    cache = feed_forward(X_train, params)
    train_cost = compute_loss(Y_train, cache["A2"])

    # testing cost
    test_params = copy.deepcopy(params)
    test_params["W1"] = np.matmul(test_params["W1"], X_train_inv)
    cache = feed_forward(X_test, test_params)
    test_cost = compute_loss(Y_test, cache["A2"])

    print("Epoch {}: training cost = {}, test cost = {}".format(i + 1, train_cost, test_cost))

print("Done." + "\n")


## IDENTITY MATRIX LEARNING
print("IDENTITY MATRIX LEARNING RESULTS (TRAINING ERROR)")
cache = feed_forward(X_train, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_train, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))
print("\n")


## TRAINING MATRIX LEARNING
print("TRAINING MATRIX LEARNING RESULTS")
params["W1"] = np.matmul(params["W1"], X_train_inv)
cache = feed_forward(X_train_orig, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_train, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))


## TESTING MATRIX LEARNING
print("TESTING MATRIX LEARNING RESULTS")

cache = feed_forward(X_test, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))
