from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# import
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

# scale
X = X / 255

# one-hot encode labels
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

# split, reshape
m = 784
m_test = X.shape[0] - m
X_train = X[:m].T
Y_train = Y_new[:,:m]

# random gaussian matrix
X_train_orig = np.random.normal(0, 1, (784, 784))
# inverse
X_train_inv = np.linalg.inv(X_train_orig)
# identity
X_train = np.round(X_train_orig.dot(X_train_inv), decimals=8)


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def compute_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

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
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


np.random.seed(138)

# hyperparameters
n_x = X_train.shape[0]
n_h = 64
learning_rate = 4
beta = .9
batch_size = 128
batches = -(-m // batch_size)

# initialization
params = { "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
           "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
           "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
           "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h) }

V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)

# train
for i in range(100):

    for j in range(batches):

        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train[:, begin:end]
        Y = Y_train[:, begin:end]
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

    cache = feed_forward(X_train, params)
    train_cost = compute_loss(Y_train, cache["A2"])
    print("Epoch {}: training cost = {}".format(i+1, train_cost))

print("Done." + "\n")


## IDENTITY MATRIX LEARNING
print("IDENTITY MATRIX LEARNING RESULTS")
cache = feed_forward(X_train, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_train, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))
print("\n")


## ORIGINAL MATRIX LEARNING
print("ORIGINAL MATRIX LEARNING RESULTS")
params["W1"] = np.matmul(params["W1"], X_train_inv)
cache = feed_forward(X_train_orig, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_train, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))
