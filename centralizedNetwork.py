import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
import matplotlib.pyplot as plt

# importing data, setting up 100 separate nodes
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]
X = X / 255

# preparing data set
digits = 10
examples = y.shape[0]
n = 50  # NUMBER OF NODES
byzantine_set = []

# list of node data
xTrainList = []
yTrainList = []

y = y.reshape(1, examples)

# iterating through each node
for i in range(n):
    Y_new = np.eye(digits)[y.astype('int32')]
    Y_new = Y_new.T.reshape(digits, examples)

    m = 60000
    m_test = X.shape[0] - m

    # splitting data into n nodes
    X_train = X[m // n * i:m * (i + 1) // n].T
    Y_train = Y_new[:, m // n * i:m // n * (i + 1)]
    # randomizing order
    # shuffle_index = np.random.permutation(m//100)
    # X_train, Y_train = X_train[: ,shuffle_index], Y_train[:, shuffle_index]

    xTrainList.append(X_train)
    yTrainList.append(Y_train)


X_test = X[m:].T
Y_test = Y_new[:, m:]


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def relu(z):
    s = np.maximum(0, z)
    return s


def compute_loss(Y0, Y_hat):
    L_sum = np.sum(np.multiply(Y0, np.log(Y_hat)))
    m0 = Y0.shape[1]
    L = -(1. / m0) * L_sum

    return L


def feed_forward(X, params):
    thisCache = {}

    thisCache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    thisCache["A1"] = sigmoid(thisCache["Z1"])
    thisCache["Z2"] = np.matmul(params["W2"], thisCache["A1"]) + params["b2"]
    thisCache["A2"] = np.exp(thisCache["Z2"]) / np.sum(np.exp(thisCache["Z2"]), axis=0)

    return thisCache


def back_propagate(X, Y, params, thisCache):
    dZ2 = thisCache["A2"] - Y
    dW2 = (1 / xTrainList[0].shape[1]) * np.matmul(dZ2, thisCache["A1"].T)
    db2 = (1 / xTrainList[0].shape[1]) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(thisCache["Z1"]) * (1 - sigmoid(thisCache["Z1"]))
    dW1 = (1 / xTrainList[0].shape[1]) * np.matmul(dZ1, X.T)
    db1 = (1 / xTrainList[0].shape[1]) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


def find_average(this_dict):
    total = 0
    counter = 0
    for n in this_dict:
        total = total + this_dict[n]
        counter = counter + 1

    return total / counter


def update_step():
    # calculating mean gradient vector
    mean_V_dW1 = np.zeros(params["W1"].shape)
    mean_V_db1 = np.zeros(params["b1"].shape)
    mean_V_dW2 = np.zeros(params["W2"].shape)
    mean_V_db2 = np.zeros(params["b2"].shape)

    for i in range(n):
        mean_V_dW1 = mean_V_dW1 + derivatives["V_dW1i" + str(i)]
        mean_V_db1 = mean_V_db1 + derivatives["V_db1i" + str(i)]
        mean_V_dW2 = mean_V_dW2 + derivatives["V_dW2i" + str(i)]
        mean_V_db2 = mean_V_db2 + derivatives["V_db2i" + str(i)]

    mean_V_dW1 = mean_V_dW1 / n
    mean_V_db1 = mean_V_db1 / n
    mean_V_dW2 = mean_V_dW2 / n
    mean_V_db2 = mean_V_db2 / n

    # update step
    params["W1"] = params["W1"] - learning_rate * mean_V_dW1
    params["b1"] = params["b1"] - learning_rate * mean_V_db1
    params["W2"] = params["W2"] - learning_rate * mean_V_dW2
    params["b2"] = params["b2"] - learning_rate * mean_V_db2


### LEARNING STUFF ###
np.random.seed(138)

# hyperparameters
n_x = xTrainList[0].shape[0]
n_h = 64
learning_rate = 1


### Initializing parameters ###
params = {}
derivatives = {}

params["W1"] = np.random.randn(n_h, n_x) * np.sqrt(1. / n_x)
params["b1"] = np.zeros((n_h, 1)) * np.sqrt(1. / n_x)
params["W2"] = np.random.randn(digits, n_h) * np.sqrt(1. / n_h)
params["b2"] = np.zeros((digits, 1)) * np.sqrt(1. / n_h)

for i in range(n):
    derivatives["V_dW1i" + str(i)] = np.zeros(params["W1"].shape)
    derivatives["V_db1i" + str(i)] = np.zeros(params["b1"].shape)
    derivatives["V_dW2i" + str(i)] = np.zeros(params["W2"].shape)
    derivatives["V_db2i" + str(i)] = np.zeros(params["b2"].shape)

### TRAINING THE MODEL ###
cacheList = {}
trainCostList = {}
testCostList = {}
for i in range(n):
    cacheList[str(i)] = 0
    trainCostList[str(i)] = 0
    testCostList[str(i)] = 0

# training step: loop through epochs
for x in range(50):
    permutation = np.random.permutation(xTrainList[i].shape[1])
    xTrainList[i] = xTrainList[i][:, permutation]
    yTrainList[i] = yTrainList[i][:, permutation]

    # looping through each nodes training step
    for i in range(n):
        ### CORRUPT NODES
        if i in byzantine_set:
            derivatives["V_dW1i" + str(i)] = 1
            derivatives["V_db1i" + str(i)] = 1
            derivatives["V_dW2i" + str(i)] = -1
            derivatives["V_db2i" + str(i)] = -1
        ### HONEST NODES
        else:
            cacheList[i] = feed_forward(xTrainList[i], params)
            grads = back_propagate(xTrainList[i], yTrainList[i], params, cacheList[i])

            ## GRADIENTS
            derivatives["V_dW1i" + str(i)] = grads["dW1"]
            derivatives["V_db1i" + str(i)] = grads["db1"]
            derivatives["V_dW2i" + str(i)] = grads["dW2"]
            derivatives["V_db2i" + str(i)] = grads["db2"]

            cacheList[i] = feed_forward(xTrainList[i], params)
            trainCostList[i] = compute_loss(yTrainList[i], cacheList[i]["A2"])
            cacheList[i] = feed_forward(X_test, params)
            testCostList[i] = compute_loss(Y_test, cacheList[i]["A2"])


    ### UPDATING PARAMETERS ###
    update_step()

    # calculate mean costs
    train_cost = find_average(trainCostList)
    test_cost = find_average(testCostList)

    if (x + 1) % 10 == 0:
        print("Epoch {}: training cost = {}, test cost = {}".format(x + 1, train_cost, test_cost))

print("Done.")

### post-analysis ###
cache = feed_forward(X_test, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))

