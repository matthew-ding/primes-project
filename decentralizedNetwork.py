import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
import matplotlib.pyplot as plt

# importing data, setting up n separate nodes
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]
X = X / 255

# preparing data set
digits = 10
examples = y.shape[0]
n = 100  # NUMBER OF NODES
byzantine_set = []

adjList = {}
for i in range(n):
    adjList["Node" + str(i)] = []
    for j in range(n):
        if i != j:
            adjList["Node" + str(i)].append(j)

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


def feed_forward(X, params, i):
    thisCache = {}

    thisCache["Z1"] = np.matmul(params["W1i" + str(i)], X) + params["b1i" + str(i)]
    thisCache["A1"] = sigmoid(thisCache["Z1"])
    thisCache["Z2"] = np.matmul(params["W2i" + str(i)], thisCache["A1"]) + params["b2i" + str(i)]
    thisCache["A2"] = np.exp(thisCache["Z2"]) / np.sum(np.exp(thisCache["Z2"]), axis=0)

    return thisCache


def back_propagate(X, Y, params, thisCache, i):
    dZ2 = thisCache["A2"] - Y
    dW2 = (1 / xTrainList[0].shape[1]) * np.matmul(dZ2, thisCache["A1"].T)
    db2 = (1 / xTrainList[0].shape[1]) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2i" + str(i)].T, dZ2)
    dZ1 = dA1 * sigmoid(thisCache["Z1"]) * (1 - sigmoid(thisCache["Z1"]))
    dW1 = (1 / xTrainList[0].shape[1]) * np.matmul(dZ1, X.T)
    db1 = (1 / xTrainList[0].shape[1]) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


def find_average(this_dict):
    total = 0
    counter = 0
    for n in this_dict:
        if n not in byzantine_set:
            total = total + this_dict[n]
            counter = counter + 1

    return total / counter


def update_step(g):
    # calculating mean gradient vector
    mean_V_dW1 = np.zeros(params["W1i0"].shape)
    mean_V_db1 = np.zeros(params["b1i0"].shape)
    mean_V_dW2 = np.zeros(params["W2i0"].shape)
    mean_V_db2 = np.zeros(params["b2i0"].shape)

    for i in adjList["Node" + str(g)]:
        mean_V_dW1 = mean_V_dW1 + derivatives["V_dW1i" + str(i)]
        mean_V_db1 = mean_V_db1 + derivatives["V_db1i" + str(i)]
        mean_V_dW2 = mean_V_dW2 + derivatives["V_dW2i" + str(i)]
        mean_V_db2 = mean_V_db2 + derivatives["V_db2i" + str(i)]

    mean_V_dW1 = mean_V_dW1 / len(adjList["Node" + str(g)])
    mean_V_db1 = mean_V_db1 / len(adjList["Node" + str(g)])
    mean_V_dW2 = mean_V_dW2 / len(adjList["Node" + str(g)])
    mean_V_db2 = mean_V_db2 / len(adjList["Node" + str(g)])

    # update step
    for i in range(n):
        params["W1i" + str(i)] = params["W1i" + str(i)] - learning_rate * mean_V_dW1
        params["b1i" + str(i)] = params["b1i" + str(i)] - learning_rate * mean_V_db1
        params["W2i" + str(i)] = params["W2i" + str(i)] - learning_rate * mean_V_dW2
        params["b2i" + str(i)] = params["b2i" + str(i)] - learning_rate * mean_V_db2


### LEARNING STUFF ###
np.random.seed(138)

# hyperparameters
n_x = xTrainList[0].shape[0]
n_h = 64
learning_rate = 1

### Initializing parameters ###
params = {}
derivatives = {}
for i in range(n):
    params["W1i" + str(i)] = np.random.randn(n_h, n_x) * np.sqrt(1. / n_x)
    params["b1i" + str(i)] = np.zeros((n_h, 1)) * np.sqrt(1. / n_x)
    params["W2i" + str(i)] = np.random.randn(digits, n_h) * np.sqrt(1. / n_h)
    params["b2i" + str(i)] = np.zeros((digits, 1)) * np.sqrt(1. / n_h)

    derivatives["V_dW1i" + str(i)] = np.zeros(params["W1i0"].shape)
    derivatives["V_db1i" + str(i)] = np.zeros(params["b1i0"].shape)
    derivatives["V_dW2i" + str(i)] = np.zeros(params["W2i0"].shape)
    derivatives["V_db2i" + str(i)] = np.zeros(params["b2i0"].shape)

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
            derivatives["V_dW1i" + str(i)].fill(5)
            derivatives["V_db1i" + str(i)].fill(5)
            derivatives["V_dW2i" + str(i)].fill(-5)
            derivatives["V_db2i" + str(i)].fill(-5)
        ### HONEST NODES
        else:
            cacheList[i] = feed_forward(xTrainList[i], params, i)
            grads = back_propagate(xTrainList[i], yTrainList[i], params, cacheList[i], i)

            ## GRADIENTS
            derivatives["V_dW1i" + str(i)] = grads["dW1"]
            derivatives["V_db1i" + str(i)] = grads["db1"]
            derivatives["V_dW2i" + str(i)] = grads["dW2"]
            derivatives["V_db2i" + str(i)] = grads["db2"]

            cacheList[i] = feed_forward(xTrainList[i], params, i)
            trainCostList[i] = compute_loss(yTrainList[i], cacheList[i]["A2"])
            cacheList[i] = feed_forward(X_test, params, i)
            testCostList[i] = compute_loss(Y_test, cacheList[i]["A2"])

    ### UPDATING PARAMETERS
    for i in range(n):
        update_step(i)

    # calculate mean costs
    train_cost = find_average(trainCostList)
    test_cost = find_average(testCostList)

    if (x + 1) % 10 == 0:
        print("Epoch {}: training cost = {}, test cost = {}".format(x + 1, train_cost, test_cost))

print("Done.")

### post-analysis ###
accuracy = 0
counter = 0
for i in range(n):
    if i not in byzantine_set:
        cache = feed_forward(X_test, params, i)
        predictions = np.argmax(cache["A2"], axis=0)
        labels = np.argmax(Y_test, axis=0)

        if i == 0:
            print("Node 0 Data:")
            print(confusion_matrix(predictions, labels))
            print(classification_report(predictions, labels))

        print("Node " + str(i) + " Accuracy: " + str(accuracy_score(predictions, labels)))
        accuracy = accuracy + accuracy_score(predictions, labels)
        counter = counter + 1

print("Average Accuracy: " + str(accuracy/counter))
