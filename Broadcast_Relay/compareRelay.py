import matplotlib.pyplot as plt
from Broadcast_Relay import broadcastRelay
from Broadcast_Relay import trivialRelay
from Other import originalLinReg

diameter = 10  # diameter of graph
maxIter = 3000  # number of iterations of gradient descent

print("Beginning Broadcast Relay")
broadcastCost = broadcastRelay.main(diameter, maxIter)

xList = []
for i in range(len(broadcastCost)):
    xList.append(i)


print("\n" + "Beginning Trivial Relay")
trivialCost, diameter, my_data, d, m, n = trivialRelay.main(diameter, maxIter)

print("\n" + "Beginning True Optimization")
originalLinReg.trueOptimization(d, n, m, my_data)


xList2 = []
for i in range(len(trivialCost)):
    xList2.append(i * diameter)

plt.plot(xList, broadcastCost, label="New Broadcast")
plt.plot(xList2, trivialCost, label="Trivial Broadcast")


plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Novel vs Trivial Gradient Descent: Diameter = " + str(diameter))
plt.legend()
plt.savefig("cost" + str(diameter) + ".png")
# plt.show()

