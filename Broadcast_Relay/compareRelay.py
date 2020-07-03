import matplotlib.pyplot as plt
import broadcastRelay
import trivialRelay

diameter = 100  # diameter of graph
maxIter = 800  # number of iterations of gradient descent

print("Beginning Broadcast Relay")
broadcastCost = broadcastRelay.main(diameter, maxIter)

xList = []
for i in range(len(broadcastCost)):
    xList.append(i)


print("\n" + "Beginning Trivial Relay")
trivialCost, diameter = trivialRelay.main(diameter, maxIter)

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
plt.show()

