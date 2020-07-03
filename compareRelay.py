import matplotlib.pyplot as plt
import broadcastRelay
import trivialRelay


print("Beginning Broadcast Relay")
broadcastCost = broadcastRelay.main()

xList = []
for i in range(len(broadcastCost)):
    xList.append(i)


print("\n" + "Beginning Trivial Relay")
trivialCost, diameter = trivialRelay.main()

xList2 = []
for i in range(len(trivialCost)):
    xList2.append(i * diameter)

plt.plot(xList, broadcastCost, label="New Broadcast")
plt.plot(xList2, trivialCost, label="Trivial Broadcast")


plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Novel vs Trivial Gradient Descent")
plt.legend()
plt.savefig("cost.png")
plt.show()

