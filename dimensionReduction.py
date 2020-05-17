from sympy import *

# number of dimensions
n = 30

# dictionary of variables
symbolDict = {}
for i in range(n):
    symbolDict["x" + str(i)] = Symbol("x" + str(i))


f = symbolDict["x0"] ** 2
for i in range(1, n):
    f = f + i*(symbolDict["x" + str(i)] - i) ** 2


def function():
    output = f
    for i in range(n):
        output = output.subs(symbolDict["x" + str(i)], thetaDict["x" + str(i)])

    return output

# dictionary of partial derivatives
gradientDict = {}
for i in range(n):
    gradientDict["x" + str(i)] = f.diff(symbolDict["x" + str(i)])


# Data
# dictionary of parameters
thetaDict = {}
for i in range(n):
    thetaDict["x" + str(i)] = i*100
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