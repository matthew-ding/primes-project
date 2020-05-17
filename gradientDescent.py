from sympy import *

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

f = x ** 2 + y ** 2 - 2 * x * y


def function(x, y):
    return x ** 2 + y ** 2 - 2 * x * y


# First partial derivative with respect to x
fpx = f.diff(x)

# First partial derivative with respect to y
fpy = f.diff(y)

# Gradient
grad = [fpx, fpy]

# Data
theta = 830  # x
theta1 = 220  # y
alpha = .01
iterations = 0
check = 0
precision = 1 / 1000000
printData = True
maxIterations = 5000

while True:
    tempthetax = theta - alpha * N(fpx.subs(x, theta).subs(y, theta1)).evalf()
    tempthetay = theta1 - alpha * N(fpy.subs(y, theta1)).subs(x, theta).evalf()

    # If the number of iterations goes up too much, maybe theta (and/or theta1)
    # is diverging! Let's stop the loop and try to understand.
    iterations += 1
    if iterations > maxIterations:
        print("Too many iterations. Adjust alpha and make sure that the function is convex!")
        printData = False
        break

    # If the value of theta changes less of a certain amount, our goal is met.
    if abs(tempthetax - theta) < precision and abs(tempthetay - theta1) < precision:
        break

    # Cost function output
    if iterations % 50 == 0:
        print("Iteration " + str(iterations) + ", Cost: " + str(function(tempthetax, tempthetay)))

    # Simultaneous update
    theta = tempthetax
    theta1 = tempthetay

if printData:
    print("The function " + str(f) + " converges to a minimum")
    print("Number of iterations:", iterations, sep=" ")
    print("theta (x0) =", tempthetax, sep=" ")
    print("theta1 (y0) =", tempthetay, sep=" ")

# Output
#
# The function x**2 - 2*x*y + y**2 converges to a minimum
# Number of iterations: 401
# theta (x0) = 525.000023717248
# theta1 (y0) = 524.999976282752