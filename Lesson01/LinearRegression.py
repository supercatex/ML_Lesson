import numpy as np
import matplotlib.pyplot as plt


# Generate sample data.
m = 50
X = np.random.randn(m, 2)
X[:, 0] = 1
# Assume k and b for y = kx + b
k = 1.3
b = 0.9
y = k * X[:, 1] + b + np.random.randn(m, 1)[:, 0]
# Draw the graphic
plt.scatter(X[:, 1], y)
plt.show()

# Define parameters
theta = [0, 0]
alpha = 0.1


# Hypothesis function h
def h(x):
    global theta
    return theta[1] * x + theta[0]


# Cost function J
def J():
    global theta, X, y, m
    cost = 0
    for i in range(m):
        x = X[i, 1]
        cost += (h(x) - y[i]) ** 2
    return cost / 2 / m


# Goal
def minJ(j):
    global theta, X, y, m
    delta = 0
    for i in range(m):
        x = X[i, 1]
        delta += (h(x) - y[i]) * X[i, j]
    return delta / m


# Gradient Descent
def GD():
    global theta, alpha
    new_theta = theta.copy()
    for j in range(len(theta)):
        new_theta[j] -= alpha * minJ(j)
    return new_theta


# Training step
for iteration in range(100):
    print("Iteration:", iteration, "Cost:", J(), "theta:", theta)
    theta = GD()

    plt.cla()
    plt.scatter(X[:, 1], y)
    xx = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]))
    yy = xx * theta[1] + theta[0]
    plt.plot(xx, yy)
    plt.show(block=False)
    plt.pause(0.01)
plt.show()
