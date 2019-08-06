import numpy as np
import matplotlib.pyplot as plt


# Generate sample data.
m = 50
X = np.random.randn(m, 3)
X[:, 0] = 1
# Assume k and b for y = kx + b
k = 1.3
b = 0.9
y = np.int8(k * X[:, 1] + b - X[:, 2] > 0)
# Draw the graphic
idx1 = np.where(y == 0)
idx2 = np.where(y == 1)
plt.scatter(X[idx1, 1], X[idx1, 2], marker="o", c="r")
plt.scatter(X[idx2, 1], X[idx2, 2], marker="o", c="b")
plt.show()

# Define parameters
theta = [0, 0]
alpha = 0.1


# Sigmoid function
def g(x):
    return 1 / (1 + np.e ** -x)


# Hypothesis function h
def h(x):
    global theta
    return g(theta[1] * x + theta[0])


# Cost function J
def J():
    global theta, X, y, m
    cost = 0
    for i in range(m):
        x = X[i, 1]
        cost += y[i] * np.log(h(x)) + (1 - y[i]) * np.log(1 - h(x))
    return -cost / m


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
    idx1 = np.where(y == 0)
    idx2 = np.where(y == 1)
    plt.scatter(X[idx1, 1], X[idx1, 2], marker="o", c="r")
    plt.scatter(X[idx2, 1], X[idx2, 2], marker="o", c="b")
    xx = np.linspace(np.min(X[:, 2]), np.max(X[:, 2]))
    yy = xx * theta[1] + theta[0]
    plt.plot(xx, yy)
    plt.show(block=False)
    plt.pause(0.01)
plt.show()
