import math
from math import *
import numpy
from numpy import *

global X1, X2, Y, W, component, default_rate
X1 = [[0], [0], [1], [1]]
X2 = [[0], [1], [0], [1]]
Y = [[0], [0], [0], [1]]

X = ones((4, 1), dtype=int)
X = append(X, X1, axis=1)
X = append(X, X2, axis=1)  # (4, 3)

default_rate = [[0.005], [0.005], [0.005]]

n = 4  # number of exampples
component = int(3)  # số tham số: bias, w1, w2
W = zeros((3, 1))


def y_prediction():
    y = X @ W   # (4, 1)
    return y


def derived_W():
    d = X.transpose() @ (y_prediction() - Y) * 1 / n * 2
    return d

def predict():
    y = X @ W
    for _ in range (size(y)):
        if abs(1 - y[_]) < abs(y[_]):
            y[_] = int(1)
            print(y[_])
        else:
            y[_] = int(0)
            print(Y[_])
    return y
def gradient_descent(learn_rate, iteration=100):
    global W
    for _ in range(iteration):
        W -= learn_rate * derived_W()
    return predict()

def input():
    global learn_rate, iteration
    learn_rate = zeros(component, dtype, int)
    learn_rate = float(input("Input learning rate (bias first): ").split())
    iteration = int(input("Input iteration (integer) :"))


gradient_descent(default_rate, 10000)
