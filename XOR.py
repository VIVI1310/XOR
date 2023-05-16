import math
from math import *
import numpy
from numpy import *

global n
n = 4


def column_input(name):
    a = []
    for _ in range(n):
        r = []
        print(name, _ + 1, " element: ")
        r.append(int(input()))
        a.append(r)
    return a


def input_Y():
    Y = column_input("Y")
    return Y


def default_X():
    X1 = [[0], [0], [1], [1]]
    X2 = [[0], [1], [0], [1]]
    X = ones((n, 1), dtype=int)
    X = append(X, X1, axis=1)
    X = append(X, X2, axis=1)
    return X


def input_X():
    X1 = column_input("X1")
    X2 = column_input("X2")
    X = ones((n, 1), dtype=int)
    X = append(X, X1, axis=1)
    X = append(X, X2, axis=1)  # (4, 3)
    return X


def input_X(a, b):  # two column matrix
    X = ones((n, 1), dtype=int)
    X = append(X, a, axis=1)
    X = append(X, b, axis=1)

# LINEAR


def Linear_y_prediction(W, X, Y):
    y = X @ W  # (4, 1)
    return y


def Linear_derive(W, X, Y):
    d = X.transpose() @ (Linear_y_prediction(W, X, Y) - Y) * 1 / n * 2
    return d


def Linear_predict(W, X, Y):
    y = X @ W
    for _ in range(n):
        if abs(1 - y[_]) < abs(y[_]):
            y[_] = int(1)
        else:
            y[_] = int(0)
    return y


def Linear(X, Y):
    iteration = 10000
    learn_rate = [[0.005], [0.005], [0.005]]
    W = zeros((3, 1))
    for _ in range(iteration):
        W -= learn_rate * Linear_derive(W, X, Y)
    print(Linear_predict(W, X, Y))
    return Linear_predict(W, X, Y)
# END LINEAR

# NOT


def NOT(m):
    for _ in range(n):
        if m[_] == 0:
            m[_] = 1
        else:
            m[_] = 0
    return m
# END NOT

# Linear(input_X(), input_Y())

# LOGISTIC


def sigmoid(x):
    sig = 1 / (1 + exp(-x))
    return sig


def Logistic_y_prediction(W, X, Y):
    prediction = sigmoid(X @ W)
    return prediction.transpose()


def L(W, X, Y):
    loss = log(Logistic_y_prediction(W, X, Y)) @ Y + log(ones((1, 4)) -
                                                         Logistic_y_prediction(W, X, Y)) * (ones((4, 1)) - Y) * -1 / n
    return loss


def Logistic_derive(W, X, Y):
    D = (X.transpose() @ (Logistic_y_prediction(W, X, Y).transpose() - Y)) * 1 / n
    return D


def Logistic_predict(W, X, Y):
    p = sigmoid(X @ W)
    for _ in range(size(p)):
        if p[_] < 0.5:
            p[_] = int(0)
        else:
            p[_] = int(1)
    return p


def Logistic(X, Y):
    iteration = 10000
    learning_rate = [[0.005], [0.005], [0.005]]
    W = zeros((3, 1))
    for _ in range(iteration):
        W -= learning_rate * Logistic_derive(W, X, Y)

    print(Logistic_predict(W, X, Y))
    return Logistic_predict(W, X, Y)

# Logistic(input_X(), input_Y())
# END LOGISTIC


X1 = [[0], [0], [1], [1]]
X2 = [[0], [1], [0], [1]]
Y_and = [[0], [0], [0], [1]]
Y_or = [[0], [1], [1], [1]]
Y_xor = [[0], [1], [1], [0]]

# test A and B
mAn = Logistic(default_X(), Y_and)
mOn = Logistic(default_X(), Y_or)
print("AND :", mAn)
print("mORn: ")
print(mOn)
NmOn = NOT(mAn)
