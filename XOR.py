import math
from math import *
import numpy
from numpy import *

global n
n = 4
def y_prediction(W, X, Y):
    y = X @ W  # (4, 1)
    return y


def derived_W(W, X, Y):
    d = X.transpose() @ (y_prediction(W, X, Y) - Y) * 1 / n * 2
    return d


def predict(W, X, Y):
    y = X @ W
    for _ in range(n):
        if abs(1 - y[_]) < abs(y[_]):
            y[_] = int(1)
        else:
            y[_] = int(0)
    return y


def gradient_descent(X, Y):
    iteration = 10000
    learn_rate = [[0.005], [0.005], [0.005]]
    W = zeros((3, 1))
    for _ in range(iteration):
        W -= learn_rate * derived_W(W, X, Y)
    print(predict(W, X, Y))
    return predict(W, X, Y)

def column_input(name):
    a = []
    for _ in range(n):
        r = []
        print(name, _ + 1, " element: ")
        r.append(int(input()))
        a.append(r)
    return a

def  input_Y():
    Y = column_input("Y")
    return Y
def input_X():
    X1 = column_input("X1")
    X2 = column_input("X2")
    X = ones((4, 1), dtype=int)
    X = append(X, X1, axis=1)
    X = append(X, X2, axis=1)  # (4, 3)
    return X

gradient_descent(input_X(), input_Y())

# Logistic