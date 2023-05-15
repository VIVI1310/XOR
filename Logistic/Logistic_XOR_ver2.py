import math
from math import *
import numpy
from numpy import *

global n
n = 4

learning_rate = [[0.005], [0.005], [0.005]]

# dim (X @ W) = (4, 1)
# dim y_prediction = (1, 4)
#dim log(y_prediction @ Y) = 1

def sigmoid(x):
    sig = 1 / (1 + exp(-x))
    return sig
def y_prediction(W, X, Y):
	prediction = sigmoid(X @ W)
	return prediction.transpose()

def L(W, X, Y):
	loss = log(y_prediction(W, X, Y) ) @ Y + log(ones((1, 4)) - y_prediction(W, X, Y)) * (ones((4, 1)) - Y ) * -1 / n
	return loss

def derivative(W, X, Y):
	D = (X.transpose() @ (y_prediction(W, X, Y).transpose() - Y) ) * 1 / n
	# (3, 4) @ (4, 1) = (3, 1)
	return D

def predict(W, X, Y):
	p = sigmoid(X @ W)
	for _ in range(size(p)):
		if p[_] < 0.5:
			p[_] = int(0)
		else:
			p[_] = int(1)
	return p
def logistic_regression (X, Y):
    iteration = 10000
    W = zeros((3, 1))
    for _ in range(iteration):
        W -= learning_rate * derivative(W, X, Y)

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

logistic_regression (input_X(), input_Y())