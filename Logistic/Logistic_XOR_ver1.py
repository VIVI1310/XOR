import math
from math import *
import numpy
from numpy import *

global X, Y, n, W 
n = 4 # examples
X1= [[0], [0], [1], [1]]
X2 = [[0], [1], [0], [1]]
Y = [[0], [0], [0], [1]]

X = ones((4, 1), dtype = int)
X = append(X, X1, axis= 1)
X = append(X, X2, axis = 1)

W = zeros((3, 1))
learning_rate = [[0.005], [0.005], [0.005]]

# dim (X @ W) = (4, 1)
# dim y_prediction = (1, 4)
#dim log(y_prediction @ Y) = 1

def sigmoid(x):
    sig = 1 / (1 + exp(-x))
    return sig
def y_prediction():
	prediction = sigmoid(X @ W)
	return prediction.transpose()

def L():
	loss = log(y_prediction() ) @ Y + log(ones((1, 4)) - y_prediction()) * (ones((4, 1)) - Y ) * -1 / n
	return loss

def derivative():
	D = (X.transpose() @ (y_prediction().transpose() - Y) ) * 1 / n
	# (3, 4) @ (4, 1) = (3, 1)
	return D

def predict():
	p = sigmoid(X @ W)
	for _ in range(size(p)):
		if p[_] < 0.5:
			p[_] = int(0)
			print(p[_])
		else:
			p[_] = int(1)
			print(p[_])
	return p
def logistic_regression (iteration ):
	global W
	for _ in range(iteration):
		W -= learning_rate * derivative()
	#print(predict())
	return predict()

logistic_regression (10000)