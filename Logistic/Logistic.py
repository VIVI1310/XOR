import numpy as np
import copy
import math
import matplotlib
from mathplotlib import pyplot as pllt

X1 = np.random.randint(1, 50, size = 50)
X2 = np.random.randint(60, 101, size = 50)
#nửa đầu là các số nhỏ hơn 50
#nửa sau là các số lớn hơn 50
X = np.concatenate((X1,X2))
print(X)
Y1 = np.random.randint(0, 1, size = 50)
Y2 = np.random.randint(1, 2, size = 50)
#nửa đầu = 0
#nửa sau = 1
Y = np.concatenate((Y1,Y2))
print(Y)

# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(X, Y, marker='x', c='r')

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
#plt.show()





def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def y_prediction(w, bias, x):
	prediction = sigmoid(w * x + bias)
	return prediction

def L(w, bias, X, Y, n = 100):
	loss = 0
	for _ in range(n):

		temp = -(1/n) * (Y[_] * math.log(y_prediction(w, bias, X[_])) + (1 - Y[_]) * math.log(1 - y_prediction(w, bias, X[_])))
		loss = loss + temp
	return loss

def db(w, bias, X, Y, n):
	DB = 0
	for _ in range(n):
		temp = (1/n) * (y_prediction(w, bias, X[_]) - Y[n])
		DB = DB + temp
	return DB

def dw(w, bias, X, Y, n):
	DW = 0
	for _ in range(n):
		temp = (1/n) * X[n] * (y_prediction(w, bias, X[_]) - Y[n])
		DW = DW + temp
	return DW


def Draw(w, b, n):
	x = range(100)
	model = np.random.randint(-1, 0, size = 100)
	for i in range(n + 1):
		model[i] = y_prediction(w, b, x[i])
	plt.scatter(x, model, marker='x', c='r')

	# Set the title
	plt.title("Training")
	# Set the y-axis label
	plt.ylabel('guess')
	# Set the x-axis label
	plt.xlabel('data')
	plt.show()

def logistic_regression (X, Y, lr_w , lr_b, n, iteration ):
	weight = 0
	bias = 0

	for _ in range(iteration):
		w_variation = lr_w * dw(weight, bias, X, Y, n)
		b_variation = lr_b * db(weight, bias, X, Y, n)
		###print("weight", _ ,": ",weight, "_bias ",_,": ", bias)
		weight = weight - w_variation
		bias = bias - b_variation

	print("weight = ", weight, "_bias = ", bias)
	Draw(weight, bias, n)


logistic_regression (X, Y, 0.001, 0.0002, 99, 1000)