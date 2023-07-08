import math
from math import *
import numpy
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT DATA FROM THE DATA SHEET
data = pd.read_csv('dataset.csv').values
N, d = data.shape
x_train = data[:, 0:d - 1].reshape(-1, d - 1) 
x_train = append(x_train, ones((N, 1)), axis=1)
y_train = data[:, 2].reshape(-1, 1) 
print(y_train)

# TRAIN THE MODEL

#Initialize parameters
w = zeros((d, 1))
def sigmoid(x):
    return (1 / (1 + exp(-x) ) )
#Calculate the error ~ loss function
def logistic_predict(w, x_):
    return sigmoid(x_ @ w)   # [N, d - 1] @ [d - 1, 1] = [N, 1]
def loss(w, x_train, y_train):
    l = log(logistic_predict(w, x_train)).transpose() @ y_train + (ones(N, 1) - y_train).transpose() @ log(ones(N, 1) - logistic_predict(w, x_train))
    return l/N    #scalar
def gradient_descent(w, x_train, y_train, lr, iteration):
    for _ in range(iteration):
        w -= lr * dot(x_train.transpose(), (logistic_predict(w, x_train) - y_train) ) / N
    print(w)

# PREDICTING FUNCTION
gradient_descent(w, x_train, y_train, 0.005, 10000)

# VISUALISE THE MODEL
x1 = x_train[0:10, 0]
x2 = x_train[0:10, 1]
x3 = x_train[10:20, 0]
x4 = x_train[10:20, 1]
plt.scatter(x1, x2, color = "red")
plt.scatter(x3, x4, color = "blue")
plt.show()

def predict(w, input):
    if logistic_predict(w, input) >= 0.5:
        return 1
    return 0

example = [2, 2, 1]
print("Lương ", example[0], "tr")
print("thời gian làm việc:", example[1], "năm")
print("Cho vay: ")
# print(predict(w, example))