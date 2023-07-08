from math import *
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT DATA FROM THE DATA SHEET
data_And = pd.read_csv('AND.csv').values
N_And, d_And = data_And.shape
xAnd_train = data_And[:, 0:d_And - 1].reshape(-1, d_And - 1)
xAnd_train = append(xAnd_train, ones((N_And, 1)), axis=1)
yAnd_train = data_And[:, d_And - 1].reshape(-1, 1)  # reshape to convert to column

data_Or = pd.read_csv('OR.csv').values
N_Or, d_Or = data_Or.shape
xOr_train = data_Or[:, 0: d_Or - 1]
xOr_train = append(xOr_train, ones((N_Or, 1)), axis=1)
yOr_train = data_Or[:, d_Or - 1].reshape(-1, 1)

# NOT
def NOT(x, N):
    for _ in range(N):
        if x[_] == 0:
            x[_] = 1
        else:
            x[_] = 0
    return x

# LOGISTIC
def sigmoid(x):
    return (1 / (1 + exp(-x) ) )
#Calculate the error ~ loss function
# def logistic_predict(w, x_):
#     return sigmoid(x_ @ w)   # [N, d - 1] @ [d - 1, 1] = [N, 1]
def loss(w, x_train, y_train, N):
    l = log(sigmoid(x_train @ w).transpose() @ y_train) + (ones(N, 1) - y_train).transpose() @ log(ones(N, 1) - sigmoid(x_train @ w) )
    return l/N    #scalar
def logistic_graddescent(w, x_train, y_train, lr, iteration, N):
    for _ in range(iteration):
        w -= lr * dot(x_train.transpose(), (sigmoid(x_train @ w) - y_train) ) / N
    return w
def logistic_predict(w, input, N):
    y = zeros((N, 1))
    for _ in range(N):
        if sigmoid(input[_] @ w) >= 0.5:
            y[_] = 1
    return y

#Initialize parameters
w_And = zeros((d_And, 1))
w_Or = zeros((d_Or, 1))
w_xor = zeros((d_And, 1))

# NOT(A AND B) AND (A OR B) <=> xor( n(a()), o())

# (A AND B)
wfa = logistic_graddescent(w_And, xAnd_train, yAnd_train, 0.01, 8000, N_And) # weight of function a
fa = logistic_predict(wfa, xAnd_train, N_And)

# NOT (A AND B)
fn = NOT(fa, N_And) # value

# A OR B
wfo = logistic_graddescent(w_Or, xOr_train, yOr_train, 0.01, 2000, N_Or)
fo = logistic_predict(wfo, xOr_train, N_Or)

# NOT(A AND B) AND (A OR B)
xor = append(fn, fo, axis=1)
xor = append(xor, ones((N_And, 1)), axis=1)
print("xor train data:", xor)
wxor = logistic_graddescent(w_xor, xor, [[0], [1], [1], [0]], 0.05, 5000, N_And)
print("xor weight",wxor)
fxor = logistic_predict(w_xor, xor, N_And)
print(fxor)