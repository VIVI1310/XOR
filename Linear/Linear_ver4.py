from math import *
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

# IMPORT DATA
data = pd.read_csv('data_linear.csv').values    # nhập hêết dữ liệu từ data sheet
N, d = data.shape       # N = row   d = column  # lấy số cột và dòng
x = data[:, 0].reshape(-1, 1)                   # tachs dữ liệu của x -> ma trận cột [N, 1]
x_train = append(x, ones((N, 1)), axis = 1)     # thêm 1 côt và fill = 1
y_train = data[:, 1].reshape(-1, 1)             # tách dữ liệu của y

# TRAIN MODEL
w = zeros((d, 1))
def linear_predict(w, x_):
    return x_ @ w       #predict y
def gradient_descent(w, x_train, y_train, lr, iteration):
    for _ in range(iteration):
        w -= multiply(lr / N, x_train.transpose() @ (linear_predict(w, x_train) - y_train) ) 
    print(w)

# MODEL
gradient_descent(w, x_train, y_train, 0.000005, 2000)

# TEST
example = 50
print(linear_predict(w, [example,1]))

#VISUALIZE DATA AND PREDICTION
plt.scatter(x, y_train, color = "blue")
plt.scatter(50, linear_predict(w, [example,1]), color = "red")
plt.show()