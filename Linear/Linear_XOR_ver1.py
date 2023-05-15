import math
from math import *
import numpy
from numpy import *

global X1, X2, Y, W
X1 = [0, 0, 1, 1]
X2 = [0, 1, 0, 1]
Y = [[0], [0], [0], [1]]
one = [[1], [1], [1], [1]]

n = 4
component = 3
# W = zeros(component, dtype = int)


def y_prediction(w1, w2, bias):
    # y = w1 * x1 + w2 * x2 - bias
    y = multiply(X1, w1) + multiply(X2, w2) - bias
    return y.transpose()


def d_weight1(w1, w2, bias):
    dw1 = 1 / n * 2 * sum(X1 @ (y_prediction(w1, w2, bias) - Y))
    return dw1


def d_weight2(w1, w2, bias):
    dw2 = 1 / n * 2 * sum(X2 @ (y_prediction(w1, w2, bias) - Y))
    return dw2


def d_bias(w1, w2, bias):
    db = 1 / n * 2 * sum(y_prediction(w1, w2, bias) - Y)
    return db


def gradient_descent(lr_w1=0.0001, lr_w2=0.0001, lr_b=0.0001, iteration=100):
    weight1 = 0
    weight2 = 0
    bias = 0
    for _ in range(iteration):

        w1_variation = lr_w1 * d_weight1(weight1, weight2, bias)
        w2_variation = lr_w2 * d_weight2(weight1, weight2, bias)
        b_variation = lr_b * d_bias(weight1, weight2, bias)

        weight1 = weight1 - w1_variation
        weight2 = weight2 - w1_variation
        bias = bias - b_variation

    print(weight1, weight2, bias)
    print()


gradient_descent(0.005, 0.005, 0.005, 100)
