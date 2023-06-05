# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def create_sample(n_sample=5000, n_feature=5):
    return make_regression(n_samples=n_sample, n_features=n_feature)


def mseLoss(X, Y, n, w, b):
    return round(0.5 * np.sum((np.sum(X * w, axis=1) + b - Y) ** 2) / n, 4)


def lassoLoss(X, Y, n, w, b, alpha):
    return round(0.5 * (np.sum((np.sum(X * w, axis=1) + b - Y) ** 2)) / n + alpha * np.sum(np.abs(w)), 4)


def ridgeLoss(X, Y, n, w, b, alpha):
    return round(0.5 * (np.sum((np.sum(X * w, axis=1) + b - Y) ** 2) +
                        alpha * np.sum(w ** 2)) / n, 4)


def elasticLoss(X, Y, n, w, b, alpha, rho):
    return 0.5 / n * np.sum(np.sum(X * w, axis=1) + b - Y) ** 2 + alpha * rho * np.sum(
        abs(w)) + 0.5 * alpha * (1 - rho) * np.sum(w ** 2)


class LinearBase:
    def __init__(self, LR=1e-6, maxIter=5000):
        self.weights = []
        self.bias = 0
        self.LR = LR
        self.maxIter = maxIter

    def _init_params(self, n):
        self.weights = np.random.rand(n)
        self.bias = np.random.rand()

    def _lossPlot(self, x, Loss):
        plt.figure(figsize=(6, 4))
        sns.lineplot(x, Loss)
        sns.scatterplot(x, Loss, color='r')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    def show_train_result(self, ):
        print("Weights: ", self.weights)
        print("Bias: ", self.bias)
