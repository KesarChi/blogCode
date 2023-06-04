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
