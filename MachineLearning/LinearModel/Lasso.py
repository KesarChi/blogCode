# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.
import numpy as np

from base_linear import *
import copy
from sympy import diff, symbols


class LassoRegression(LinearBase):
    def __init__(self, LR, maxIter, alpha):
        super(LassoRegression, self).__init__(LR, maxIter)
        self.alpha = alpha

    def lassoLoss(self, X, Y, n, alpha, w, b):
        return round(0.5 * (np.sum((np.sum(X * w, axis=1) + b - Y) ** 2) + alpha * np.sum(np.abs(w))) / n, 4)

    def fit(self, X, Y):
        self._init_params(len(X[0]))
        Loss = [self.lassoLoss(X, Y, X.shape[0], self.alpha, self.weights, self.bias)]
        print("Init Loss: ", Loss[-1])
        for t in range(self.maxIter):
            tmp = copy.copy(self.weights)
            for i, w in enumerate(self.weights):
                grad = np.sum(X[:, i] * (np.sum(X * self.weights, axis=1) + self.bias - Y)) / len(X[0]) + self.alpha * np.sign(w)
                self.weights[i] -= self.LR * grad
                self.bias -= self.LR * np.sum((np.sum(self.weights * X, axis=1) + self.bias - Y))
            if (t + 1) % 100 == 0:
                Loss.append(self.lassoLoss(X, Y, X.shape[0], self.alpha, self.weights, self.bias))
        print("End Loss: ", self.lassoLoss(X, Y, X.shape[0], self.alpha, self.weights, self.bias))
        self._lossPlot(np.arange(0, self.maxIter + 100, 100), Loss)


if __name__ == "__main__":
    X, Y = create_sample(5000, 5)
    model = LassoRegression(LR=1e-6, maxIter=3000, alpha=1)
    model.fit(X, Y)


