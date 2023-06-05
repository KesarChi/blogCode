# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.
import numpy as np

from base_linear import *
from sklearn.linear_model import LinearRegression as lr


class LinearRegression(LinearBase):
    def fit(self, X, Y):
        self._init_params(len(X[0]))
        Loss = [mseLoss(X, Y, X.shape[0], self.weights, self.bias)]
        print("Init Loss: ", Loss[-1])
        for t in range(self.maxIter):
            self.weights -= self.LR * \
                            np.sum(X.T * (np.sum(self.weights * X, axis=1) +
                                          self.bias - Y), axis=1)
            self.bias -= self.LR * \
                         np.sum((np.sum(self.weights * X, axis=1) + self.bias - Y))
            if (t + 1) % 100 == 0:
                Loss.append(mseLoss(X, Y, X.shape[0], self.weights, self.bias))
        print("End Loss: ", mseLoss(X, Y, X.shape[0], self.weights, self.bias))
        self._lossPlot(np.arange(0, self.maxIter + 100, 100), Loss)


if __name__ == "__main__":
    X, Y = create_sample(5000, 5)
    model = LinearRegression(LR=1e-6, maxIter=3000)
    model.fit(X, Y)
    model.show_train_result()

    pkg = lr()
    pkg.fit(X, Y)
    print(pkg.coef_, pkg.intercept_)
    print(mseLoss(X, Y, X.shape[0], pkg.coef_, pkg.intercept_))


