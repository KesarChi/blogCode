# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from base_linear import *


class LinearRegression(LinearBase):
    def mseLoss(self, x, y, n):
        return round(0.5 * np.sum((np.sum(x * self.weights, axis=1) + self.bias - y) ** 2), 4) / n

    def fit(self, X, Y):
        self._init_params(len(X[0]))
        Loss = [self.mseLoss(X, Y, X.shape[0])]
        print("Init Loss: ", Loss[-1])
        for t in range(self.maxIter):
            self.weights -= self.LR * \
                            np.sum(X.T * (np.sum(self.weights * X, axis=1) +
                                          self.bias - Y), axis=1)
            self.bias -= self.LR * \
                         np.sum((np.sum(self.weights * X, axis=1) + self.bias - Y))
            if (t + 1) % 100 == 0:
                Loss.append(self.mseLoss(X, Y, X.shape[0]))
        print("End Loss: ", self.mseLoss(X, Y, X.shape[0]))
        self._lossPlot(np.arange(0, self.maxIter + 100, 100), Loss)


if __name__ == "__main__":
    X, Y = create_sample(5000, 5)
    model = LinearRegression(LR=1e-6, maxIter=3000)
    model.fit(X, Y)


