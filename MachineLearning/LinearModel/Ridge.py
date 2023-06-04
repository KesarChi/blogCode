# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from base_linear import *


class RidgeRegression(LinearBase):
    def __init__(self, LR=1e-6, maxIter=5000, alpha=1):
        super(RidgeRegression, self).__init__(LR, maxIter)
        self.alpha = alpha

    def fit(self, X, Y):
        self._init_params(len(X[0]))
        Loss = [self.ridgeLoss(X, Y, X.shape[0], self.alpha)]
        print(f"Init Loss {Loss[-1]}")
        for t in range(self.maxIter):
            self.weights -= self.LR * (np.sum(X.T * (np.sum(self.weights * X, axis=1) +
                                                     self.bias - Y), axis=1) + self.alpha * self.weights)
            self.bias -= self.LR * \
                np.sum((np.sum(self.weights * X, axis=1) + self.bias - Y))
            if (t+1) % 100 == 0:
                Loss.append(self.ridgeLoss(X, Y, X.shape[0], self.alpha))
        print("End Loss: ", Loss[-1])
        self._lossPlot(np.arange(0, self.maxIter+100, 100), Loss)


if __name__ == "__main__":
    X, Y = create_sample(5000, 4)
    model = RidgeRegression()
    model.fit(X, Y)