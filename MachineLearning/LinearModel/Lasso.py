# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from base_linear import *
from sklearn.linear_model import Lasso


class LassoRegression(LinearBase):
    def __init__(self, LR, maxIter, alpha):
        super(LassoRegression, self).__init__(LR, maxIter)
        self.alpha = alpha

    def fit(self, X, Y):
        self._init_params(len(X[0]))
        Loss = [lassoLoss(X, Y, X.shape[0], self.weights, self.bias, self.alpha)]
        print("Init Loss: ", Loss[-1])
        for t in range(self.maxIter):
            for i, w in enumerate(self.weights):
                grad = np.sum(X[:, i] * (np.sum(X * self.weights, axis=1) + self.bias - Y)) / len(X[0]) + self.alpha * np.sign(w)
                self.weights[i] -= self.LR * grad
                self.bias -= self.LR * np.sum((np.sum(self.weights * X, axis=1) + self.bias - Y))
            if (t + 1) % 100 == 0:
                Loss.append(lassoLoss(X, Y, X.shape[0], self.weights, self.bias, self.alpha))
        print("End Loss: ", lassoLoss(X, Y, X.shape[0], self.weights, self.bias, self.alpha))
        self._lossPlot(np.arange(0, self.maxIter + 100, 100), Loss)


if __name__ == "__main__":
    X, Y = create_sample(5000, 5)
    model = LassoRegression(LR=1e-6, maxIter=5000, alpha=1)
    model.fit(X, Y)
    model.show_train_result()

    pkg = Lasso()
    pkg.fit(X, Y)
    print(pkg.coef_, pkg.intercept_)
    print(lassoLoss(X, Y, X.shape[0], pkg.coef_, pkg.intercept_, 1))


