# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from base_linear import *
from sklearn.linear_model import ElasticNet


class ElasticNetRegression(LinearBase):
    def __init__(self, LR, maxIter, alpha, rho):
        super(ElasticNetRegression, self).__init__(LR, maxIter)
        self.alpha = alpha
        self.rho = rho

    def fit(self, X, Y):
        self._init_params(len(X[0]))
        Loss = [elasticLoss(X, Y, len(X[0]), self.weights, self.bias, self.alpha, self.rho)]
        print("Init Loss: ", Loss[-1])
        for t in range(self.maxIter):
            for i, w in enumerate(self.weights):
                # grad = np.sum(X[:, i] * (np.sum(X * self.weights, axis=1) + self.bias - Y)) / len(
                #     X[0]) + self.alpha * self.rho * np.sign(w) + self.alpha * (1 - self.rho) * w
                #             self.weights[i] -= self.LR * grad
                #             self.bias -= self.LR * np.sum((np.sum(self.weights * X, axis=1) + self.bias - Y))
                grad = np.dot(X[:, i], np.sum(X * self.weights, axis=1) + self.bias - Y) / len(
                    X) + self.alpha * self.rho * np.sign(w) + self.alpha * (1 - self.rho) * w
                self.weights[i] -= self.LR * grad
                self.bias -= self.LR * np.sum(np.sum(self.weights * X, axis=1) + self.bias - Y) / len(X)
            if (t + 1) % 100 == 0:
                Loss.append(elasticLoss(X, Y, len(X[0]), self.weights, self.bias, self.alpha, self.rho))
        print("End Loss: ", elasticLoss(X, Y, len(X[0]), self.weights, self.bias, self.alpha, self.rho))
        self._lossPlot(np.arange(0, self.maxIter + 100, 100), Loss)


if __name__ == "__main__":
    X, Y = create_sample(5000, 5)
    model = ElasticNetRegression(LR=1e-2, maxIter=3000, alpha=1, rho=0.5)
    model.fit(X, Y)
    model.show_train_result()

    pkg = ElasticNet(max_iter=3000)
    pkg.fit(X, Y)
    print(pkg.coef_, pkg.intercept_)
    print(elasticLoss(X, Y, X.shape[0], pkg.coef_, pkg.intercept_, 1, 0.5))