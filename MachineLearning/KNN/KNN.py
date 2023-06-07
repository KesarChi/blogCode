# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import operator
import numpy as np
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')


class KNN:
    def __init__(self, k):
        self.K = k
        self.X = np.array([])
        self.Label = []
        self.N = 0

    def fit(self, X, Y):
        self.X = np.array(X)
        self.Label = Y
        self.N = X.shape[0]

    def predict(self, data):
        data = np.array(data)
        res = []
        if data.shape[1] != self.X.shape[1]:
            raise IndexError("Dimension of input must equals to {}".format(self.X.shape[1]))
        if data.ndim == 1:
            raise ValueError("The format of input data must be: [[1,2,3]] or [[1,2,3],[2,3,4]]")
        for i, d in enumerate(data):
            temp = np.tile(d, (self.X.shape[0], 1))
            diff = (((temp - self.X) ** 2).sum(axis=1)) ** 0.5
            idx = diff.argsort()
            classCnt = dict()
            for j in range(self.K):
                classCnt[self.Label[idx[j]]] = classCnt.get(self.Label[idx[j]], 0) + 1
            res.append(sorted(classCnt.items(), key=operator.itemgetter(1), reverse=True)[0][0])
        return res


def evaluate(pred, True_label):
    nums = len(np.where(pred == True_label)[0])
    print("Accuracy: {}/{}".format(nums, len(pred)))


if __name__ == "__main__":
    X, Y = make_classification(n_samples=5000, n_informative=5, n_features=5, n_repeated=0, n_redundant=0, n_classes=5)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, shuffle=True)

    model = KNN(k=5)
    model.fit(train_X, train_Y)
    res_model = model.predict(test_X)
    evaluate(res_model, test_Y)

    pkg = KNeighborsClassifier()
    pkg.fit(train_X, train_Y)
    res_pkg = pkg.predict(test_X)
    evaluate(res_pkg, test_Y)





