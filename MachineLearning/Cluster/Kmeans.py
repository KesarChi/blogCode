# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import numpy as np
from sklearn.datasets import make_classification


class KMeans:
    def __init__(self, k, iteration=1000):
        self.k = k
        self.Iteration = iteration
        self.centers = {i: None for i in range(k)}
        self.Groups = {i: [] for i in range(k)}

    def fit(self, X):
        self.centers = self._init_centers(X)
        for t in range(self.Iteration):
            flag = self._run(X)
            if not flag:
                break

    def get_results(self):
        print("Cluster Centers: {}".format(self.centers))
        for i in range(self.k):
            print("Cluster{} Data: {}\n".format(i, self.Groups[i]))
        print("Clustered Data: {}")
        return self.centers, self.Groups

    def _init_centers(self, X):
        idxs = np.random.randint(0, X.shape[0], self.k)
        return X[idxs]

    def _run(self, X):
        change = True
        groups = {i: [] for i in range(self.k)}
        centers = np.array([self.centers[key] for key in sorted(self.centers.keys())])
        for i in range(X.shape[0]):
            tmp = np.tile(X[i], (self.k, 1))
            dists = np.sum((tmp - centers)**2, axis=1)
            groups[np.argmin(dists)].append(X[i])
        change = self._update_centers(groups)
        return change

    def _update_centers(self, groups):
        flag = False
        new_center = {i:[] for i in range(self.k)}
        for g in range(self.k):
            tmp = groups[g]
            nc = np.mean(tmp, axis=0)
            new_center[g] = nc
            if not all([int(c) for c in nc] == self.centers[g]):
                flag = True
        return flag




