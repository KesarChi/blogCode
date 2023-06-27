# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from cluster_utils import *
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans


class Kmean_Kernel:
    def __init__(self, k, iteration=100):
        self.k = k
        self.Iteration = iteration
        self.centers = {i: None for i in range(k)}
        self.Groups = {i: [] for i in range(k)}

    def get_results(self):
        print("Cluster Centers: {}".format(self.centers))
        return self.centers, self.Groups

    def fit(self, X):
        self._init_centers(X)
        for t in range(self.Iteration):
            flag = self._step(X)
            if not flag:
                break

    def _init_centers(self, X):
        pass

    def _step(self, X):
        pass

    def _update_centers(self, group):
        pass


class KmeansSimple(Kmean_Kernel):
    def _init_centers(self, X):
        idxs = np.random.randint(0, X.shape[0], self.k)
        for i in range(self.k):
            self.centers[i] = X[idxs[i]]

    def _step(self, X):
        groups = {i: [] for i in range(self.k)}
        centers = np.array([self.centers[key] for key in sorted(self.centers.keys())])
        for i in range(X.shape[0]):
            tmp = np.tile(X[i], (self.k, 1))
            dists = np.sum((tmp - centers)**2, axis=1)
            groups[np.argmin(dists)].append(X[i])
        self.Groups = groups
        change = self._update_centers(groups)
        return change

    def _update_centers(self, groups):
        flag = False
        new_center = {i: [] for i in range(self.k)}
        for g in range(self.k):
            tmp = groups[g]
            nc = np.mean(tmp, axis=0)
            new_center[g] = nc
            if not all([c for c in nc] == self.centers[g]):
                flag = True
        self.centers = new_center
        return flag


if __name__ == "__main__":
    X, y = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1)

    model = KmeansSimple(k=2)
    model.fit(X)
    centers, group = model.get_results()
    plot_cluster(centers, group)

    pkg = KMeans(n_clusters=2)
    pkg.fit(X)
    print(pkg.cluster_centers_)




