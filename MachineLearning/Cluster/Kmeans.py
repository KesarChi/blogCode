# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from cluster_utils import *
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans


class Kmeans:
    def __init__(self, k, iteration=100):
        self.k = k
        self.Iteration = iteration
        self.centers = {i: None for i in range(k)}
        self.Groups = {i: [] for i in range(k)}

    def get_results(self):
        # print("Cluster Centers: {}".format(self.centers))
        return self.centers, self.Groups

    def fit(self, X):
        self._init_centers(X)
        for t in range(self.Iteration):
            flag = self._step(X)
            if not flag:
                break

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
            # if not all([c for c in nc] == self.centers[g]):
            #     flag = True
            # print(nc)
            # if np.sum([abs(nc[i] - self.centers[g][i]) for i in range(len(nc))]) >= 0.05 * np.sum(self.centers[g]):
            #     flag = True
        self.centers = new_center
        return flag


def _calc_totalDist(center, group):
    idx = sorted(center.keys())
    centers = np.array([np.array(center[idx[i]]) for i in range(len(idx))])
    groups = np.array([np.array(group[idx[i]]) for i in range(len(idx))])
    dist = 0
    for i in range(len(idx)):
        dist += np.sum((np.tile(centers[i], (groups[i].shape[0], 1)) - group[i]) ** 2)
    return dist


class KmeansAutoK(Kmeans):
    def __init__(self, lower_bound, upper_bound, eps=2, iteration=100):
        self.k = 0
        self.low = lower_bound
        self.up = upper_bound
        self.eps = eps
        self.Iteration = iteration
        self.centers = {i: None for i in range(self.k)}
        self.Groups = {i: [] for i in range(self.k)}

    def fit(self, X):
        self._test_K(X)
        self._init_centers(X)
        for t in range(self.Iteration):
            flag = self._step(X)
            if not flag:
                break

    def _test_K(self, X, mode='inf'):
        if mode == 'inf':
            self._inflection_k(X)
        elif mode == 'gap':
            self._gap_k(X)

    def _inflection_k(self, X):
        totalDist, diff = [], []
        i = 1
        for k in range(self.low, self.up+1):
            km = Kmeans(k)
            km.fit(X)
            center, group = km.get_results()
            totalDist.append(_calc_totalDist(center, group))
            if len(totalDist) > 1:
                diff.append(totalDist[i-1]-totalDist[i])
                if len(diff) > 1 and (diff[i-1]*self.eps < diff[i-2]):
                    self.k = k-1
                    return
                i += 1
        self.k = np.argmin(totalDist)+1

    def _gap_k(self, X):
        sample_mean, sample_std = np.mean(X, axis=0), np.std(X, axis=0)
        DK = {}
        for t in range(20):
            data = np.empty((X.shape[0], X.shape[1]))
            for i in range(len(sample_std)):
                m, s = sample_mean[i], sample_std[i]
                tmp = np.random.normal(m, s, X.shape[0])
                data[:, i] = tmp
            for k in range(self.low, self.up):
                km = Kmeans(k)
                km.fit(data)
                centers, groups = km.get_results()
                DK[k].append(np.log(_calc_totalDist(centers, groups)))

        return


if __name__ == "__main__":
    X, y = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0, n_classes=4, n_clusters_per_class=1)

    # model = Kmeans(k=4)
    # model.fit(X)
    # centers, group = model.get_results()
    # plot_cluster(centers, group)
    #
    # pkg = KMeans(n_clusters=4)
    # pkg.fit(X)
    # print(pkg.cluster_centers_)

    # 标准化数据最佳eps分析
    # times = 50
    # for K in range(2, 5):
    #     cnt = 0
    #     for t in range(times):
    #         X, y = make_classification(n_samples=2000, n_features=4, n_informative=4, n_redundant=0, n_classes=K, n_clusters_per_class=1)
    #         model = KmeansAutoK(lower_bound=2, upper_bound=8, eps=1.8)
    #         model.fit(X)
    #         if model.k == K: cnt += 1
    #     print("K={} {}/{}".format(K, cnt, times))






