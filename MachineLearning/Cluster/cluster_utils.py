# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import matplotlib.pyplot as plt
import numpy as np


def plot_cluster(center, groups, plot_dim=[0, 1]):
    for g in groups.keys():
        data = np.array(groups[g])
        plt.scatter(x=data[:, plot_dim[0]], y=data[:, plot_dim[1]], label=g)
        plt.scatter(x=center[g][plot_dim[0]], y=center[g][plot_dim[1]], label=g, marker='x')
    plt.legend()
    plt.show()