#!/bin/python
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename, max_rows=None):
    m = np.genfromtxt(filename, delimiter=',', dtype=float, skip_header=1, max_rows=max_rows)
    return m

def distance(p1, p2, axis=None):
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=axis))

def print_cluster_distribution(result):
    clusters = len(set(result)) - (-1 in result)
    print clusters, ":"
    for c in range(clusters):
        print c, np.count_nonzero(result == c)
    print -1, np.count_nonzero(result == -1)

def plot_clusters(data, labels):
    clusters = len(set(labels)) - (-1 in labels)
    def plot_points(points, color='k'):
        plt.plot(points[:, 0], points[:, 1], color + 'o', markersize=10)

    for i, color in zip(range(clusters), cycle('rgbcmy')):
        cluster_points = data[labels == i]
        plot_points(data[labels == i], color)
    plot_points(data[labels == -1])
    plt.show()