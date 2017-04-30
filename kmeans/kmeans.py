#!/bin/python
import argparse
import numpy as np
from cluster import read_csv, plot_clusters, distance

def kmeans_chosen(data, centers):
    def best_cluster(p):
        distances = [distance(p, c) for c in centers]
        return np.argmin(distances)
    return np.apply_along_axis(best_cluster, 1, data)

def random_point(pmin, pmax):
    return pmin + np.random.random_sample((2,)) * (pmax - pmin)

def kmeans(data, k, EPS=1e-9, iterations=1000, verbose=True):
    pmax = np.max(data, axis=0)
    pmin = np.min(data, axis=0)
    centers = np.array([random_point(pmin, pmax) for c in range(k)])
    eps = np.inf
    it = 0
    while eps > EPS and it < iterations:
        it += 1
        chosen = kmeans_chosen(data, centers)
        while len(set(chosen)) < k:
            for c in range(k):
                if np.count_nonzero(chosen == c) == 0:
                    centers[c] = random_point(pmin, pmax)
            chosen = kmeans_chosen(data, centers)
        new_centers = np.array([np.mean(data[chosen == c, :], axis=0) for c in range(k)])
        eps = distance(centers, new_centers)
        assert(new_centers.shape == centers.shape)
        centers = new_centers
        if verbose:
            print "eps = ", eps
    return chosen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("-k", dest="clusters", type=int, required=True)
    args = parser.parse_args()
    data = read_csv(args.filename)
    result = kmeans(data, args.clusters)
    plot_clusters(data, result)
