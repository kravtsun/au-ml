#!/bin/python
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def kmeans_chosen(data, centers):
    def best_cluster(p):
        distances = [distance(p, c) for c in centers]
        return np.argmin(distances)
    return np.apply_along_axis(best_cluster, 1, data)
    # return [best_cluster(p) for p in data]

def kmeans(k, data, EPS=1e-9, iterations=1000):
    pmax = np.max(data, axis=0)
    pmin = np.min(data, axis=0)
    centers = pmin + np.random.random_sample((k, 2)) * (pmax - pmin)
    eps = np.inf
    it = 0
    while eps > EPS and it < iterations:
        it += 1
        chosen = kmeans_chosen(data, centers)
        new_centers = np.array([np.mean(data[chosen == c, :], axis=0) for c in range(k)])
        eps = distance(centers, new_centers)
        assert(new_centers.shape == centers.shape)
        centers = new_centers
        print "eps = ", eps
    # return np.append(data, chosen)
    # return np.hstack((data, chosen))
    return chosen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("-k", dest="clusters", type=int, required=True)
    args = parser.parse_args()
    data = read_csv(args.filename)
    result = kmeans(args.clusters, data)
    colors = ['r', 'g', 'b', 'y']
    for i in range(args.clusters):
        # where = result[2, :] == i
        cluster_points = data[result == i]
        plt.plot(cluster_points[:,0], cluster_points[:,1], colors[i]+'o', markersize=10)
        cluster_center = np.mean(cluster_points, axis=0)
        plt.plot(cluster_center[0], cluster_center[1], colors[i] + 'o', markersize=20)
    plt.show()


