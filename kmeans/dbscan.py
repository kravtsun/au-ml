#!/bin/python
import argparse
import numpy as np
from cluster import read_csv, plot_clusters, distance, print_cluster_distribution

def kmeans_chosen(data, centers):
    def best_cluster(p):
        distances = [distance(p, c) for c in centers]
        return np.argmin(distances)
    return np.apply_along_axis(best_cluster, 1, data)
    # return [best_cluster(p) for p in data]

def dbscan(data, eps, m):
    n = data.shape[0]
    cur_cluster = 0
    worked = [False] * n
    marked = [-1] * n
    def propagate(i, cur_cluster):
        if worked[i]:
            return []
        worked[i] = True
        directly_reachable = [j for j in range(n) if i != j and distance(data[i, :], data[j, :]) < eps]
        if len(directly_reachable) < m:
            return []
        marked[i] = cur_cluster
        next_work = filter(lambda j: marked[j] == -1, directly_reachable)
        for j in filter(lambda j: marked[j] != cur_cluster, directly_reachable):
            marked[j] = cur_cluster
        return next_work

    for i in range(n):
        if worked[i]: continue
        next_work = [i]
        while len(next_work) > 0:
            next_next_work = []
            for j in next_work:
                next_next_work += propagate(j, cur_cluster)
            next_work = next_next_work
        if marked[i] != -1:
            cur_cluster += 1
    result = np.array(marked)
    assert result.shape == (n,)
    return result

def sklearn_bruteforce(data, clusters):
    from sklearn import cluster
    for m in range(1, 21):
        for eps in np.arange(0.01, 0.5, 0.01):
            core_samples, result = cluster.dbscan(data, min_samples=m, eps=eps)
            if len(set(result)) == clusters+1:
                print m, eps, np.count_nonzero(result == 0), np.count_nonzero(result == -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("-e", dest="eps", type=float, required=True)
    parser.add_argument("-m", dest="m", type=int, default=10)

    args = parser.parse_args()
    data = read_csv(args.filename)

    result = dbscan(data, args.eps, args.m)
    print_cluster_distribution(result)
    plot_clusters(data, result)



