#!/bin/python
import sys
import argparse
import numpy as np
from collections import Counter
from common import read_csv, plot_clusters, distance, print_cluster_distribution
from sklearn.decomposition import PCA
from kmeans import kmeans
from affinity_propagation import affinity_propagation

def pca2(data):
    return PCA(n_components=2).fit(data).transform(data)


def number_to_cluster(labels, result):
    assert len(labels) == len(result)
    res = [0] * 10
    right = 0
    all = 0
    print "mapping : "
    for i in range(10):
        indexes = np.where(labels == i)[0]
        clustered = result[indexes]
        res[i] = Counter(clustered).most_common(1)[0][0]
        chosen_right = np.count_nonzero(result[indexes] == res[i])
        print i, res[i], chosen_right / float(len(clustered))
        right += chosen_right
        all += len(clustered)
    assert all == len(labels)
    print "accuracy: ", float(right) / all
    return res

def cluster_to_number(labels, result):
    assert len(labels) == len(result)
    res = []
    for i in range(10):
        indexes = np.where(result == i)
        res.append(list(Counter(labels[indexes]).items()))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    method_param_group = parser.add_mutually_exclusive_group(required=True)
    method_param_group.add_argument("--af", action='store_true', default=False)
    method_param_group.add_argument("--kmeans", action='store_true', default=False)
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("-v", "--verbose", dest="verbose", action='store_true', default=False)
    parser.add_argument("-k", dest="clusters", type=int, required=False, default=10)
    parser.add_argument("-p", "--pref", dest="pref", type=int, required=False, default=-15000)
    parser.add_argument("-w", dest="weight", type=float, required=False, default=0.75)

    args = parser.parse_args()
    data = read_csv(args.filename)
    labels = data[:,0]
    features = data[:, 1:]
    data2 = pca2(features)
    if args.af:
        result = affinity_propagation(data2,pref=args.weight, w=args.weight, verbose=args.verbose)
    elif args.kmeans:
        result = kmeans(data2, args.clusters, verbose=args.verbose)
    else:
        assert False

    if args.verbose:
        plot_clusters(data2, result)
        print_cluster_distribution(result)

    num_to_cluster = number_to_cluster(labels, result)
    cluster_to_num = cluster_to_number(labels, result)
    for i, c in enumerate(cluster_to_num):
        print i, c
    print cluster_to_num
