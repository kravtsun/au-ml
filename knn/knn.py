#!/usr/bin/python3
import numpy as np
from functools import partial
# from collections import Counter
import argparse

def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m;

def euclidean(a, b):
    # assert (a.shape[0] == b.shape[0])
    return np.linalg.norm(a - b)

def knn_filter_distance_by_number(distances, k):
    return distances.argsort()[1:k+1]

def knn_filter_distance_by_radius(distances, r):
    assert r > 0
    close_distances = distances[distances < r]
    return close_distances.argsort()

def knn(data, knn_filter):
    features = data[:, :-1]
    answers = data[:, -1].astype(int).flatten()
    nsamples = data.shape[0]
    assert nsamples == answers.shape[0]
    nfeatures = features.shape[1]
    assert nfeatures + 1 == data.shape[1]

    def test_sample(row):
        # distances = np.apply_along_axis(euclidean, 1, features, b=cur_feature)
        # calculate euclidian distances between samples and current feature vector.
        distances = np.sqrt(np.sum((features - row) ** 2, axis=1))

        assert distances.shape == (nsamples,)

        # can be bad when several points are one. but we can exclude this cases manually.
        best_knn_indexes = knn_filter(distances)
        best_knn_answers = answers[best_knn_indexes]

        # requires integer labels starting from 0. to be replaced with np.unique otherwise.
        return np.bincount(best_knn_answers).argmax() # as k is relatively small, this optimization has no future.

    assumed_labels = np.apply_along_axis(test_sample, 1, features)
    b = np.not_equal(assumed_labels, answers)
    return float(sum(b)) / nsamples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run KNN classifier with given arguments")
    parser.add_argument("-f", dest="filename", required=True)
    knn_param_group = parser.add_mutually_exclusive_group(required=True)
    knn_param_group.add_argument("-k", type=int, dest="k")
    knn_param_group.add_argument("-r", type=float, dest="r")
    args = parser.parse_args()

    knn_filter = None
    if args.k:
        knn_filter = partial(knn_filter_distance_by_number, k=args.k)
    else:
        knn_filter = partial(knn_filter_distance_by_radius, r=args.r)

    data = read_csv(args.filename)
    print(knn(data, knn_filter))
