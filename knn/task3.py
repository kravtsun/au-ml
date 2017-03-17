#!/usr/bin/python3
import knn
from functools import partial
import numpy as np
import argparse


def calculate_distances(data):
    features = data[:, :-1]

    def func(row):
        row_distances = knn.calc_distances_to_feature_vector(features, row)
        return np.partition(row_distances, 1)[1], max(row_distances)
    pair_distances = np.apply_along_axis(func, axis=1, arr=features)
    return pair_distances


def find_limit_radii(data):
    distances = calculate_distances(data)
    return np.percentile(distances[:, 0], 95), max(distances[:, 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run KNN classifier with given arguments")
    parser.add_argument("-f", dest="filename", required=True)
    parser.add_argument("-l", dest="l", type=float, default=None)
    parser.add_argument("-r", dest="r", type=float, default=None)
    parser.add_argument("-n", "--normalize", dest="normalize", action="store_true", default=False)
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true", default=False)
    args = parser.parse_args()

    csv_data = knn.read_csv(args.filename)
    if args.normalize:
        knn.normalize_data(csv_data)
    knn_run = partial(knn.knn, data=csv_data)

    def func(radius):
        knn_filter = partial(knn.knn_filter_distance_by_radius, r=radius)
        loo = knn_run(knn_filter=knn_filter)
        return loo

    l, r = find_limit_radii(csv_data)
    if args.l is not None:
        l = args.l
    if args.r is not None:
        r = args.r

    EPS = 1e-1
    while r - l > EPS:
        m1 = l + (r - l) / 3.
        m2 = r - (r - l) / 3.

        if func(m1) < func(m2):
            r = m2
        else:
            l = m1
        if not args.quiet:
            print(l, r)

    best_radius = (l + r) / 2.
    best_loo = func(best_radius)
    print(best_loo)
