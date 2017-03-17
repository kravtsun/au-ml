#!/usr/bin/python3
import sys
import knn
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

def calculate_distances(data):
    features = data[:, :-1]
    def func(row):
        row_distances = knn.calc_distances_to_feature_vector(features, row)
        return (np.partition(row_distances, 1)[1], max(row_distances))
    pair_distances = np.apply_along_axis(func, axis=1, arr=features)
    return pair_distances

def find_limit_radii(data):
    distances = calculate_distances(data)
    l = np.percentile(distances[:, 0], 95)
    r = max(distances[:, 1])
    return l, r

if __name__ == '__main__':
    data = knn.read_csv(sys.argv[1])
    knn_run = partial(knn.knn, data=data)
    d = {}

    def func(radius):
        knn_filter = partial(knn.knn_filter_distance_by_radius, r=radius)
        loo = knn_run(knn_filter=knn_filter)
        d[radius] = loo
        return loo

    l, r = find_limit_radii(data)
    # r *= 2
    EPS = 1e-1
    while r - l > EPS:
        m1 = l + (r - l) / 3.
        m2 = r - (r - l) / 3.

        if func(m1) < func(m2):
            r = m2
        else:
            l = m1
        print(l, r, d[m1], d[m2])

    print("result = ", func((l+r)/2.))
    print(d)