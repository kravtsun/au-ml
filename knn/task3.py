#!/usr/bin/python3
import knn
from functools import partial
import numpy as np
import argparse


def ternary_search(func, l, r, quiet=False):
    def best_of_two(l, r):
        return (l + r) / 2.
    EPS = 1e-3
    while r - l > EPS:
        if not quiet:
            print(l, r, func(best_of_two(l, r)))
        m1 = l + (r - l) / 3.
        m2 = r - (r - l) / 3.

        fm1, fm2 = func(m1), func(m2)
        if fm1 < fm2 + EPS:
            r = m2
        else:
            l = m1

    best_x = best_of_two(l, r)
    best_y = func(best_x)
    return best_x, best_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run KNN classifier with given arguments")
    parser.add_argument("-f", dest="filename", required=True, help="CSV file.")
    parser.add_argument("-l", dest="l", type=float, default=None)
    parser.add_argument("-r", dest="r", type=float, default=None)
    parser.add_argument("-n", "--normalize", dest="normalize", action="store_true", default=False)
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true", default=False)
    args = parser.parse_args()

    csv_data = knn.read_csv(args.filename)
    if args.normalize:
        knn.normalize_data(csv_data)
    distances = knn.precalc_data(csv_data)
    assert distances.shape == (csv_data.shape[0], csv_data.shape[0])

    def func(radius):
        loo = knn.main(["-r", str(radius)], data=csv_data, distances=distances)
        return loo

    l, r = 0, np.max(distances)
    if args.l is not None:
        l = args.l
    if args.r is not None:
        r = args.r

    best_radius, best_loo = ternary_search(func, l, r, args.quiet)
    print(best_radius, best_loo)
