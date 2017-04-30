#!/usr/bin/python
import numpy as np
from functools import partial
from collections import Counter
import argparse
from sklearn.neighbors import KDTree
from common import read_csv, get_features, calc_distances_to_feature_vector


def knn_filter_distance_by_number(distances, k):
    return distances.argsort()[:k+1]


def knn_filter_distance_by_radius(distances, r):
    res = np.where(distances <= r)
    return res


def precalc_data(data):
    features = get_features(data)
    nsamples = features.shape[0]
    func = partial(calc_distances_to_feature_vector, features)
    distances = np.apply_along_axis(func, 1, features)
    assert distances.shape == (nsamples, nsamples)
    return distances


def knn_with_kdtree(data, k):
    features = get_features(data)
    answers = data[:, -1]
    nsamples = data.shape[0]
    assert nsamples == answers.shape[0]
    nfeatures = features.shape[1]
    assert nfeatures + 1 == data.shape[1]

    most_common_label = Counter(answers).most_common(1)[0][0]
    tree = KDTree(features, leaf_size=2)
    def test_sample(row_data):
        assert row_data.shape == (nfeatures + 1,)
        row_label = row_data[-1]
        row_features = row_data[:-1]
        best_knn_distances, best_knn_indexes = tree.query(row_features.reshape(1, -1), k=k+1)
        best_knn_answers = answers[best_knn_indexes.flatten()]
        counter = Counter(list(best_knn_answers))
        counter.subtract([row_label])
        most_common = counter.most_common(1)
        if len(most_common) == 0 or most_common[0][1] == 0:
            # can't classify: no elements in the neighborhood.
            return most_common_label
        else:
            return most_common[0][0]

    assumed_labels = np.array([test_sample(row_data) for row_data in data])
    assert assumed_labels.shape == (nsamples,)
    b = np.not_equal(assumed_labels, answers)
    return float(sum(b)) / nsamples


def knn(data, knn_filter, distances=None, kdtree=None):
    features = get_features(data)
    answers = data[:, -1]
    nsamples = data.shape[0]
    assert nsamples == answers.shape[0]
    nfeatures = features.shape[1]
    assert nfeatures + 1 == data.shape[1]

    most_common_label = Counter(answers).most_common(1)[0][0]

    def test_sample(row_data, row_distances):
        assert row_data.shape == (nfeatures + 1,)
        assert row_distances.shape == (nsamples, )
        row_label = row_data[-1]
        best_knn_indexes = knn_filter(row_distances)
        best_knn_answers = answers[best_knn_indexes]
        counter = Counter(list(best_knn_answers))
        counter.subtract([row_label])
        most_common = counter.most_common(1)
        if len(most_common) == 0 or most_common[0][1] == 0:
            # can't classify: no elements in the neighborhood.
            return most_common_label
        else:
            return most_common[0][0]
    assumed_labels = np.array([ test_sample(row_data, row_distances) for row_data, row_distances in zip(data, distances) ])
    assert assumed_labels.shape == (nsamples,)
    b = np.not_equal(assumed_labels, answers)
    return float(sum(b)) / nsamples


def main(argv, data=None, distances=None):
    def create_args_parser():
        parser = argparse.ArgumentParser(description="run KNN classifier with given arguments")
        parser.add_argument("-f", dest="filename", required=data is None)
        parser.add_argument("--kdtree", dest="kdtree", action='store_true', required=False, default=False)
        knn_param_group = parser.add_mutually_exclusive_group(required=True)
        knn_param_group.add_argument("-k", type=int, dest="k")
        knn_param_group.add_argument("-r", type=float, dest="r")
        parser.add_argument("-n", "--normalize", dest="normalize", action="store_true", default=False)
        return parser
    parser = create_args_parser()
    args = parser.parse_args(argv)

    if data is None:
        data = read_csv(args.filename)
    if args.normalize:
        normalize_data(data)

    if args.kdtree:
        assert args.k is not None
        return knn_with_kdtree(data, args.k)

    knn_filter = None
    if args.k:
        knn_filter = partial(knn_filter_distance_by_number, k=args.k)
    else:
        knn_filter = partial(knn_filter_distance_by_radius, r=args.r)

    if distances is None:
        distances = precalc_data(data)
    return knn(data, knn_filter, distances, kdtree=args.kdtree)


if __name__ == '__main__':
    print(main(None))
