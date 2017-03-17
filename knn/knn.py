#!/usr/bin/python3
import numpy as np
from functools import partial
from collections import Counter
import argparse


def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m


def knn_filter_distance_by_number(distances, k):
    return distances.argsort()[:k+1]


def knn_filter_distance_by_radius(distances, r):
    res = np.where(distances <= r)
    return res


def calc_distances_to_feature_vector(features, row):
    distances = np.sqrt(np.sum((features - row) ** 2, axis=1))
    return distances


def get_features(data):
    return data[:, :-1]


def precalc_data(data):
    features = get_features(data)
    nsamples = features.shape[0]
    func = partial(calc_distances_to_feature_vector, features)
    distances = np.apply_along_axis(func, 1, features)
    assert distances.shape == (nsamples, nsamples)
    return distances


def knn(data, knn_filter, distances):
    features = get_features(data)
    answers = data[:, -1]
    nsamples = data.shape[0]
    assert nsamples == answers.shape[0]
    nfeatures = features.shape[1]
    assert nfeatures + 1 == data.shape[1]
    assert distances.shape == (nsamples, nsamples)

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


def normalize_data(data):
    features = get_features(data)
    max_features = np.max(features, axis=0)
    nfeatures = data.shape[1] - 1
    assert max_features.shape == (nfeatures,)
    max_features[max_features == 0.0] = 1.0
    features /= max_features
    assert np.max(features) <= 1.0


def main(argv, data=None, distances=None):
    def create_args_parser():
        parser = argparse.ArgumentParser(description="run KNN classifier with given arguments")
        parser.add_argument("-f", dest="filename", required=data is None)
        knn_param_group = parser.add_mutually_exclusive_group(required=True)
        knn_param_group.add_argument("-k", type=int, dest="k")
        knn_param_group.add_argument("-r", type=float, dest="r")
        parser.add_argument("-n", "--normalize", dest="normalize", action="store_true", default=False)
        return parser
    parser = create_args_parser()
    args = parser.parse_args(argv)

    knn_filter = None
    if args.k:
        knn_filter = partial(knn_filter_distance_by_number, k=args.k)
    else:
        knn_filter = partial(knn_filter_distance_by_radius, r=args.r)

    if data is None:
        data = read_csv(args.filename)

    if args.normalize:
        normalize_data(data)

    if distances is None:
        distances = precalc_data(data)

    return knn(data, knn_filter, distances)


if __name__ == '__main__':
    print(main(None))
