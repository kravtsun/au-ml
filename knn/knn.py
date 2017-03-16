#!/usr/bin/python3
import numpy as np
from functools import partial
from collections import Counter


def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)[:1000, :]
    return m;

def euclidean(a, b):
    # assert (a.shape[0] == b.shape[0])
    return np.linalg.norm(a - b)

def knn(data, k):
    features = data[:, :-1]
    answers = data[:, -1].astype(int).flatten()
    nsamples = data.shape[0]
    assert nsamples == answers.shape[0]
    nfeatures = features.shape[1]
    assert nfeatures + 1 == data.shape[1]

    def test_sample(row):
        cur_feature = row[:-1]
        cur_label = int(row[-1])

        f = partial(euclidean, cur_feature)
        distances = np.apply_along_axis(f, 1, features)
        # assert distances.shape == (nsamples,)

        # can be bad when several points are one.
        # but we can exclude this cases manually.
        best_knn_indexes = distances.argsort()[1:k+1]
        best_knn_answers = list(answers[best_knn_indexes])

        return Counter(best_knn_answers).most_common(1)[0][0]
        # cntr[cur_label] -= 1
        # label = cntr.most_common(1)[0][0]
        # return label
    assumed_labels = np.apply_along_axis(test_sample, 1, data)
    b = np.not_equal(assumed_labels, answers)
    return float(sum(b)) / nsamples

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    data = read_csv(filename)
    # n = features.shape[0]
    # assert n > 1
    # print("accuracy is ", estimate_quality(knn_classifier, *test))
    # print(float(sum(answers) / len(answers)))
    print("LOO is ", knn(data, 5))

