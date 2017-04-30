#!/bin/python
import sys
sys.path.append("..")

from common import read_csv, normalize_data, prepare_regression, get_features, get_labels, calculate_precision_recall
import numpy as np
import matplotlib.pyplot as plt
import argparse

def choices_unique(seq, k=1):
    from random import shuffle
    shuffle(seq)
    a = np.array(seq)
    assert a.shape == (n,)
    return a[:k]

def train(features, labels, step, k, iterations=100):
    n, nfeatures = features.shape
    assert k <= n
    sign_labels, w = prepare_regression(features, labels)
    assert w.shape == (nfeatures,)
    it = 0

    best_ein = 1e+50
    best_w = np.zeros((nfeatures,))
    kmul = -1. / k
    while it < iterations:
        cur_labels = np.sign(features.dot(w))
        diff = cur_labels != sign_labels
        ein = np.sum(diff, dtype=float) / n
        if ein < best_ein:
            best_ein = ein
            best_w = w
        # print "it = ", it, "ein = ", ein
        # print ein
        assert cur_labels.shape == (n,)
        choices = choices_unique(range(n), k)
        ii = np.sort(choices)
        xi, wx, yi = features[ii, :], cur_labels[ii], sign_labels[ii]
        assert yi.shape == (k,)
        yitile = np.tile(yi, (nfeatures, 1)).T
        assert yitile.shape == xi.shape
        nominator = yx = xi * yitile
        assert yx.shape == (k, nfeatures)
        ywx = yi * wx
        assert ywx.shape == (k, )
        denominator = 1 + np.exp(ywx)
        assert denominator.shape == (k, )

        denominator = np.tile(denominator, (nfeatures, 1)).T
        assert denominator.shape == (k, nfeatures)
        assert nominator.shape == (k, nfeatures)
        wdelta = kmul * np.sum(nominator / denominator, axis=0)
        assert wdelta.shape == (nfeatures,)
        w = w - step * wdelta
        it += 1
    print "best_ein = ", best_ein
    return best_w

def plot_roc(features, labels, w):
    # labels[np.where(labels == 0)] = -1
    assert list(set(labels)) == [0, 1]
    res = features.dot(w)
    threshold =  np.arange(-1, 1, 0.01)
    n = len(labels)
    def calc_metrics(t):
        rest = res > t
        acc = np.sum(rest != labels, dtype=float) / n
        precision, recall, tpr, fpr = calculate_precision_recall(rest, labels)
        return tpr, fpr, acc

    metrics = np.array([calc_metrics(t) for t in threshold])
    plt.figure("threshold")
    plt.plot(threshold, metrics[:,2], "o", label="threshold")
    plt.legend()

    plt.figure("ROC")
    plt.plot(metrics[:,1], metrics[:,0], label="ROC")
    plt.legend()
    plt.show()
    print "best accuracy is ", np.min(metrics[:, 2]), "at threshold = ", threshold[np.argmin(metrics[:,2])]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stochastic linear regression.")
    parser.add_argument("--train", dest="train", required=True, help="Train filename")
    parser.add_argument("--test", dest="test", required=True, help="Test filename")

    args = parser.parse_args()
    train_data = read_csv(args.train)
    # normalize_data(train_data)

    train_features, train_labels = get_features(train_data), get_labels(train_data)
    n = train_labels.shape[0]
    w = train(train_features, train_labels, k=10, step=0.1, iterations=1000)
    # plot_roc(train_features, train_labels, w)

    test_data = read_csv(args.test)
    normalize_data(test_data)
    test_features, test_labels = get_features(test_data), get_labels(test_data)
    plot_roc(test_features, test_labels, w)
