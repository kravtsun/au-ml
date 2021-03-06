#!/usr/bin/python
import numpy as np
import argparse
import matplotlib.pyplot as plt

def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m

def threshold_results(feature, labels, filter=True):
    assert len(feature.shape) == 1 or feature.shape[1] == 1
    assert len(feature) == len(labels)
    n = len(labels)
    # feature = np.random.rand(n)
    unique_feature = np.unique(feature.flatten())
    unique_feature = np.hstack((unique_feature[0] - 1, unique_feature, unique_feature[-1] + 1))
    nuniq = len(unique_feature)
    results = np.array([feature >= uf for uf in unique_feature])
    assert results.shape == (nuniq, n)
    row_positives = np.sum(results, axis=1)
    if (filter):
        not_nan_indexes = np.where(row_positives > 0)
        results = results[not_nan_indexes]
        unique_feature = unique_feature[not_nan_indexes]
    return unique_feature, results

def plot_pareto(precision, recall, graph_label="Pareto"):
    assert precision.shape == recall.shape
    assert precision.shape[0] == recall.shape[0]
    n = len(precision)
    def filter_good(i):
        if np.isnan(precision[i]) or np.isnan(recall[i]):
            return False
        bigger_precision = precision > precision[i]
        bigger_recall = recall > recall[i]
        bigger_both = np.logical_and(bigger_precision, bigger_recall)
        return np.sum(bigger_both) == 0
    pp = np.array([(recall[i], precision[i]) for i in range(n) if filter_good(i)])
    plt.figure(graph_label)
    plt.xlim((0.0, 1.0))
    plt.xlabel("recall")
    plt.ylim((0.0, 1.0))
    plt.ylabel("precision")
    plt.plot(recall, precision, 'ob')
    plt.plot(pp[:, 0], pp[:,1], 'or')
    # plt.show()

def plot_roc(fpr, tpr, **kwargs):
    plt.figure("ROC")
    t = np.linspace(0, 1.0, 100)
    plt.plot(t, t, 'b')
    asort = fpr.argsort()
    plt.plot(fpr[asort], tpr[asort], 'og-', markersize=10)

def auc(fpr, tpr):
    n = len(fpr)
    roc = np.hstack((fpr.reshape((n,1)), tpr.reshape(n,1)))
    assert roc.shape == (fpr.shape[0], 2)
    roc = roc[roc[:,0].argsort()]
    return np.trapz(y=roc[:,1], x=roc[:,0])

def accuracy(results, labels):
    return float(np.sum(results == labels)) / len(labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("-k", dest="k", type=int, required=True, help="feature index (starting from end).")
    args = parser.parse_args()
    data = read_csv(args.filename)
    nfeatures = data.shape[1] - 1
    # k = args.k

    labels = data[:, -1]
    def auc_value(k):
        unique_feature, results = threshold_results(feature=data[:, -1 - k], labels=labels)
        precision, recall, tpr, fpr = calculate_precision_recall(results, labels)
        auc(fpr, tpr)
        # plot_pareto(precision, recall)
        # plot_roc(fpr, tpr)
        # plt.show()
        return auc(fpr, tpr)

    feature_labels = None
    with open(args.filename) as f:
        feature_labels = f.readline().split(',')

    # aucs = [(feature_labels[nfeatures - i], auc_value(i)) for i in range(1, nfeatures + 1)]
    aucs = [(nfeatures - i, auc_value(i)) for i in range(1, nfeatures + 1)]

    # best_ten = sorted(aucs, key=lambda p: p[1])[-10:]
    # from operator import itemgetter
    # print list(map(itemgetter(0), best_ten))
    # print best_ten

    plt.plot(aucs, 'o')
    plt.show()
    print aucs