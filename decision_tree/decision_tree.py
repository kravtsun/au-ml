#!/usr/bin/python
import numpy as np
import argparse
import matplotlib.pyplot as plt

def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m

def calculate_precision_recall(feature, labels):
    assert len(feature.shape) == 1 or feature.shape[1] == 1
    assert len(feature) == len(labels)
    n = len(labels)
    unique_feature = np.unique(feature.flatten())
    unique_feature = np.hstack((unique_feature[0] - 1, unique_feature, unique_feature[-1] + 1))
    nuniq = len(unique_feature)
    results = np.array([feature >= uf for uf in unique_feature])
    assert results.shape == (nuniq, n)

    assert np.sum(np.unique(labels) - np.array([0, 1])) == 0.0
    total_positives = np.sum(labels)
    assert total_positives > 0
    def count_true_positives(row_result):
        return np.sum(np.logical_and(row_result, labels))
    true_positives = np.apply_along_axis(count_true_positives, 1, results).astype(dtype=np.float)
    row_positives = np.sum(results, axis=1)
    precision = true_positives / row_positives
    recall = true_positives / total_positives
    tpr = true_positives / total_positives
    fpr = (row_positives - true_positives) / (n - total_positives)
    not_nan_indexes = np.where(row_positives > 0)
    return unique_feature[not_nan_indexes], precision[not_nan_indexes], \
           recall[not_nan_indexes], tpr[not_nan_indexes], fpr[not_nan_indexes]

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
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.plot(recall, precision, 'ob')
    plt.plot(pp[:, 0], pp[:,1], 'or')
    # plt.show()

def plot_roc(fpr, tpr):
    plt.figure("ROC")
    t = np.linspace(0, 1.0, 100)
    plt.plot(t, t, 'b')
    plt.plot(fpr, tpr, 'og')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("-k", dest="k", type=int, required=True, help="feature index (starting from end).")
    args = parser.parse_args()
    data = read_csv(args.filename)
    uniq_feature, precision, recall, tpr, fpr = calculate_precision_recall(data[:, -1 - args.k], data[:, -1])
    plot_pareto(precision, recall)
    plot_roc(fpr, tpr)
    plt.show()



