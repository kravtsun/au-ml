#!/usr/bin/python
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import Counter

def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m

def threshold_results(feature, labels):
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
    not_nan_indexes = np.where(row_positives > 0)
    results = results[not_nan_indexes]
    unique_feature = unique_feature[not_nan_indexes]
    return unique_feature, results

def calculate_precision_recall(feature, labels):
    n = len(labels)
    unique_feature, results = threshold_results(feature, labels)
    row_positives = np.sum(results, axis=1)
    assert np.all(row_positives > 0)
    def count_true_positives(row_result):
        return np.sum(np.logical_and(row_result, labels))
    true_positives = np.apply_along_axis(count_true_positives, 1, results).astype(dtype=np.float)
    true_positives = true_positives
    precision = true_positives / row_positives

    assert np.sum(np.unique(labels) - np.array([0, 1])) == 0.0
    total_positives = np.sum(labels)
    assert total_positives > 0
    recall = true_positives / total_positives
    tpr = true_positives / total_positives
    fpr = (row_positives - true_positives) / (n - total_positives)

    return unique_feature, precision, recall, tpr, fpr

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

def plot_roc(fpr, tpr):
    plt.figure("ROC")
    t = np.linspace(0, 1.0, 100)
    plt.plot(t, t, 'b')
    plt.plot(fpr, tpr, 'og')

def auc(fpr, tpr):
    n = len(fpr)
    roc = np.hstack((fpr.reshape((n,1)), tpr.reshape(n,1)))
    assert roc.shape == (fpr.shape[0], 2)
    roc = roc[roc[:,0].argsort()]
    return np.trapz(y=roc[:,1], x=roc[:,0])


def entropy(labels):
    assert len(labels.shape) == 1
    n = len(labels)
    assert n > 0
    a = np.sum(labels == 0, dtype=float)
    b = n - a
    INF = 1e50
    an = a / n
    bn = b / n
    log2an = -INF if a == 0 else np.log2(an)
    log2bn = -INF if b == 0 else np.log2(bn)
    return - an * log2an - bn * log2bn

def gain_criteria(result, labels):
    n = len(labels)
    indexes = np.where(result == 1)
    other_indexes = np.where(result == 0)
    r1, r2 = labels[indexes], labels[other_indexes]
    lenr1, lenr2 = len(r1), len(r2)
    if lenr1 == n or lenr2 == n:
        return -np.inf
    nfloat = float(n)
    return entropy(labels) - lenr1 / nfloat * entropy(r1) - lenr2 / nfloat * entropy(r2)

def gini_criteria(result, labels):
    n = len(labels)
    indexes = np.where(result == 1)
    other_indexes = np.where(result == 0)
    r1, r2 = labels[indexes], labels[other_indexes]
    lenr1, lenr2 = len(r1), len(r2)
    nfloat = float(n)
    return 2 * (lenr1 / nfloat) * (lenr2 / nfloat)

class DecisionTree:
    class Node:
        def __init__(self, ifeature, threshold, label):
            self.left_ = None
            self.right_ = None
            # self.indexes = indexes
            self.ifeature = ifeature
            self.threshold = threshold
            self.label = label

        def predict(self, feature):
            if self.ifeature is None:
                return self.label
            else:
                if feature[self.ifeature] > self.threshold:
                    if self.right_ is not None:
                        return self.right_.predict(feature)
                    else:
                        return self.label
                else:
                    if self.left_ is not None:
                        return self.left_.predict(feature)
                    else:
                        return self.label

    def __init__(self, features, labels, criteria, depth, verbose=False):
        assert len(labels.shape) == 1
        n = len(features)
        assert n == len(labels)
        self.verbose = verbose
        self.criteria = criteria
        self.max_depth = depth
        self.root = self.create_node(features, labels, range(n), 0)

    def create_node(self, features, labels, indexes, level):
        def debug(msg, level):
            if self.verbose:
                print (" " * (level * 4)) + str(msg)
        if len(indexes) == 0:
            return None
        curfeatures = features[indexes]
        curlabels = labels[indexes]
        curbestlabel = Counter(curlabels).most_common(1)[0][0]
        n = len(curlabels)
        nf = curfeatures.shape[1]

        def calc_criteria(feature):
            uniq_feature, results = threshold_results(feature, curlabels)
            nuniq = len(uniq_feature)
            assert results.shape == (nuniq, n)
            curcriteria = [(uniq_feature[i], results[i,:], self.criteria(results[i,:], curlabels)) for i in range(nuniq)]
            return max(curcriteria, key=lambda p : p[-1])
        calculated_criteria = [calc_criteria(curfeatures[:, i]) for i in range(nf)]
        ifeature, (threshold, result, best_gain) = max(zip(range(nf), calculated_criteria), key=lambda p:p[1][-1])

        if best_gain == -np.inf or level == self.max_depth - 1:
            debug("<node>", level)
            root = DecisionTree.Node(None, threshold, curbestlabel)
            debug("</node>", level)
            return root

        debug("<node>", level)
        debug((ifeature, threshold, best_gain), level)
        root = DecisionTree.Node(ifeature, threshold, curbestlabel)
        left_indexes = np.where(result == 0)[0]
        debug("<left>", level)
        root.left_ = self.create_node(curfeatures, curlabels, left_indexes, level + 1)
        debug("</left>", level)

        right_indexes = np.where(result == 1)[0]
        assert len(left_indexes) + len(right_indexes) == n
        debug("<right>", level)
        root.right_ = self.create_node(curfeatures, curlabels, right_indexes, level + 1)
        debug("</right>", level)
        debug("</node>", level)
        return root

    def predict(self, feature):
        return self.root.predict(feature)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decision tree classification.")
    parser.add_argument("--train", dest="train", type=str, required=True)
    parser.add_argument("--test", dest="test", type=str, required=True)
    parser.add_argument("--depth", dest="depth", type=int, required=False, default=10)
    parser.add_argument("--verbose", dest="verbose", action='store_true', required=False, default=False)
    parser.add_argument("--g", dest="verbose", action='store_true', required=False, default=False)

    args = parser.parse_args()
    data = read_csv(args.train)
    features = data[:, :-1]
    labels = data[:, -1]
    tree = DecisionTree(features, labels, gain_criteria, args.depth, args.verbose)

    test_data = read_csv(args.test)
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]
    result = np.apply_along_axis(tree.predict, 1, test_features)
    print args.depth, np.sum(result == test_labels) / float(len(test_labels))
