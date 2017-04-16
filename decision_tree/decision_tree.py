#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter
from dtree import read_csv, threshold_results, accuracy, plot_roc, calculate_precision_recall


INF = 1e50

def entropy(labels):
    assert len(labels.shape) == 1
    n = len(labels)
    assert n > 0
    a = np.sum(labels == 0, dtype=float)
    b = n - a
    an = a / n
    bn = b / n
    log2an = -INF if a == 0 else np.log2(an)
    log2bn = -INF if b == 0 else np.log2(bn)
    return - an * log2an - bn * log2bn

def impurity(labels):
    n = len(labels)
    assert n > 0
    # r1, r2 = labels[np.where(labels == 0)], labels[np.where(labels == 1)]
    # lenr1, lenr2 = len(r1), len(r2)
    lenr1 = np.count_nonzero(labels == 0)
    lenr2 = n - lenr1
    return 2 * lenr1 * lenr2 / float(n)

def gain_criteria(result, labels):
    n = len(labels)
    indexes, other_indexes = np.where(result == 0), np.where(result == 1)
    r1, r2 = labels[indexes], labels[other_indexes]
    lenr1, lenr2 = len(r1), len(r2)
    if lenr1 == n or lenr2 == n:
        return -np.inf
    nfloat = float(n)
    return entropy(labels) - lenr1 / nfloat * entropy(r1) - lenr2 / nfloat * entropy(r2)

def gini_criteria(result, labels):
    n = len(labels)
    indexes, other_indexes = np.where(result == 0), np.where(result == 1)
    r1, r2 = labels[indexes], labels[other_indexes]
    lenr1, lenr2 = len(r1), len(r2)
    nfloat = float(n)
    res = 0
    if lenr1 > 0:
        res -= impurity(r1)
    if lenr2 > 0:
        res -= impurity(r2)
    return res

    # if lenr1 == n or lenr2 == n:
    #     return -np.inf
    # nfloat = float(n)
    # return impurity(labels) - lenr1 / nfloat * impurity(r1) - lenr2 / nfloat * impurity(r2)

class DecisionTree:
    max_level = 0
    class Node:
        def __init__(self, ifeature, threshold, label, confidence):
            self.left_ = None
            self.right_ = None
            # self.indexes = indexes
            self.ifeature = ifeature
            self.threshold = threshold
            self.label = label
            self.confidence = confidence
            pass

        def predict(self, feature):
            if self.ifeature is not None:
                if feature[self.ifeature] > self.threshold and self.right_ is not None:
                    return self.right_.predict(feature)
                elif feature[self.ifeature] <= self.threshold and self.left_ is not None:
                    return self.left_.predict(feature)
            return self.label, self.confidence

    def __init__(self, features, labels, criteria, depth=100, verbose=False):
        assert len(labels.shape) == 1
        n = len(features)
        assert n == len(labels)
        self.verbose = verbose
        self.criteria = criteria
        self.max_depth = depth
        self.min_samples_leaf = 1
        self.root = self.create_node(features, labels, range(n), 0)

    def debug(self, msg, level):
        if self.verbose:
            print (" " * (level * 4)) + str(msg)

    def create_leaf(self, curbestlabel, level, confidence):
        self.debug("<node>", level)
        root = DecisionTree.Node(None, None, curbestlabel, confidence)
        self.debug("</node>", level)
        return root

    def create_node(self, features, labels, indexes, level):
        DecisionTree.max_level = max(DecisionTree.max_level, level)
        if len(indexes) < self.min_samples_leaf:
            return None
        curfeatures, curlabels = features[indexes], labels[indexes]
        curbestlabel, curbestlen = Counter(curlabels).most_common(1)[0]
        n, nf = len(curlabels), curfeatures.shape[1]
        curconfidence = accuracy(curlabels, [1.0] * n)
        if curbestlen == len(curlabels):
            return self.create_leaf(curbestlabel, level, curconfidence)

        if self.max_depth is not None and level + 1 == self.max_depth:
            return self.create_leaf(curbestlabel, level, curconfidence)
        def calc_criteria(feature):
            uniq_feature, results = threshold_results(feature, curlabels, False)
            nuniq = len(uniq_feature)
            assert results.shape == (nuniq, n)
            curcriteria = [(uniq_feature[i], results[i,:],\
                            self.criteria(results[i,:], curlabels)) for i in range(nuniq)]
                            # accuracy(results[i, :], curlabels)) \
            return max(curcriteria, key=lambda p : p[-1])
        calculated_criteria = [(i,) + calc_criteria(curfeatures[:, i]) for i in range(nf)]
        ifeature, threshold, result, best_gain = max(calculated_criteria, key=lambda p: p[-1:])
        left_indexes, right_indexes = np.where(result == 0)[0], np.where(result == 1)[0]
        left_len, right_len = len(left_indexes), len(right_indexes)
        assert left_len + right_len == n
        self.debug((ifeature, threshold, (left_len, right_len)), level)

        if best_gain == -np.inf:# or (self.max_depth is not None and level == self.max_depth - 1):
            return self.create_leaf(curbestlabel, level, curconfidence)

        self.debug("<node>", level)
        root = DecisionTree.Node(ifeature, threshold, curbestlabel, curconfidence)
        self.debug("<left>", level)
        root.left_ = self.create_node(curfeatures, curlabels, left_indexes, level + 1)
        self.debug("</left>", level)

        self.debug("<right>", level)
        root.right_ = self.create_node(curfeatures, curlabels, right_indexes, level + 1)
        self.debug("</right>", level)
        self.debug("</node>", level)
        return root

    def predict(self, feature):
        return self.root.predict(feature)

def test_sklearn(train_features, train_labels, test_features, test_labels):
    from sklearn import tree
    t = tree.DecisionTreeClassifier(criterion="entropy")
    t.fit(train_features, train_labels)
    test_results = np.array([t.predict(test_features[i,:].reshape(1, -1)) for i in range(len(test_labels))]).flatten()
    print "sklearn: ", accuracy(test_results, test_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decision tree classification.")
    parser.add_argument("--train", dest="train", type=str, required=True)
    parser.add_argument("--test", dest="test", type=str, required=True)
    parser.add_argument("--depth", dest="depth", type=int, required=False, default=None)
    parser.add_argument("--verbose", dest="verbose", action='store_true', required=False, default=False)
    parser.add_argument("--gini", dest="gini", action='store_true', required=False, default=False)
    parser.add_argument("--best", dest="best", action='store_true', required=False, default=False)
    parser.add_argument("--roc", dest="roc", action='store_true', required=False, default=False)
    parser.add_argument("--sklearn", dest="sklearn", action='store_true', required=False, default=False)

    args = parser.parse_args()
    data = read_csv(args.train)
    features = data[:, :-1]
    labels = data[:, -1]

    test_data = read_csv(args.test)
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]
    best_feature_indexes = np.array([6, 18, 4, 15, 56, 52, 20, 54, 55, 51])
    if args.best:
        features = features[:, best_feature_indexes]
        test_features = test_features[:, best_feature_indexes]
    result = None
    if args.sklearn:
        from sklearn import tree
        criteria = "gini" if args.gini else "entropy"
        t = tree.DecisionTreeClassifier(criterion=criteria, max_depth=args.depth, splitter="best")
        t.fit(features, labels)
        result = np.array([t.predict(test_features[i, :].reshape(1, -1)) for i in range(len(test_labels))]).flatten()
        assert result.shape == test_labels.shape
    else:
        criteria = gini_criteria if args.gini else gain_criteria
        tree = DecisionTree(features, labels, criteria, args.depth, args.verbose)
        result = np.apply_along_axis(tree.predict, 1, test_features)
        confidence = result[:, 1]
        result = result[:, 0]
        if args.roc:
            uniq_conf = set(confidence)
            uniq_conf = uniq_conf.union(set([0.0, 1.0]))
            print "Confidence: ", Counter(confidence)
            tmp = np.array([calculate_precision_recall(confidence > conf, test_labels) for conf in uniq_conf])
            # tpr_fpr = [calculate_precision_recall(confidence > conf, test_labels)[-2:] for conf in uniq_conf]
            tpr, fpr = tmp[:, -2], tmp[:, -1]
            # print "max_level = ", DecisionTree.max_level
            plot_roc(fpr, tpr, marker_size=10)
            plt.show()

    print args.depth, np.sum(result == test_labels) / float(len(test_labels))

