#!/bin/python
import argparse
from common import *
import numpy as np
# from sklearn.naive_bayes import GaussianNB

class NaiveBayes():
    EPS = 1e-15
    # Training
    def fit(self, features, labels):
        assert features.shape[0] == labels.shape[0]
        assert len(features.shape) == 2
        n, nfeatures = features.shape
        labels = labels.astype(int)
        assert list(set(labels)) == [0, 1]
        self.means, self.vars, self.pys = [], [], []
        self.multiplier = []
        for i in range(2):
            cur_features = features[labels == i]
            assert cur_features.shape[0] > 0
            means = np.mean(cur_features, axis=0)
            vars = np.var(cur_features, axis=0)
            vars = np.maximum(vars, vars.max() * NaiveBayes.EPS)
            assert means.shape == (nfeatures, ) and vars.shape == (nfeatures, )
            self.means.append(means)
            self.vars.append(vars)
            self.pys.append(cur_features.shape[0] / float(n))
            self.multiplier.append(1.0 / np.sqrt(2.0 * np.pi * vars))
        pass

    def likelihood(self, x, label):
        xbias2 = np.square(x - self.means[label])
        pxiy = self.multiplier[label] * np.exp(-xbias2 / (2 * self.vars[label]))
        # where = np.isnan(pxiy)
        # pxiy[where] = x[where] == self.means[label][where]
        return np.maximum(pxiy, pxiy.max() * NaiveBayes.EPS)

    def nfeatures(self):
        return self.means[0].shape[0]

    def likelihood_ratio(self, x):
        logratio = np.log(self.likelihood(x, 1) / self.likelihood(x, 0))
        assert logratio.shape[0] == self.nfeatures()
        return np.sum(logratio) + np.log(self.pys[1] / self.pys[0])

    def prob(self, x):
        # p0 = np.prod(self.likelihood(x, 0))
        # p1 = np.prod(self.likelihood(x, 1))
        assert (p0 + p1) > 0
        return p0 / (p0 + p1), p1 / (p0 + p1)

    def log_probas(self, features):
        q = np.array([self.likelihood_ratio(x) for x in features])
        expq = np.exp(q)
        return expq / (1 + expq)

    def probas(self, features):
        nfeatures = self.nfeatures()
        assert features.shape[1] == nfeatures
        res = []
        for i, x in enumerate(features):
            res.append(self.prob(x))
        return res
        # return np.array([self.prob(x) for x in features])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Support vector machine.")
    parser.add_argument("--train", dest="train", required=True, help="Train path")
    parser.add_argument("--test", dest="test", required=True, help="Test path")
    args = parser.parse_args()

    classifier = NaiveBayes()
    # Training.
    train_data = read_csv(args.train)
    train_features, train_labels = get_features(train_data), get_labels(train_data)
    classifier.fit(train_features, train_labels)

    def all_work(classifier, filename, sample_name):
        data = read_csv(filename)
        features, labels = get_features(data), get_labels(data)
        probs = classifier.log_probas(features)
        predicted = probs > 0.5
        print sample_name + "_accuracy = ", accuracy(predicted, labels)
        return probs, labels

    all_work(classifier, args.train, "train")
    probs, labels = all_work(classifier, args.test, "test")
    thresholds = np.linspace(0, 1.0, 100)
    results = np.array([calculate_precision_recall(probs > threshold, labels) for threshold in thresholds])
    tpr, fpr = results[:, 2], results[:, 3]
    plot_roc(fpr, tpr)
    # plot_roc(thresholds, results[:, 0])
    plt.show()







