#!/bin/python
import argparse
from common import *
import numpy as np
import re

rewords = re.compile("\w+")
def line_to_words(line):
    return [w.lower() for w in rewords.findall(line)]

def read_text_csv(filename):
    with open(filename, "r") as f:
        return [line_to_words(line) for line in f]

class Multinomial():
    EPS = 1e-9
    def __init__(self, alpha = 1.0, bernoulli=False):
        self.dict = {}
        self.alpha = alpha
        self.bernoulli = bernoulli

    def index(self, word):
        if not self.dict.has_key(word):
            return -1
        return self.dict[word]

    @staticmethod
    def concatenate_lists(lists):
        res = sum(lists, [])
        return res

    def nwords(self):
        return len(self.unique_words)

    def fit(self, word_lists, labels):
        assert len(labels) == len(word_lists)
        assert {False, True} == set(labels)
        unique_labels = list(set(labels))
        nsamples = float(len(labels))
        from collections import Counter, defaultdict

        all_words = Multinomial.concatenate_lists(word_lists)
        self.unique_words = list(set(all_words))
        self.unique_words_index = defaultdict(lambda : -1, {(w, i) for i, w in enumerate(self.unique_words)})
        filter_count = lambda v, value: sum(el == value for el in v)
        self.pys = [filter_count(labels, False) / nsamples, filter_count(labels, True) / nsamples]

        if self.bernoulli:
            unique_word_lists = [list(set(wl)) for wl in word_lists]
            assert unique_labels[0] == False and unique_labels[1] == True
            counters = [Counter(Multinomial.concatenate_lists(uwl for j, uwl in enumerate(unique_word_lists) if labels[j] == label)) for label in unique_labels]
            self.pw = [np.array([counter[w] / nsamples for w in self.unique_words]) for counter in counters]
        else:
            counters = [Counter(Multinomial.concatenate_lists(wl for j, wl in enumerate(word_lists) if labels[j] == label)) for label in unique_labels]
            Ny = [sum(c.values()) for c in counters]
            def formula(w, counter, ny):
                return (counter[w] + self.alpha) / (ny + self.alpha * self.nwords())
            self.theta = [defaultdict(lambda : 0, dict((w, formula(w, counter, ny)) for w in self.unique_words)) for ny, counter in zip(Ny, counters)]
        pass

    def likelihood(self, x, label):
        if self.bernoulli:
            xxi = np.zeros((self.nwords(),), dtype=int)
            wi = np.array(filter(lambda x: x != -1, [self.unique_words_index[w] for w in set(x)]))
            if len(wi) > 0:
                xxi[wi] = 1
            ppi = self.pw[label]
            nwords = len(self.unique_words)
            assert type(x) == list and xxi.shape == (nwords,) and ppi.shape == (nwords,)
            real_likelihood = ppi * xxi + (1 - ppi) * (1 - xxi)
        else:
            real_likelihood = np.array([self.theta[label][w] for w in x])
        return np.maximum(real_likelihood, Multinomial.EPS)

    def likelihood_ratio(self, x):
        logratio = np.log(self.likelihood(x, 1) / self.likelihood(x, 0))
        return np.sum(logratio) + np.log(self.pys[1] / self.pys[0])

    def log_probas(self, features):
        q = np.array([self.likelihood_ratio(x) for x in features])
        expq = np.exp(q)
        return expq / (1 + expq)

    # def prepare_data(self, features):
    #     if self.bernoulli:
    #         return list(set(features))
    #     else:
    #         return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multinomial distribution for Naive bayesian classification.")
    parser.add_argument("--train", dest="train", required=True, help="Train path")
    parser.add_argument("--test", dest="test", required=True, help="Test path")
    parser.add_argument("--alpha", dest="alpha", type=float, default=1.0, required=False, help="Smoothing coefficient")
    parser.add_argument("--bernoulli", dest="bernoulli", required=False, action='store_true', help="Use Bernoulli distribution.")
    args = parser.parse_args()

    def load_features_labels(filename):
        lines = read_text_csv(filename)[1:]
        labels = [t[0] == "spam" for t in lines]
        features = [t[1:] for t in lines]
        return features, labels

    def train(filename):
        features, labels = load_features_labels(filename)
        classifier = Multinomial(alpha=args.alpha, bernoulli=args.bernoulli)
        classifier.fit(features, labels)
        return classifier

    def all_work(filename, classifier):
        features, labels = load_features_labels(filename)
        probs = classifier.log_probas(features)
        predicted = probs > 0.5
        print(accuracy(predicted, labels))
        return probs, labels

    classifier = train(args.train)
    all_work(args.train, classifier)
    probs, labels = all_work(args.test, classifier)
    thresholds = np.linspace(0, 1.0, 100)
    results = np.array([calculate_precision_recall(probs > threshold, labels) for threshold in thresholds])
    tpr, fpr = results[:, 2], results[:, 3]
    plot_roc(fpr, tpr)
    # plot_roc(thresholds, results[:, 0])
    plt.show()