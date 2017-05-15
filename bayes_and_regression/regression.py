#!/bin/python
import argparse
import numpy as np

import sys
sys.path.append("..")
from common import *


class LinearRegressor():
    def __init__(self, poly, alpha=0):
        assert poly >= 1
        self.poly = poly
        self.alpha = alpha
        pass

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        X = self.prepare_data(args.poly, X)
        n, nfeatures = X.shape
        tmp = X.T.dot(X) + self.alpha * np.eye(nfeatures, nfeatures)
        pinv = np.linalg.inv(tmp).dot(X.T)
        self.w = pinv.dot(y)

    def nfeatures(self):
        return self.w.shape[0]

    def predict(self, X):
        X = LinearRegressor.prepare_data(args.poly, X)
        assert X.shape[1] == self.nfeatures()
        w = self.w.reshape(self.nfeatures(), 1)
        return X.dot(w)

    @staticmethod
    def prepare_data(poly, X):
        if len(X.shape) == 1 or X.shape[1] == 1:
            assert poly
            return np.array(list([x ** p for p in range(poly + 1)] for x in X.flatten()))
        else:
            assert poly <= 2
            n = X.shape[0]
            nfeatures = X.shape[1]
            def bit_ones(x):
                if x == 0:
                    return 0
                return bit_ones(x / 2) + x % 2

            if poly == 2:
                for mask in range(1 << nfeatures):
                    p = bit_ones(mask)
                    if p != 2: continue
                    y = np.ones(shape=(n, 1))
                    for i in range(nfeatures):
                        if (1 << i) & mask > 0:
                            # print(y.shape, X[:, i].reshape((n, 1)).shape)
                            np.multiply(y, X[:, i].reshape((n, 1)), y)
                            # y = y * X[:, i]
                            assert len(y) == X.shape[0]
                    X = np.hstack((X, y))
                for i in range(nfeatures):
                    squared_feature = X[:, i].reshape((n, 1)) ** 2
                    X = np.hstack((X, squared_feature))
            X = np.hstack((np.ones(shape=(n, 1)), X))
            return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multinomial distribution for Naive bayesian classification.")
    parser.add_argument("--train", dest="train", required=True, help="Train path")
    parser.add_argument("--test", dest="test", required=True, help="Test path")
    parser.add_argument("--poly", dest="poly", type=int, default=2, required=False, help="Regression pow")
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.0, required=False, help="Regularization coefficient")
    parser.add_argument("--show", dest="show", action='store_true', required=False, default=False)

    args = parser.parse_args()

    def error(y_true, y_pred):
        assert len(y_true.shape) == 1
        # nfloat = float(len(y_true))
        # return np.sum(np.abs(y_true - y_pred)) / nfloat
        return ((y_true - y_pred) ** 2).sum()

    def rscore(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        u = error(y_true, y_pred)
        ymean = np.mean(y_true)
        v = error(y_true, ymean * np.ones(shape=y_true.shape))
        return 1 - u / v

    # for poly in range(2, 10+1):
    clf = LinearRegressor(poly=args.poly, alpha=args.alpha)
    train_x, train_y = load_data(args.train)

    clf.fit(train_x, train_y)
    train_predicted = clf.predict(train_x)

    def plot_line(x, y, style, **kwargs):
        x = x.flatten()
        y = y.flatten()
        asort = x.argsort()
        plt.plot(x[asort], y[asort], style, **kwargs)
    test_x, test_y = load_data(args.test)
    test_predicted = clf.predict(test_x)

    # dx = abs(train_x.max() - train_x.min())
    # dy = abs(train_y.max() - train_y.min())
    # f = lambda x, dx: (x.min() - dx, x.max() + dx)
    print(rscore(test_y, test_predicted))
    # plot_square_line(clf.w, (train_x.min(), train_x.max()), (train_y.min(), train_y.max()))

    if train_x.shape[1] == 1 and args.show:
        plt.figure("noisy_cosine_p_{}".format(args.poly))
        plot_line(train_x, train_y, 'ob', label="train-real")
        plot_line(train_x, train_predicted, 'g-', label="train-predict")
        plot_line(test_x, test_y, 'or', label="test-real")
        plot_line(test_x, test_predicted, 'k-', label="test-predict")
        plt.legend()
        plt.show()



