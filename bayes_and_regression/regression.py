#!/bin/python
import argparse
from mailcap import show

from common import *
import numpy as np

class LinearRegressor():

    def __init__(self, poly):
        assert poly >= 1
        self.poly = poly
        pass

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        X = self.prepare_data(X)
        pinv = np.linalg.pinv(X)
        self.w = pinv.dot(y)

    def nfeatures(self):
        return self.w.shape[0]

    def predict(self, X):
        X = self.prepare_data(X)
        assert X.shape[1] == self.nfeatures()
        w_ = self.w.reshape(self.nfeatures(), 1)
        return X.dot(w_)

    def prepare_data(self, X):
        assert len(X.shape) == 1 or X.shape[1] == 1
        X = np.array(list([x**p for p in range(self.poly+1)] for x in X.flatten()))
        return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multinomial distribution for Naive bayesian classification.")
    parser.add_argument("--train", dest="train", required=True, help="Train path")
    parser.add_argument("--test", dest="test", required=True, help="Test path")
    parser.add_argument("--poly", dest="poly", type=int, default=2, required=False, help="Regression pow")

    args = parser.parse_args()

    def load_data(filename):
        data = read_csv(filename)
        n = data.shape[0]
        # x = np.array(list([1, row[0]] for row in data))
        x = data[:, 0]
        y = data[:, 1]
        return x, y

    def error(real, predicted):
        assert len(real.shape) == 1
        nfloat = float(len(real))
        return np.sum(np.abs(real - predicted)) / nfloat

    def rsquare(real, predicted):
        real = real.flatten()
        predicted = predicted.flatten()
        u = error(real, predicted)
        ymean = np.mean(real)
        v = error(real, ymean * np.ones(shape=real.shape))
        return 1 - u / v

    for poly in range(2, 10+1):
        clf = LinearRegressor(poly)
        train_x, train_y = load_data(args.train)
        clf.fit(train_x, train_y)

        def plot_line(x, y, style, **kwargs):
            asort = x.argsort()
            plt.plot(x[asort], y[asort], style, **kwargs)

        sqr_features = None

        plt.figure("noisy_cosine_p_{}".format(poly))
        plot_line(train_x, train_y, 'ob', label="train-real")
        plot_line(train_x, clf.predict(train_x), 'g-', label="train-predict")

        test_x, test_y = load_data(args.test)
        plot_line(test_x, test_y, 'or', label="test-real")
        test_predicted = clf.predict(test_x)
        plot_line(test_x, test_predicted, 'k-', label="test-predict")

        dx = abs(train_x.max() - train_x.min())
        dy = abs(train_y.max() - train_y.min())
        f = lambda x, dx: (x.min() - dx, x.max() + dx)
        print("r-score=", rsquare(test_y, test_predicted), "poly=", poly)

        # plot_square_line(clf.w, (train_x.min(), train_x.max()), (train_y.min(), train_y.max()))
        # plt.legend()
        # plt.show()
