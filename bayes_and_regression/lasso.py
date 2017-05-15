#!/bin/python
import argparse
import sys
sys.path.append("..")
sys.path.append(".")
from common import *
from regression import LinearRegressor

from sklearn import linear_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multinomial distribution for Naive bayesian classification.")
    parser.add_argument("--train", dest="train", required=True, help="Train path")
    parser.add_argument("--test", dest="test", required=True, help="Test path")
    parser.add_argument("--poly", dest="poly", type=int, default=2, required=False, help="Regression pow")
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.5, required=False, help="Regularization coefficient")
    args = parser.parse_args()

    clf = linear_model.Lasso(alpha=args.alpha, max_iter=100000, fit_intercept=True, normalize=False)

    train_x, train_y = load_data(args.train)
    train_x = LinearRegressor.prepare_data(args.poly, train_x)
    clf.fit(train_x, train_y)
    print(clf.coef_)

    test_x, test_y = load_data(args.test)
    test_x = LinearRegressor.prepare_data(args.poly, test_x)
    print clf.score(test_x, test_y)
