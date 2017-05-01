#!/usr/bin/python
import numpy as np
import argparse
import matplotlib.pyplot as plt
from common import plot_line, plot_points, plot_square_line, prepare_regression, accuracy

def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m

def pocket_pla(features, labels, iterations=10000, EPS=1e-5):
    n, nfeatures = features.shape
    sign_labels, w = prepare_regression(features, labels)
    best_w, best_ein = w, 1e50
    assert w.shape == (nfeatures,)
    it = 0
    while it < iterations:
        cur_labels = np.sign(features.dot(w))
        diff = cur_labels != sign_labels
        ein = np.sum(diff, dtype=float) / n
        if best_ein < EPS:
            break
        if ein < best_ein:
            best_cnt = np.sum(diff)
            best_w, best_ein = w, ein
        print it, ein
        assert cur_labels.shape == (n,)
        i = np.argmax(diff)
        xi, yi = features[i,:], cur_labels[i]
        w = w - xi * yi
        it += 1
    assert best_w.shape == (nfeatures,)
    print "best_ein = ", best_ein, best_cnt
    return best_w.flatten()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("--line", dest="line", action='store_true', required=False, default=False)
    parser.add_argument("-p", dest="polynomials", action='store_true', required=False, default=2)
    parser.add_argument("--iter", dest="iterations", type=int, required=False, default=10000)
    args = parser.parse_args()
    data = read_csv(args.filename)
    labels, features = data[:, -1], data[:, :-1]
    n, nfeatures = len(labels), features.shape[1]
    assert nfeatures == 2
    xf, yf = features[:,0], features[:,1]

    zeroes = features[np.where(labels == 0)]; plot_points(zeroes, 'ro')
    ones = features[np.where(labels == 1)]; plot_points(ones, 'bx')

    if args.line:
        features = np.hstack((np.ones((n, 1)), features.copy()))
        w  = pocket_pla(features, labels, args.iterations)
        xminmax = np.min(xf), np.max(xf)
        plot_line(w, *xminmax)
    else:
        sqr_features = None
        for k in range(args.polynomials + 1):
            for i in range(k+1):
                xfi = (xf**(i)).reshape((n, 1))
                yfki = (yf**(k-i)).reshape((n, 1))
                if sqr_features is None:
                    sqr_features = xfi * yfki
                else:
                    sqr_features = np.hstack((sqr_features, xfi * yfki))
        w = pocket_pla(sqr_features, labels, args.iterations)
        xminmax = np.min(xf), np.max(xf)
        yminmax = np.min(yf), np.max(yf)
        plot_square_line(w, xminmax, yminmax)
    plt.show()

