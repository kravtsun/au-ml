#!/usr/bin/python
import numpy as np
import argparse
import matplotlib.pyplot as plt

def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m

def accuracy(results, labels):
    return float(np.sum(results == labels)) / len(labels)

def pocket_pla(features, labels, iterations=500, EPS=1e-5):
    n, nfeatures = features.shape
    features = np.hstack((np.ones((n, 1)), features.copy()))
    assert features.shape == (n, nfeatures + 1)
    sign_labels = labels.copy()
    sign_labels[np.where(labels == 0)] = -1
    assert len(set(sign_labels.flatten())) == 2
    pinv = np.linalg.pinv(features)
    ein = best_ein = 1e50
    w = best_w = pinv.dot(sign_labels)
    assert w.shape == (3,)
    it = 0
    while it < iterations:
        cur_labels = np.sign(features.dot(w))
        diff = cur_labels != sign_labels
        ein = np.sum(diff, dtype=float) / n
        if best_ein < EPS:
            break
        if ein < best_ein:
            best_w, best_ein = w, ein
        print it, ein
        assert cur_labels.shape == (n,)
        i = np.argmax(diff)
        xi, yi = features[i,:], cur_labels[i]
        w = w - xi * yi
        it += 1
    assert best_w.shape == (3,)
    print "best_ein = ", best_ein
    return best_w.flatten()

def plot_line(w, xmin, xmax, N=1000):
    assert w.shape == (3,)
    xx = np.linspace(xmin, xmax, N)
    yy = (-w[0] - w[1] * xx) / w[2]
    plt.plot(xx, yy, color='green')

def plot_points(data, style=""):
    plt.plot(data[:,0], data[:,1], style, markersize=10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    args = parser.parse_args()
    data = read_csv(args.filename)
    nfeatures = data.shape[1] - 1
    assert nfeatures == 2
    labels = data[:, -1]
    n = len(labels)
    features = data[:, :-1]
    zeroes = features[np.where(labels == 0)]
    ones = features[np.where(labels == 1)]
    plot_points(zeroes, 'ro')
    plot_points(ones, 'bx')
    w  = pocket_pla(features, labels)
    xmin, xmax = np.min(features[:,0]), np.max(features[:,0])
    plot_line(w, xmin, xmax)
    plt.show()

