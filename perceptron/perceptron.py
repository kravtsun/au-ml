#!/usr/bin/python
import numpy as np
import argparse
import matplotlib.pyplot as plt

def read_csv(filename):
    m = np.genfromtxt(filename, delimiter=',', dtype=np.double, skip_header=1)
    return m

def accuracy(results, labels):
    return float(np.sum(results == labels)) / len(labels)

def pocket_pla(features, labels, iterations=10000, EPS=1e-5):
    n, nfeatures = features.shape
    sign_labels = labels.copy()
    sign_labels[np.where(labels == 0)] = -1
    assert len(set(sign_labels.flatten())) == 2
    pinv = np.linalg.pinv(features)
    ein = best_ein = 1e50
    w = best_w = pinv.dot(sign_labels)
    assert w.shape == (nfeatures,)
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
    assert best_w.shape == (nfeatures,)
    print "best_ein = ", best_ein
    return best_w.flatten()

def plot_line(w, xmin, xmax, N=1000):
    assert w.shape == (3,)
    xx = np.linspace(xmin, xmax, N)
    yy = (-w[0] - w[1] * xx) / w[2]
    plt.plot(xx, yy, color='green')

def plot_square_line(w, xminmax, yminmax, N=1000):
    wlen = len(w)
    assert wlen == 3 or wlen == 6 or wlen == 10 or wlen == 15
    w.resize((15,), refcheck=False)
    max_pow = 4
    scale = 0.3
    dx, dy = scale * (xminmax[1] - xminmax[0]), scale * (yminmax[1] - yminmax[0])
    xx = np.linspace(xminmax[0] - dx, xminmax[1] + dx, N)
    yy = np.linspace(yminmax[0] - dy, yminmax[1] + dy, N)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros(X.shape)
    wi = 0
    for k in range(max_pow + 1):
        if wi >= wlen:
            break
        for i in range(k+1):
            Z = Z + w[wi] * X**(i) * Y**(k-i)
            wi += 1
    # Z = w[0] + w[1] * X + w[2] * Y + w[3] * X**2 + w[4] * X * Y + w[5] * Y**2
    # plt.xlim((-2,5))
    plt.contour(X, Y, Z, [0])

def plot_points(data, style=""):
    plt.plot(data[:,0], data[:,1], style, markersize=10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run k-means clusterization with given arguments")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("--line", dest="line", action='store_true', required=False, default=False)
    parser.add_argument("-p", dest="polynomials", action='store_true', required=False, default=3)
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

