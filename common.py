#!/bin/python
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename, max_rows=None):
    m = np.genfromtxt(filename, delimiter=',', dtype=float, skip_header=1, max_rows=max_rows)
    return m

def distance(p1, p2, axis=None):
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=axis))

def get_features(data):
    return data[:, :-1]

def get_labels(data):
    return data[:, -1]

def normalize_data(data):
    features = get_features(data)
    max_features = np.max(features, axis=0)
    nfeatures = data.shape[1] - 1
    assert max_features.shape == (nfeatures,)
    max_features[max_features == 0.0] = 1.0
    features /= max_features
    assert np.max(features) <= 1.0

def calc_distances_to_feature_vector(features, row):
    distances = np.sqrt(np.sum((features - row) ** 2, axis=1))
    return distances

# TODO: place plot routines to a separate file.
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

def prepare_regression(features, labels):
    sign_labels = labels.copy()
    sign_labels[np.where(labels == 0)] = -1
    assert len(set(sign_labels.flatten())) == 2
    pinv = np.linalg.pinv(features)
    w = best_w = pinv.dot(sign_labels)
    return sign_labels, w

def calculate_precision_recall(results, labels):
    n = len(labels)
    row_positives = np.sum(results, axis=-1)
    if np.all(row_positives == 0):
        return np.nan, 0, 0, 0
    assert np.all(row_positives > 0)
    def count_true_positives(row_result):
        return np.sum(np.logical_and(row_result, labels))
    true_positives = np.apply_along_axis(count_true_positives, -1, results).astype(dtype=np.float)
    true_positives = true_positives
    precision = true_positives / row_positives
    assert np.sum(np.unique(labels) - np.array([0, 1])) == 0.0
    total_positives = np.sum(labels)
    assert total_positives > 0
    recall = true_positives / total_positives
    tpr = true_positives / total_positives
    fpr = (row_positives - true_positives) / (n - total_positives)
    return precision, recall, tpr, fpr
