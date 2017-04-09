#!/bin/python
import argparse
import numpy as np
from common import read_csv, plot_clusters, distance, print_cluster_distribution

def similarity(p1, p2):
    return -distance(p1, p2, axis=1)

def affinity_propagation(data, pref=None, w=0.5, iterations=300, EPS=1e-3, verbose=True):
    n = data.shape[0]
    s = np.apply_along_axis(lambda p: similarity(p, data), 1, data)
    s = np.ma.masked_array(s, mask=False)
    if pref is None:
        pref = np.median(s)
    if w is None:
        w = 0.5
    s.flat[::n+1] = pref

    assert(s.shape == (n, n))
    r = np.zeros((n, n))
    a = np.ma.masked_array(np.zeros((n, n)), mask=False)
    it = 0
    rpositive = np.ma.zeros(shape=(n,n))
    rpositive.mask = False
    natural_index = np.arange(n)
    while it < iterations or np.sum(np.diag(a + r) > 0) == 0:
        it += 1
        oldr = r.copy()
        olda = a.copy()
        assum = a + s
        first_max_index = np.argmax(assum, axis=1) # I
        first_max = assum[natural_index, first_max_index].copy() # Y
        assum[natural_index, first_max_index] = -np.inf
        second_max = np.max(assum, axis=1)
        r = s - first_max[:, np.newaxis]
        r[natural_index, first_max_index] = s[natural_index, first_max_index] - second_max
        r = (1 - w) * r + w * oldr
        rpositive = np.maximum(r, 0)
        rpositive.flat[::n + 1] = r.flat[::n + 1]
        rpositive -= np.sum(rpositive, axis=0)
        a = np.clip(rpositive, 0, np.inf)
        a.flat[::n + 1] = np.diag(rpositive)
        a = w * olda - (1 - w) * a
        dist = distance(np.diag(olda) + np.diag(oldr), np.diag(a) + np.diag(r))
        if verbose:
            print "dist = ", dist
        if dist < EPS and np.sum(np.diag(a + r) > 0) > 0:
            break
    if verbose:
        print "it = ", it
    iexemplars = np.where(np.diag(a + r) > 0)[0]
    clusters = np.argmax(s[:, iexemplars], axis=1)
    clusters[iexemplars] = np.arange(n)
    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Affinity Propagation clusterization with given parameters")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("-p", "--pref", dest="pref", type=int, required=False, default=None)
    parser.add_argument("-w", dest="weight", type=float, required=False, default=0.75)
    args = parser.parse_args()
    data = read_csv(args.filename)
    labels = affinity_propagation(data, pref=args.pref, w=args.weight)
    plot_clusters(data, labels)
    print_cluster_distribution(labels)
