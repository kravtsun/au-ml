#!/bin/python
import argparse
import numpy as np
from common import read_csv, plot_clusters, distance, print_cluster_distribution

def similarity(p1, p2):
    return -distance(p1, p2, axis=1)

def sklearn_affinity_propagation(data, pref=None, damping=0.80):
    from sklearn.cluster import AffinityPropagation

    af = AffinityPropagation(preference=pref, damping=damping, verbose=True).fit(data)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    print pref, n_clusters_
    return labels

def affinity_propagation(data, pref=None, w=0.5, iterations=100, EPS=1e-3):
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
    ra = np.diag(r) + np.diag(a)
    rpositive = np.ma.zeros(shape=(n,n))
    rpositive.mask = False
    while it < iterations or np.sum(np.diag(a + r) > 0) == 0:
        it += 1
        oldr = r.copy()
        for k in range(n):
            r[:, k] = s[:, k]
            assert id(r[0, k]) != s[0, k]
            a.mask[:, k] = True
            s.mask[:, k] = True
            r[:, k] -= np.max(a + s, axis=1)
            s.mask[:, k] = False
            a.mask[:, k] = False
        r = w * oldr + (1 - w) * r
        rpositive.data[:] = np.maximum(0, r)
        rpositive.data.flat[::n+1] = r.flat[::n+1]
        olda = a.copy()

        for i in range(n):
            rpositive.mask[i, :] = True
            rpositive_sum = np.sum(rpositive, axis=0)
            a[i, :] = np.minimum(0, rpositive_sum)
            a[i, i] = rpositive_sum[i]
            rpositive.mask[i, :] = False
        a = w * olda + (1 - w) * a
        dist = distance(np.diag(olda) + np.diag(oldr), np.diag(a) + np.diag(r))
        print "dist = ", dist
        if dist < EPS and np.sum(np.diag(a + r) > 0) > 0:
            break
    print "it = ", it
    iexemplars = np.where(np.diag(a + r) > 0)[0]
    clusters = np.argmax(s[:, iexemplars], axis=1)
    clusters[iexemplars] = np.arange(n)
    # labels = iexemplars[clusters]
    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Affinity Propagation clusterization with given parameters")
    parser.add_argument("-f", dest="filename", type=str, required=True)
    parser.add_argument("-p", "--pref", dest="pref", type=int, required=False, default=None)
    parser.add_argument("-w", dest="weight", type=float, required=False, default=0.75)
    parser.add_argument("--sklearn", dest="sklearn", action="store_true", default=False)
    args = parser.parse_args()
    data = read_csv(args.filename)
    if args.sklearn:
        labels = sklearn_affinity_propagation(data, args.pref, args.weight)
    else:
        labels = affinity_propagation(data, pref=args.pref, w=args.weight)
    plot_clusters(data, labels)
    print_cluster_distribution(labels)
