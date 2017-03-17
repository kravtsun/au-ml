#!/usr/bin/python
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import LeaveOneOut
from collections import Counter
from sklearn.preprocessing import normalize

filename = sys.argv[1]
k = int(sys.argv[2])

v = np.loadtxt(filename, delimiter=',',skiprows=1)
nsamples = v.shape[0]
x = v[:, :-1]
x = normalize(x, axis=0, norm='max')
y = v[:, -1]

neigh = NearestNeighbors(n_neighbors=k, metric='euclidean')

cnt = 0
loo = LeaveOneOut()

for train_index, test_index in loo.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier = neigh.fit(x_train, y_train)
    indexes = neigh.kneighbors(x_test, return_distance=False)
    labels = y_train[indexes][0]
    chosen_label = Counter(labels).most_common(1)[0][0]
    assert len(y_test) == 1
    cnt += chosen_label != y_test[0]

print(float(cnt) / nsamples)
