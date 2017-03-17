#!/usr/bin/python
import argparse
from operator import itemgetter
from itertools import starmap
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import normalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run KNN classifier with given arguments")
    parser.add_argument("-f", dest="filename", required=True, help="CSV file.")
    parser.add_argument("-k", type=int, dest="k", required=True)
    parser.add_argument("-n", "--normalize", dest="normalize", action="store_true", default=False)
    args = parser.parse_args()

    v = np.loadtxt(args.filename, delimiter=',', skiprows=1)
    nsamples = v.shape[0]
    x = v[:, :-1]
    if args.normalize:
        x = normalize(x, axis=0, norm='max')
    y = v[:, -1]

    neigh = KNeighborsClassifier(n_neighbors=args.k, metric='euclidean')

    loo = LeaveOneOut()
    def split_func(train_index, test_index):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            neigh.fit(x_train, y_train)
            labels = neigh.predict(x_test)
            assert labels.shape == (1,)
            return labels[0]
    loo_split = list(loo.split(x, y))
    predict_labels = list(starmap(split_func, loo_split))
    test_indexes = zip(*map(itemgetter(1), loo_split))[0]
    test_labels = y[np.array(test_indexes)]
    missed_count = np.count_nonzero(test_labels != predict_labels)
    print(float(missed_count) / nsamples)
