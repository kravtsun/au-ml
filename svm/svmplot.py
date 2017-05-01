import sys
sys.path.append("..")

from common import read_csv, get_features, get_labels, normalize_data, plot_points
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn import svm, preprocessing

def work(train_features, train_labels, test_features, test_labels, kernel, **kwargs):
    clf = svm.SVC(kernel=kernel, shrinking=True, **kwargs)
    clf.fit(train_features, train_labels)
    train_error = clf.score(train_features, train_labels)
    test_error = clf.score(test_features, test_labels)
    return sum(clf.n_support_), train_error, test_error, train_error - test_error

def load_data(filename):
    data = np.genfromtxt(filename, dtype=[('label', 'S1'), ('data', '30f')], skip_header=1, delimiter=",")
    labels = np.array([row[0] == 'B' for row in data])
    features = np.array([row[1] for row in data])
    # std0 = np.std(features, axis=0)
    # std1 = np.std(features, axis=1)
    # features = preprocessing.scale(features)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # features = min_max_scaler.fit_transform(features)
    return features, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Support vector machine.")
    parser.add_argument("--train", dest="train", required=True, help="Train filename")
    parser.add_argument("--test", dest="test", required=True, help="Test filename")
    args = parser.parse_args()
    train_features, train_labels = load_data(args.train)
    test_features, test_labels = load_data(args.test)
    assert train_features.shape[1] == test_features.shape[1]
    kernel = "linear"
    cc = np.arange(0.01, 1.01, 0.2)
    gg = np.arange(0.01, 1.01, 0.2)
    arr = []
    for kernel in ["linear", "poly", "rbf"]:
        for c in cc:
            for g in gg:
                if kernel == "poly":
                    for degree in range(2, 10+1):
                        arr.append(work(train_features, train_labels, test_features, test_labels, kernel, C=c, gamma=g, degree=degree))
                        # if arr[-1][0] > 200:
                        #     print kernel, c, g
                else:
                    arr.append(work(train_features, train_labels, test_features, test_labels, kernel, C=c, gamma=g))
                    # if arr[-1][0] > 200:
                    #     print kernel, c, g
    arr = np.array(arr)

    # kernel_name = kernel
    # kernel_name += "-" + "-".join(map(lambda p: str(p[0]) + ":" + str(p[1]), kwargs.items()))

    plt.figure("train_error")
    plot_points(arr[:, (0, 1)], 'o', markersize=10)
    plt.savefig("train_error.png")

    plt.figure("test_error")
    plot_points(arr[:, (0, 2)], 'o', markersize=10)
    plt.savefig("test_error.png")

    plt.figure("diff_error")
    plot_points(arr[:, (0, 3)], 'o', markersize=10)
    plt.savefig("diff_error.png")
