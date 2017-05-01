import sys
sys.path.append("..")

from common import read_csv, get_features, get_labels, normalize_data, plot_points
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn import svm, preprocessing

def work(features, labels, kernel, **kwargs):
    kernel_name = kernel
    if kernel == "poly":
        kernel_name += "-" + "-".join(map(str, kwargs.values()))
    plt.figure(kernel_name)
    clf = svm.SVC(kernel=kernel, C=1.0, gamma=0.5, shrinking=True, **kwargs)
    # sign_labels = np.asarray(labels, dtype=int)
    # sign_labels[np.where(sign_labels) == 0.0] = -1.0
    clf.fit(features, labels)
    N = 500
    xminmax, yminmax = (np.min(features[:, 0]) - 0.5, np.max(features[:, 0]) + 0.5), \
                       (np.min(features[:, 1]) - 0.5, np.max(features[:, 1]) + 0.5)
    xx, yy = np.meshgrid(np.linspace(xminmax[0], xminmax[1], N),
                         np.linspace(yminmax[0], yminmax[1], N))
    ones = features[np.where(labels == 1.0)]
    zeros = features[np.where(labels == 0.0)]
    plot_points(zeros, "ob", markersize=10)
    plot_points(ones, "or", markersize=10)
    plot_points(clf.support_vectors_, "*k", markersize=10)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.summer)
    plt.savefig(kernel_name + ".png", bbox_inches = 'tight')
    # plt.show()
    print kernel_name, clf.score(features, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Support vector machine.")
    parser.add_argument("--data", dest="data", required=True, help="Data filename")
    args = parser.parse_args()
    data = read_csv(args.data)
    features = get_features(data)
    features = preprocessing.scale(features)
    labels = get_labels(data)

    work(features, labels, "poly", degree=10)
    work(features, labels, "linear")
    work(features, labels, "rbf")

    for degree in range(2, 10+1):
        work(features, labels, "poly", degree=degree)

