#!/bin/python
import sys
sys.path.append("..")
import os

from common import read_csv, normalize_data, prepare_regression, get_features, get_labels, calculate_precision_recall
import numpy as np
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
sess = tf.InteractiveSession()

PNG_EXTENSION = ".png"
PNG_DIMENSION = 28
PNG_SIZE = PNG_DIMENSION ** 2
NCLASSES = 10
def load_data(path):
    assert os.path.exists(path)
    features = []
    classes = []
    for root, dirs, files in os.walk(args.train):
        for f in files:
            if f.endswith(PNG_EXTENSION):
                label_letter = root.split("/")[-1]
                label = ord(label_letter) - ord('A')
                class_label = np.zeros((NCLASSES,))
                class_label[label] = 1
                filename = os.path.join(root, f)
                try:
                    v = plt.imread(filename)#.flatten()
                    features.append(v)
                    classes.append(class_label)
                except IOError:
                    pass
    return np.array(features), np.array(classes)

def train_and_test(features, classes, test_features, test_classes, epochs=1000):
    assert features.shape[0] == classes.shape[0]
    n = features.shape[0]
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def next_batch(k):
        from random import shuffle
        assert features.shape[0] > k and classes.shape[0] > k
        arr = range(n)
        shuffle(arr)
        ii = arr[:k]
        return features[ii], classes[ii]

    activation_func = tf.nn.relu
    def dense_layer(x, k):
        # x should already be flat.
        # x_flat = tf.reshape(x, [-1, PNG_SIZE])
        assert len(x.shape) == 2
        size = int(x.shape[1])
        W = weight_variable([size, k])
        b = bias_variable([k])
        s = tf.matmul(x, W) + b
        x = activation_func(s)
        return x

    def readout_layer(x):
        assert len(x.shape) == 2
        size = int(x.shape[1])
        W = weight_variable([size, NCLASSES])
        b = bias_variable([NCLASSES])
        s = tf.matmul(x, W) + b
        x = activation_func(s)
        return x

    x = tf.placeholder(tf.float32, shape=[None, PNG_DIMENSION, PNG_DIMENSION])
    y_ = tf.placeholder(tf.float32, shape=[None, NCLASSES])
    keep_prob = tf.placeholder(tf.float32)
    
    x_flat = tf.reshape(x, [-1, PNG_SIZE])
    x_1 = dense_layer(x_flat, 4096)
    x_2 = dense_layer(x_1, 2048)
    x_3 = dense_layer(x_2, 1024)

    # readout layer.
    y_conv = readout_layer(x_3)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        batch = next_batch(100)
        if i % 5 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            test_accuracy = accuracy.eval(feed_dict={x: test_features, y_: test_classes})
            print("%d,%g,%g" % (i, train_accuracy, test_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolutional neural network.")
    parser.add_argument("--train", dest="train", required=True, help="Train path")
    parser.add_argument("--test", dest="test", required=True, help="Test path")
    # parser.add_argument("--validate", dest="validate", required=True, help="Validate path")
    args = parser.parse_args()

    features, classes = load_data(args.train)
    test_features, test_classes = load_data(args.test)
    train_and_test(features, classes, test_features, test_classes)
