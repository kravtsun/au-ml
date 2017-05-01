#!/bin/python
import matplotlib.pyplot as plt
import argparse
import ast
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filename", help="Input file")
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        d = ast.literal_eval(f.readlines()[-1])
        train_accuracy, test_accuracy = d['acc'], d['val_acc']
        plt.figure(os.path.basename(args.filename))
        plt.plot(train_accuracy, 'bo', label="Train", markersize=10)
        plt.plot(train_accuracy, 'b')
        plt.plot(test_accuracy, 'ro', label="Test", markersize=10)
        plt.plot(test_accuracy, 'r')
        plt.legend()
        plt.show()