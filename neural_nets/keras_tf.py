#!/bin/python
import sys
sys.path.append("..")
import os

from common import read_csv, normalize_data, prepare_regression, get_features, get_labels, calculate_precision_recall
import numpy as np
import matplotlib.pyplot as plt
import argparse

PNG_EXTENSION = ".png"
PNG_DIMENSION = 28
PNG_SHAPE = (PNG_DIMENSION, PNG_DIMENSION)
PNG_SIZE = PNG_DIMENSION ** 2
NCLASSES = 10
def load_data(path):
    assert os.path.exists(path)
    features = []
    classes = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(PNG_EXTENSION):
                label_letter = root.split("/")[-1]
                label = ord(label_letter) - ord('A')
                # class_label = np.zeros((NCLASSES,))
                # class_label[label] = 1
                class_label = label
                filename = os.path.join(root, f)
                try:
                    v = plt.imread(filename)#.flatten()
                    features.append(v)
                    classes.append(class_label)
                except IOError:
                    pass
    return np.array(features), np.array(classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolutional neural network.")
    parser.add_argument("--train", dest="train", required=True, help="Train path")
    parser.add_argument("--test", dest="test", required=True, help="Test path")
    parser.add_argument("--epochs", dest="epochs", type=int, required=False, default=1, help="Number of epochs")
    parser.add_argument("--activate", dest="activate", type=str, required=False, default="relu", help="Activation function")
    parser.add_argument("--decide", dest="decide", type=str, required=False, default="softmax", help="Deciding function")
    parser.add_argument("--var", dest="var", type=int, default=0)
    # parser.add_argument("--validate", dest="validate", required=True, help="Validate path")
    args = parser.parse_args()
    # args.activate = "tanh"

    features, labels = load_data(args.train)
    test_features, test_labels = load_data(args.test)
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Conv2D, Reshape, Flatten, MaxPool2D
    from keras import optimizers

    deciding_func = "softmax"

    models = [
        Sequential([
            Flatten(input_shape=PNG_SHAPE),
            Dense(4092, input_dim=PNG_SIZE),
            Activation(args.activate),
            Dense(2048),
            Activation(args.activate),
            Dense(1024),
            Activation(args.activate),
            Dense(10),
            Activation(deciding_func)
        ]),

        Sequential([
            Reshape((PNG_DIMENSION, PNG_DIMENSION, 1), input_shape=PNG_SHAPE),
            Conv2D(3, padding="same", strides=(1, 1), kernel_size=(3, 3)),
            Activation(args.activate),
            Flatten(),
            Dense(4092),
            Activation(args.activate),
            Dense(2048),
            Activation(args.activate),
            Dense(1024),
            Activation(args.activate),
            Dense(10),
            Activation(deciding_func)
        ]),

        Sequential([
            Reshape((PNG_DIMENSION, PNG_DIMENSION, 1), input_shape=PNG_SHAPE),
            Conv2D(3, padding="same", strides=(1, 1), kernel_size=(3, 3)),
            MaxPool2D(pool_size=(2,2), strides=1),
            Activation(args.activate),
            Conv2D(3, padding="same", strides=(1, 1), kernel_size=(3, 3)),
            MaxPool2D(pool_size=(2, 2), strides=1),
            Activation(args.activate),
            Flatten(),
            Dense(4092),
            Activation(args.activate),
            Dense(2048),
            Activation(args.activate),
            Dense(1024),
            Activation(args.activate),
            Dense(10),
            Activation(deciding_func)
        ])
    ]

    model = models[args.var]
    sgd = optimizers.SGD(lr=0.05, decay=1e-7, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hot_classes = keras.utils.to_categorical(labels, num_classes=NCLASSES)
    test_hot_classes = keras.utils.to_categorical(test_labels, num_classes=NCLASSES)
    hist = model.fit(features, hot_classes, validation_data=(test_features, test_hot_classes), epochs=args.epochs, batch_size=128)
    model.save("model_" + str(args.var) + "_" + args.activate + ".txt")
    print hist.history
