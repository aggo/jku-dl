#!/usr/bin/env python
import logging
import sys
import os
import time

import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
"""layer_defs = [];
layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});
layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:3, stride:3});
layer_defs.push({type:'softmax', num_classes:10});
"""



earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')


def train_network(x_tr, y_tr, x_va, y_va):

    print(x_tr.shape) # 10000L, 1L, 28L, 28L (samples, channels, rows, columns)
    import numpy
    # x_tr = numpy.lib.pad(x_tr, ((2,2),(2,2)),'minimum')
    # x_va = numpy.lib.pad(x_tr, ((2,2),(2,2)),'minimum')
    img_rows = x_tr.shape[2]
    img_cols = x_tr.shape[3]
    batch_size = 1
    n_epochs = 12
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    model = Sequential()  # aka linear stacks of layers
    # apply a 5x5 conv with 5 output filters on a 28x28
    model.add(Convolution2D(20, 5, 5,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols),
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(150, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())                     # flattens the input: from (None, 64, 32, 32) to (None, 65536)

    model.add(Dense(200, activation='relu'))  # Dense(100) is a fully-connected layer with 100 hidden units.
    model.add(Dropout(0.1))                  # model.add(Dropout(0.5))
    model.add(Dense(output_dim=10, activation='softmax'))



    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    history = model.fit(x_tr, y_tr, batch_size=batch_size, nb_epoch=n_epochs,
              verbose=1, validation_data=(x_va, y_va), callbacks=[earlyStopping])

    yaml = model.to_yaml()
    logging.getLogger(__name__).info("YAML representation: \n%s",yaml)

    return model


def eval_network(model, x_te, y_te):
    score = model.evaluate(x_te, y_te, verbose=0)
    logging.getLogger(__name__).info('Test score: %s', score[0])
    logging.getLogger(__name__).info('Test accuracy: %s', score[1])
    return score[0], score[1]


def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create a file handler
    fh = logging.FileHandler('../logs/experiments.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-15s === %(message)s')
    fh.setFormatter(formatter)

    # create console handler
    ch = logging.StreamHandler(sys.stdout)

    logger.addHandler(fh)
    logger.addHandler(ch)


if __name__ == '__main__':

    setup_logging()
    # logging.getLogger(__name__).info("from ")

    x_tr = np.load('../downloads/train_x.npy')
    y_tr = np.load('../downloads/train_y.npy')
    x_va = np.load('../downloads/valid_x.npy')
    y_va = np.load('../downloads/valid_y.npy')

    n_classes = 10
    y_tr = np_utils.to_categorical(y_tr, n_classes)
    y_va = np_utils.to_categorical(y_va, n_classes)

    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]):
            x_te = np.load(sys.argv[1])
            y_te = np.load(sys.argv[2])
            y_te = np_utils.to_categorical(y_te, n_classes)
            # x_te = x_te.reshape(-1, 784)   # remove this line if you use convs
        else:
            print("Invalid testset files passed")
            sys.exit(1)
    else:
        logging.getLogger(__name__).info("Using validation set instead of the actual testset")
        x_te, y_te = x_va, y_va

    # NR_EXPERIMENTS = 10
    # accuracies = []
    # for i in range(NR_EXPERIMENTS):
    model = train_network(x_tr, y_tr, x_va, y_va)
    test_score, test_accuracy = eval_network(model, x_te, y_te)
        # accuracies.append(test_accuracy)
    # logging.getLogger(__name__).info("Average accuracy over 10 experiments: %s",sum(accuracies)/NR_EXPERIMENTS)

    logging.getLogger(__name__).info("Done. ================================================\n")
