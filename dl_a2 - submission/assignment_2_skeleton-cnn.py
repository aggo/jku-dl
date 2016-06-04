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
from keras.regularizers import l2, activity_l2, l1l2
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

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')


def train_network(x_tr, y_tr, x_va, y_va):

    print(x_tr.shape) # 10000L, 1L, 28L, 28L (samples, channels, rows, columns)
    img_rows = x_tr.shape[2]
    img_cols = x_tr.shape[3]
    batch_size = 1
    n_epochs = 15

    model = Sequential()  # aka linear stacks of layers
    model.add(Convolution2D(30, 3, 3,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols),
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Convolution2D(100, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())                     # flattens the input: from (None, 64, 32, 32) to (None, 65536)

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(output_dim=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])

    history = model.fit(x_tr, y_tr, batch_size=batch_size, nb_epoch=n_epochs,
              verbose=1, validation_data=(x_va, y_va))

    return model


def eval_network(model, x_te, y_te):
    score = model.evaluate(x_te, y_te, verbose=0)
    print('Test score: %s', score[0])
    print('Test accuracy: %s', score[1])
    return score[0], score[1]


if __name__ == '__main__':
    x_tr = np.load('./train_x.npy')
    y_tr = np.load('./train_y.npy')
    x_va = np.load('./valid_x.npy')
    y_va = np.load('./valid_y.npy')

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
        print("Using validation set instead of the actual testset")
        x_te, y_te = x_va, y_va

    model = train_network(x_tr, y_tr, x_va, y_va)
    test_score, test_accuracy = eval_network(model, x_te, y_te)
