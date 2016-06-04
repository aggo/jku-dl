#!/usr/bin/env python
import sys
import os
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers


def train_network(x_tr, y_tr, x_va, y_va):
    batch_size = 128
    n_epochs = 10

    model = Sequential()
    model.add(Dense(output_dim=128, input_dim=x_tr.shape[-1], activation='relu'))

    model.add(Dense(output_dim=128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=10, activation='softmax'))

    opt = optimizers.SGD(lr=0.01, momentum=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(x_tr, y_tr, batch_size=batch_size, nb_epoch=n_epochs,
              verbose=1, validation_data=(x_va, y_va))
    return model


def eval_network(model, x_te, y_te):
    score = model.evaluate(x_te, y_te, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    x_tr = np.load('./train_x.npy')
    y_tr = np.load('./train_y.npy')
    x_va = np.load('./valid_x.npy')
    y_va = np.load('./valid_y.npy')

    # Reshape images from 1x28x28 pixels to flat vectors of 784
    # Remove this line if you use convolutions
    x_tr = x_tr.reshape(-1, 784)
    x_va = x_va.reshape(-1, 784)

    n_classes = 10
    y_tr = np_utils.to_categorical(y_tr, n_classes)
    y_va = np_utils.to_categorical(y_va, n_classes)

    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]):
            x_te = np.load(sys.argv[1])
            y_te = np.load(sys.argv[2])
            y_te = np_utils.to_categorical(y_te, n_classes)
            x_te = x_te.reshape(-1, 784)   # remove this line if you use convs
        else:
            print("Invalid testset files passed")
            sys.exit(1)
    else:
        print("Using validation set instead of the actual testset")
        x_te, y_te = x_va, y_va

    model = train_network(x_tr, y_tr, x_va, y_va)
    eval_network(model, x_te, y_te)
