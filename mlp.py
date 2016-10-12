#!/usr/bin/env python
import numpy as np
import cPickle as pkl
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


def main():

    with open('./two_spiral.pkl','rb') as f:
        data = pkl.load(f)

    batch_size = 100
    nb_classes = 2
    nb_epoch = 1000

    X_train = data['X']
    Y_train = data['Y']

    Xmax = np.max(np.abs(X_train))

    #X_train = X_train/Xmax
    Y_train = np_utils.to_categorical(Y_train, nb_classes)

    model = Sequential()
    model.add(Dense(10, input_shape=(2,)))
    #model.add(Activation('sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    opt = RMSprop()
    #opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    #opt = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
    #opt = keras.optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_train, Y_train))
    #score = model.evaluate(X_train, Y_train, show_accuracy=True, verbose=0)
    Y_pred = model.predict(X_train)
    print Y_pred
    print Y_train - Y_pred


    return


if __name__ == "__main__":
    main()
