#!/usr/bin/env python
import numpy as np
from sklearn.metrics import accuracy_score
from time import gmtime, strftime
import cPickle as pkl
import pprint

import keras
from keras import backend as T
from keras import callbacks
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Input, merge
import dataset

#from multiprocessing import Process, Value, Array, Manager
#from multiprocessing import Process, Lock
#from multiprocessing.sharedctypes import Value, Array

def neg_abs_cov(y_true, y_pred): 
    return -T.sum(T.abs(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0)))

def neg_abs_cor(y_true, y_pred): 
    return -T.sum(T.abs(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0)/T.sqrt(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_true-T.mean(y_true,0,keepdims=True)),axis=0))/T.sqrt(T.sum((y_pred - T.mean(y_pred,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0))))

def select_candidate( x, y, batch_size=128, nb_epoch=200, hidden_loss = 'cov', nb_candidates = 5, nb_positions = 1, criteria = 'best'): 
    assert(nb_positions<=nb_candidates)
    cand_input = Input(shape=(x.shape[1],))

    cand_outputs = []
    cand_layers = []

    for out_idx in range(nb_candidates):
        layer = Dense(1, activation='tanh')
        out = layer(cand_input)
        cand_outputs.append(out)
        cand_layers.append(layer)

    model = Model(input=cand_input, output=cand_outputs)


    if hidden_loss is 'cov': 
        loss_func = neg_abs_cov
    elif hidden_loss is 'cor':
        loss_func = neg_abs_cor
    else:
        loss_func = hidden_loss

    opt = RMSprop()

    model.compile(loss=loss_func, optimizer=opt)
    fit_record = model.fit(x, [y]*nb_candidates,validation_data=(x, [y]*nb_candidates), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1 ,callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')])
    losses = model.evaluate(x,[y]*nb_candidates, verbose=0)

    layer_loss = sorted(zip(cand_layers, losses[1:]), key = lambda x: x[1])

    if criteria is "random": 
        random.shuffle(layer_loss)

    weights = [np.hstack([x[0].get_weights()[0] for x in layer_loss[:nb_positions]]), np.hstack([x[0].get_weights()[1] for x in layer_loss[:nb_positions]])]

    return {'weights':weights,'model':model, 'layer_loss':zip(cand_layers, losses[1:])}
    #rec[idx] = {'model':model, 'loss':loss}

class CascadeCorrelation(object):
    def __init__(self, nb_hidden_layers = 10, positions_per_layer = 1):
        self.nb_hidden_layers = nb_hidden_layers
        self.positions_per_layer=positions_per_layer
        self.model = None
        self.history = {}
        self.hidden_weights = []
        self.input_node = None
        self.output_node = None

    def fit(self, X_train, Y_train, validation_data = None, outter_epoch = 20,  hidden_epoch = 20, batch_size = 128, nb_candidates = 5, verbose=1, show_history=False, top_loss = 'categorical_crossentropy', hidden_loss = 'cov'):

        if not validation_data:
            validation_data = (X_train, Y_train)

        X_test, Y_test = validation_data

        self.history = {}
        self.hidden_weights = []
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1] 
        hidden_feat = np.copy(X_train)
        hidden_feat_test = np.copy(X_test)
        
        warm_start = None
        self.input_node = Input(shape=(input_dim,),name='Input_Feature')
        feat_node = self.input_node
        self.output_node = None

        while True:
            in_node = Input(shape=(hidden_feat.shape[1],))
            pred_layer = Dense(output_dim, activation='softmax')
            pred_node = pred_layer(in_node)

            if warm_start: 
                warm_start[0] = np.vstack((warm_start[0], np.random.uniform(-1,1,size=(self.positions_per_layer,output_dim)))) 
                pred_layer.set_weights(warm_start)

            model = Model(input=in_node, output=pred_node)
            model.compile(optimizer='rmsprop', loss=top_loss, metrics=['accuracy'])
            fit_history = model.fit(hidden_feat, Y_train, batch_size=batch_size, nb_epoch=outter_epoch, verbose=verbose, validation_data=(hidden_feat_test, Y_test), callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')])

            #save the warm start weights
            warm_start = pred_layer.get_weights()

            #Build the output node
            output_layer = Dense(output_dim = output_dim, input_dim = input_dim, activation='softmax')
            self.output_node = output_layer(feat_node)
            output_layer.set_weights(pred_layer.get_weights())
            output_layer.name = "Output_Prediction"
            self.model=Model(input = self.input_node, output=self.output_node)

            if len(self.hidden_weights) >= self.nb_hidden_layers:
                break

            #Get residual
            pred = model.predict(hidden_feat) 
            residual = Y_train - pred 

            #Select candidate
            candidate = select_candidate(hidden_feat, residual, nb_epoch=hidden_epoch, nb_candidates=nb_candidates, hidden_loss=hidden_loss, nb_positions=self.positions_per_layer)

            #Cache hidden featuure
            cache_in_node = Input(shape=(hidden_feat.shape[1],))
            cache_hidden_layer = Dense(self.positions_per_layer, activation='tanh')
            cache_hidden_node = cache_hidden_layer(cache_in_node)
            hidden_model = Model(input=cache_in_node, output=cache_hidden_node)

            hidden_pred = hidden_model.predict(hidden_feat)
            hidden_feat = np.hstack((hidden_feat, hidden_pred))

            hidden_pred_test = hidden_model.predict(hidden_feat_test)
            hidden_feat_test = np.hstack((hidden_feat_test, hidden_pred_test))

            #Candidate tenure
            hidden_layer = Dense(self.positions_per_layer, activation='tanh')
            hidden_node = hidden_layer(feat_node)
            hidden_layer.set_weights(candidate['weights'])
            self.hidden_weights.append(candidate['weights'])
            hidden_layer.trainable = False
            hidden_layer.name = "Hidden_" + str(len(self.hidden_weights))

            #update the feat node
            feat_node = merge([feat_node, hidden_node], mode='concat')
            #print feat_node

    def pred(self, x):

        if self.model: 
            return self.model.predict(x)
        else:
            return None


if __name__ == "__main__":
    #X_train,Y_train = dataset.load_two_spirals()
    #Y_train = np_utils.to_categorical(Y_train, 2)
    
    from keras.datasets import cifar10
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = map(lambda x:x.repeat(2, axis=1).repeat(2, axis=2), [X_train, X_test])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes=10
    X_train=X_train.reshape(X_train.shape[0],-1)
    X_test=X_test.reshape(X_test.shape[0],-1)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    casco = CascadeCorrelation(nb_hidden_layers = 10, positions_per_layer = 10)
    casco.fit(X_train,Y_train, validation_data=(X_test,Y_test), show_history=True, nb_candidates=20)
