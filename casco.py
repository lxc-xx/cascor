#!/usr/bin/env python
import os
import numpy as np
from sklearn.metrics import accuracy_score
from time import gmtime, strftime
import cPickle as pkl
import pprint
from time import gmtime, strftime

import keras
from keras import backend as T
from keras import callbacks
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Input, merge
import dataset

from sklearn.cross_validation import train_test_split

#from multiprocessing import Process, Value, Array, Manager
#from multiprocessing import Process, Lock
#from multiprocessing.sharedctypes import Value, Array

def neg_abs_cov(y_true, y_pred): 
    return -T.sum(T.abs(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0)))

def neg_abs_cor(y_true, y_pred): 
    return -T.sum(T.abs(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0)/T.sqrt(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_true-T.mean(y_true,0,keepdims=True)),axis=0))/T.sqrt(T.sum((y_pred - T.mean(y_pred,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0))))

def make_candidates_model(input_shape, nb_candidates, nb_positions, node_type = 'Dense', hidden_loss = 'cov'):
    print input_shape
    cand_input = Input(shape=input_shape)

    cand_outputs = []
    cand_layers = []

    for out_idx in range(nb_candidates):
        if node_type is "SimpleRNN": 
            layer = SimpleRNN(1, activation='tanh')
        elif node_type is "GRU":
            layer = GRU(1, activation='tanh')
        elif node_type is "LSTM":
            layer = LSTM(1, activation='tanh')
        else:
            layer = Dense(1, activation='tanh')

        out = layer(cand_input)
        cand_outputs.append(out)
        cand_layers.append(layer)

    if hidden_loss is 'cov': 
        loss_func = neg_abs_cov
    elif hidden_loss is 'cor':
        loss_func = neg_abs_cor
    else:
        loss_func = hidden_loss

    model = Model(input=cand_input, output=cand_outputs)
    model.compile(loss=loss_func, optimizer='adam')

    return model, cand_layers



def select_candidate( x, y, batch_size=128, nb_epoch=200, hidden_loss = 'cov', nb_candidates = 5, nb_positions = 1, criteria = 'best', node_type='Dense'): 
    assert(nb_positions<=nb_candidates and nb_positions > 0)
    input_shape = tuple(x.shape[1:])


    model, cand_layers = make_candidates_model(input_shape, nb_candidates, nb_positions, node_type=node_type, hidden_loss = hidden_loss)
    fit_record = model.fit(x, [y]*nb_candidates,validation_data=(x, [y]*nb_candidates), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1 ,callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')])
    losses = model.evaluate(x,[y]*nb_candidates, verbose=0)

    layer_loss = sorted(zip(cand_layers, losses[1:]), key = lambda x: x[1])

    if criteria is "random": 
        random.shuffle(layer_loss)

    weights = [np.hstack([x[0].get_weights()[0] for x in layer_loss[:nb_positions]]), np.hstack([x[0].get_weights()[1] for x in layer_loss[:nb_positions]])]

    return {'weights':weights,'model':model, 'layer_loss':zip(cand_layers, losses[1:])}
    #rec[idx] = {'model':model, 'loss':loss}

class CascadeCorrelation(object):
    def __init__(self, nb_hidden_layers = 10, positions_per_layer = 1, base_layers=None):
        self.nb_hidden_layers = nb_hidden_layers
        self.positions_per_layer=positions_per_layer
        self.model = None
        self.history = {}
        self.hidden_weights = []
        self.input_node = None
        self.output_node = None
        self.base_layers = base_layers

    def fit(self, X_train, Y_train, validation_data = None, outter_epoch = 20,  hidden_epoch = 20, batch_size = 128, nb_candidates = 5, verbose=1, show_history=False, top_loss = 'categorical_crossentropy', hidden_loss = 'cov', hidden_train_ratio = -1, tenure = True, dropout_rate = -1, use_warm_start = False, save_history = False, history_folder = "/temp/" ):

        train_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

        if save_history and (not os.path.isfile(history_folder)): 
            os.mkdir(history_folder)

        if not validation_data:
            validation_data = (X_train, Y_train)

        self.history = {}
        self.hidden_weights = []
        input_shape = X_train.shape[1:]
        output_dim = Y_train.shape[1] 

        X_test, Y_test = validation_data

        if hidden_train_ratio > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=hidden_train_ratio,random_state=43)
            hidden_feat_val = X_val
        else:
            X_val = X_train
            Y_val = Y_train
            hidden_feat_val = X_train

        #hidden_feat = np.copy(X_train)
        #hidden_feat_test = np.copy(X_test)
        #hidden_feat_val = np.copy(X_val)
        
        warm_start = None
        self.input_node = Input(shape=input_shape,name='Input_Feature')
        feat_node = self.input_node
        if self.base_layers:
            self.input_node = self.base_layers[0].input
            feat_node = base_layers[-1].output

            init_mapper = Model(input=self.input_node, output=feat_node)
            hidden_feat_val = init_mapper.predict(X_val)

        if dropout_rate > 0:
            feat_node = Dropout(dropout_rate)(feat_node)

        self.output_node = None

        train_step = 0
        while True:
            #in_node = Input(shape=(hidden_feat_val.shape[1],))
            pred_layer = Dense(output_dim, activation='softmax')
            pred_layer.name = "Output_Prediction"
            pred_node = pred_layer(feat_node)

            self.model = Model(input=self.input_node, output=pred_node)

            if use_warm_start and warm_start: 
                warm_start[0] = np.vstack((warm_start[0], np.random.uniform(-1,1,size=(self.positions_per_layer,output_dim)))) 
                pred_layer.set_weights(warm_start)

            self.model.compile(optimizer='rmsprop', loss=top_loss, metrics=['accuracy'])
            fit_history = self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=outter_epoch, verbose=verbose, validation_data=(X_test, Y_test), callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')])
            train_step += 1

            #save history
            if save_history:
                pkl.dump(fit_history, open(os.path.join(history_folder, train_stamp + str(train_step) + ".pkl", 'w')))

            #save the warm start weights
            warm_start = pred_layer.get_weights()

            #Build the output node
            #output_layer = Dense(output_dim = output_dim, input_dim = input_dim, activation='softmax')
            #self.output_node = output_layer(feat_node)
            #output_layer.set_weights(pred_layer.get_weights())
            #output_layer.name = "Output_Prediction"
            #self.model=Model(input = self.input_node, output=self.output_node)

            if len(self.hidden_weights) >= self.nb_hidden_layers:
                break

            #Get residual
            pred = self.model.predict(X_val) 
            residual = Y_val - pred 

            #Select candidate
            candidate = select_candidate(hidden_feat_val, residual, nb_epoch=hidden_epoch, nb_candidates=nb_candidates, hidden_loss=hidden_loss, nb_positions=self.positions_per_layer)

            #Cache hidden featuure
            cache_in_node = Input(shape=hidden_feat_val.shape[1:])
            cache_hidden_layer = Dense(self.positions_per_layer, activation='tanh')
            cache_hidden_node = cache_hidden_layer(cache_in_node)
            hidden_model = Model(input=cache_in_node, output=cache_hidden_node)

            #hidden_pred = hidden_model.predict(hidden_feat)
            #hidden_feat = np.hstack((hidden_feat, hidden_pred))

            #hidden_pred_test = hidden_model.predict(hidden_feat_test)
            #hidden_feat_test = np.hstack((hidden_feat_test, hidden_pred_test))

            hidden_pred_val = hidden_model.predict(hidden_feat_val)
            hidden_feat_val = np.hstack((hidden_feat_val, hidden_pred_val))

            #Candidate tenure
            hidden_layer = Dense(self.positions_per_layer, activation='tanh')
            hidden_node = hidden_layer(feat_node)
            hidden_layer.set_weights(candidate['weights'])
            self.hidden_weights.append(candidate['weights'])
            hidden_layer.trainable = (not tenure)
            hidden_layer.name = "Hidden_" + str(len(self.hidden_weights))

            #update the feat node
            feat_node = merge([feat_node, hidden_node], mode='concat')
            if dropout_rate > 0:
                feat_node = Dropout(dropout_rate)(feat_node)
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
    #X_train, X_test = map(lambda x:x.repeat(2, axis=1).repeat(2, axis=2), [X_train, X_test])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes=10

    X_train=X_train.reshape(X_train.shape[0],-1)
    X_test=X_test.reshape(X_test.shape[0],-1)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    from keras.models import load_model
    #model = load_model('./revisit/vgg_2.h5')
    #base_model = Model(input=model.layers[0].input, output=model.get_layer('flatten_2').output)
    #X_test = feat_mapper.predict(X_test)
    #X_train = feat_mapper.predict(X_train)
    
    #base_model.trainable = False
    #casco = CascadeCorrelation(nb_hidden_layers = 100, positions_per_layer = 10, base_model = base_model)
    casco = CascadeCorrelation(nb_hidden_layers = 100, positions_per_layer = 2)
    casco.fit(X_train,Y_train, validation_data=(X_test,Y_test), show_history=True, nb_candidates=5, tenure=True, outter_epoch = 20,  hidden_epoch = 5 , save_history = True, history_folder = "./log/" )
