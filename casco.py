#!/usr/bin/env python
import numpy as np
from sklearn.metrics import accuracy_score
from time import gmtime, strftime
import cPickle as pkl
import pprint
import theano
import theano.tensor as T
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import threading 

#from multiprocessing import Process, Value, Array, Manager
#from multiprocessing import Process, Lock
#from multiprocessing.sharedctypes import Value, Array

def neg_abs_cov(y_true, y_pred): 
    return -T.sum(T.abs_(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0)))

def neg_abs_cor(y_true, y_pred): 
    return -T.sum(T.abs_(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0)/T.sqrt(T.sum((y_true - T.mean(y_true,0, keepdims=True))*(y_true-T.mean(y_true,0,keepdims=True)),axis=0))/T.sqrt(T.sum((y_pred - T.mean(y_pred,0, keepdims=True))*(y_pred-T.mean(y_pred,0,keepdims=True)),axis=0))))

def hidden_candidate( x, y, rec, input_dim, output_dim, idx, batch_size=128, nb_epoch=2000, hidden_loss = 'cov'): 
    model=Sequential()
    model.add(Dense(output_dim, input_shape=(input_dim,)))

    model.add(Activation('tanh'))

    if hidden_loss is 'cov': 
        loss_func = neg_abs_cov
    elif hidden_loss is 'cor':
        loss_func = neg_abs_cor
    else:
        loss_func = hidden_loss

    opt = RMSprop()

    model.compile(loss=loss_func, optimizer=opt)
    fit_record = model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    loss = model.evaluate(x,y, verbose=0)
    rec[idx] = {'model':model, 'loss':loss}

class CascadeCorrelation(object):
    def __init__(self, hidden_num = 10):
        self.hidden_num = hidden_num
        self.hidden_models = []
        self.top_model = None
        self.hist_rec = []


    def fit(self, X_train, Y_train, outter_epoch = 2000,  hidden_epoch = 2000, batch_size = 128, pool_size = 4, verbose=0, show_history=False, top_loss = 'mse', hidden_loss = 'cov'):

        self.hidden_models = []
        self.hidden_cors = []
        warm_start = None
        hidden_feat = np.copy(X_train)
        self.hist_rec = []
        while True:
            model = Sequential()
            input_dim = hidden_feat.shape[1]
            output_dim = Y_train.shape[1] 
            model.add(Dense(output_dim, input_shape=(input_dim,))) 

            if warm_start:
                warm_start[0] = np.vstack((warm_start[0], np.random.uniform(-1,1,size=(output_dim,output_dim))))
                model.set_weights(warm_start)

            model.add(Activation('tanh')) 
            #model.add(Dropout(0.5))
            opt = RMSprop() 
            model.compile(loss=top_loss, optimizer=opt) 
            fit_record = model.fit(hidden_feat, Y_train, batch_size=batch_size, nb_epoch=outter_epoch, verbose=verbose, validation_data=(hidden_feat, Y_train)) 
            pred = model.predict(hidden_feat) 
            residual = Y_train - pred 
            accuracy = accuracy_score([1 if x > 0 else 0 for x in Y_train], [1 if x>0 else 0 for x in pred])
            loss = model.evaluate(hidden_feat, Y_train, verbose=0)
            warm_start = model.get_weights() 

            self.top_model = model
            if len(self.hidden_models) >= self.hidden_num:
                break

            threads = [] 
            pool= dict() 
            #print "Looking for good hidden candidate" 
            for i in range(pool_size): 
                t = threading.Thread(target=hidden_candidate, args=(hidden_feat, residual, pool, input_dim, output_dim, i, batch_size, hidden_epoch, hidden_loss)) 
                threads.append(t) 
                t.start()

            for t in threads: 
                t.join()

            chosen = min(pool.values(), key = lambda x: x['loss']) 

            #pprint.pprint(chosen)

            self.hist_rec.append({'loss':loss,'accuracy':accuracy, 'hidden_loss': chosen['loss']})
            #pprint.pprint(self.hist_rec)

            #print "hidden idx: " + str(len(self.hist_rec))

            if show_history:
                idx = len(self.hist_rec)
                rec = self.hist_rec[-1] 
                print "hidden_idx: " + str(idx) + ", accuracy: " + str(rec['accuracy']) + ", loss: " + str(rec['loss']) + ", hidden_loss: " + str(rec['hidden_loss'])


            hidden_model=chosen['model'] 
            hidden_pred = hidden_model.predict(hidden_feat) 
            hidden_feat = np.hstack((hidden_feat, hidden_pred)) 
            self.hidden_models.append(hidden_model)

        return

    def pred(self, feat):

        if self.top_model: 
            hidden_feat = self.map(feat) 
            return self.top_model.predict(hidden_feat)
        else:
            return None


    def map(self, feat):

        hidden_feat = np.copy(feat)

        for hidden_model in self.hidden_models:
            hidden_pred = hidden_model.predict(hidden_feat) 
            hidden_feat = np.hstack((hidden_feat, hidden_pred)) 
        
        return hidden_feat

from . import dataset
def main():

    X_train,Y_train = dataset.load_two_spirals()

    casco = CascadeCorrelation(2)
    casco.fit(X_train,Y_train, show_history=True)
    
    time_stamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    pkl.dump(casco, open('./casco_cov_mse_'+time_stamp+'.pkl','wb') )

    return


if __name__ == "__main__":
    main()
