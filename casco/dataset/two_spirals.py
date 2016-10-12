import cPickle as pkl
import os.path
import numpy as np
def load_two_spirals():
    pkl_path = os.path.dirname(__file__) + "/two_spirals.pkl"
    with open(pkl_path,'rb') as f: 
        data = pkl.load(f) 

    X_train = np.mat(data['X']) 
    Y_train = np.mat(data['Y']).T

    return (X_train,Y_train)
