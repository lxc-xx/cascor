import cPickle as pkl
import numpy as np
def load_two_spirals():
    pkl_path = "../data/two_spiral.pkl" 
    with open('./two_spiral.pkl','rb') as f: 
        data = pkl.load(f) 

    X_train = np.mat(data['X']) 
    Y_train = np.mat(data['Y']).T

    return (X_train,Y_train)
