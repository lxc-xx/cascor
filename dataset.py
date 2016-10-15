import cPickle as pkl
import numpy as np
def load_two_spirals():
    pkl_path = "../data/two_spiral.pkl" 
    with open('./two_spiral.pkl','rb') as f: 
        data = pkl.load(f) 

    X_train = data['X']
    Y_train = data['Y']
    ys = []
    for y in Y_train:
        if y < 0:
            y = 0
        ys.append(y)
    Y_train = np.array(ys)

    return (X_train,Y_train)
