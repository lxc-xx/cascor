#!/usr/bin/env python
import cPickle as pkl
import numpy as np

def main():
    X = []
    Y = []
    with open('./two_spiral.data','r') as f:
        for line in f:
            seg = line.split()
            
            x1 = float(seg[0].rstrip(','))
            x2 = float(seg[1])

            if seg[-1].rstrip(';\n') is '+':
                label = 1
            else:
                label = -1
            X.append([x1,x2])
            Y.append(label)

    X = np.array(X) 
    Y = np.array(Y)

    print len(X)
    pkl.dump({'X':X, 'Y':Y}, open('two_spiral.pkl', 'wb'))
    
    #plt.scatter(X[:,0], X[:,1],c=Y)
    #plt.grid(True)
    #plt.show()

    #ax = plt.gca()
    #max_weight = 1.0
    #ax.patch.set_facecolor('gray') 
    #ax.set_aspect('equal', 'box') 
    #ax.xaxis.set_major_locator(plt.NullLocator()) 
    #ax.yaxis.set_major_locator(plt.NullLocator())

    #for [x,y], w in zip(X,Y):
    #    color = 'white' if w > 0 else 'black' 
    #    size = np.sqrt(np.abs(w)) 
    #    rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color) 
    #    ax.add_patch(rect)

    #ax.autoscale_view() 
    #ax.invert_yaxis()
    #plt.show()

    return


if __name__ == "__main__":
    main()
