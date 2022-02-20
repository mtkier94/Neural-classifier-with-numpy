import numpy as np
import matplotlib.pyplot as plt


def get_data(n: int,  r=1, bool_bias = True, seed = 42):
    '''
    Uniformly create data points in [-r,r]^2 and label them by 1 (point within r-radius circel) or 0 (otherwise).

    Inputs:
    -------
        n:  number of labeled data points to create

    Outputs:
    --------
        x,y: data points and labels; np.array of shapes (n,input) and (n,1) respectively
    '''
    np.random.seed(seed)
    # optional: adjust low and high value to e.g. look at a unit circle in [-2,2]^2
    x = np.random.uniform(low = -r, high = r, size = (n, 2))
    if bool_bias:
        # optional, but greatly improves the capacity of the neural network lateron
        # this simple trick adds a bias in the hidden layer of the neural network
        x = np.concatenate([x, np.ones((n,1))], axis = -1)
    y = (x[:,0]**2+x[:,1]**2 <= r**2).reshape((-1,1))

    return x,y


def visualize_data(x, y, r = 1):
    '''
    Create a scatter plot of data x and visualize their class-membership (either 0 or 1).
    We additionally display the ground truth, i.e. the decision boundary of the r-radius cirle.
    '''

    id = (y[:,0] == 0)
    plt.scatter(x[id, 0], x[id, 1], marker= '+', color = 'blue')
    plt.scatter(x[~id, 0], x[~id, 1], marker= '+', color = 'red')
    circle = plt.Circle((0,0), radius=r, linestyle = '--', alpha = 1, color = 'black', fill = False)
    plt.gca().add_patch(circle)
    plt.show()
    

def preprocess(x):
    '''
    Helper function: preprocessing of data. 
    Goal: take all elements of x to the power of 2

    Note: You may try and investigate other preprocessing schemes. To do so, simply adapt the return value below.
    '''
    assert( type(x) == type(np.array([0])))

    return np.square(x)