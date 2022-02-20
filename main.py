#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Implementation of a neural network with one-hidden layer for classification whether 2d points lie within a circle of radius 1.
    No standard deep-learning library is used. All computations are based on Numpy.
"""

__author__      = "Mark Kiermayer"
__copyright__   = "GNU LGPLv3"
__credits__ =  "Christian Weiss"
__version__ = "1.0.1"
__maintainer__ = "Mark Kiermayer"
__email__ = "mark.kiermayer@hs-ruhrwest.de"
__status__ = "Educational Use"


from neuralNetwork import neunet
from dataClassifier import get_data, visualize_data, preprocess

if __name__=='__main__':

    n_train = 10000
    radius = 2 # positive number
    input = 3
    x_train, y_train = get_data(n_train, r = radius, bool_bias= (input ==3))

    print('Shape of training data x,y: ')
    print(x_train.shape, y_train.shape)

    # un-comment to display visualization
    # visualize_data(x_train, y_train, r=radius)

    # initialize model
    # optional: adjust width or (hidden-layer) activation function
    model = neunet(input = input, no_neurons= 10, activation = 'sigmoid')
    # start training the model
    model.training(x_train, y_train, epochs = 30, learning_rate=0.1)

    visualize_data(x_train, model.feedforward(x_train)['a_o']>=0.5, r=radius)


    # Now again, but with pre-processed data

    # initialize model
    # optional: adjust width or (hidden-layer) activation function
    model_pp = neunet(input = input, no_neurons= 10, activation = 'sigmoid')
    # start training the model
    x_train_processed = preprocess(x_train)
    model_pp.training(x_train_processed, y_train, epochs = 30, learning_rate=0.01)

    # visualize new model
    visualize_data(x_train, model_pp.predict_class(preprocess(x_train)), r=radius)