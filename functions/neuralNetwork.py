import numpy as np

def sigmoid(x):
    '''
    Activation function, which can optionally be used later on.
    '''
    return 1 / (1 + np.exp(-x))

def phi(x, actv: str):
    '''
    Activation functions.
    '''

    if actv == 'relu':
        return x*(x>0)
    elif actv == 'linear':
        return x
    elif actv == 'tanh':
        return np.tanh(x)
    elif actv == 'sigmoid':
        return sigmoid(x)
    else:
        raise ValueError('Unknown activation function!')


def dphi(x, actv: str):
    '''
    Derivative of activation functions.
    '''

    if actv == 'relu':
        return 0.0+(x>0)
    elif actv == 'linear':
        return 1
    elif actv == 'tanh':
        return 1-np.tanh(x)**2
    elif actv == 'sigmoid':
        return sigmoid(x)*(1-sigmoid(x))
    else:
        raise ValueError('Unknown activation function!')


class neunet:

    def __init__(self, input: int, no_neurons: int, activation = 'relu', activation_output = 'sigmoid'):
        '''
        Initialize the architecture of the neural network with one hidden layer.
        Sketch of architecture:
            input -> hidden layer with no_neurons and activation function -> output (with sigmoid activation for classification)

        Inputs:
        -------
            no_neurons: number of neurons in hidden layer
            activation: activation function in the hidden layer
            input:      number of input units
        '''

        # check for valid inputs - derivative only for the following four functions implemented
        assert(activation in ['relu', 'linear', 'sigmoid', 'tanh'])
        assert(input in [2,3])

        # set object-attributes
        self.no_neurons = no_neurons
        self.hidden_activation = activation
        self.output_activation = activation_output # can be changed to vales in ['relu', 'linear', 'sigmoid', 'tanh']
        self.input = input
        self.output = 1

        # initialize weights
        self.init_weights()

    def init_weights(self):
        '''
        Initialize weights for a given network architecture.
        '''

        # Gaussian initialisation; various alternatives can be found e.g. https://www.tensorflow.org/api_docs/python/tf/keras/initializers 
        self.W0 = np.random.normal(size = (self.input, self.no_neurons))
        self.W1 = np.random.normal(size = (self.no_neurons, self.output))

        # uncomment this alternative to observe how the identical and symmetrical weights prevent proper training
        # self.W0 = np.ones((self.input, self.no_neurons))
        # self.W1 = np.ones((self.no_neurons, 1))



    def feedforward(self, x):
        '''
        Take (potentially a batch of) input x and propagate it forward through the network architecture.

        Input:
        ------
            x:  (batch of) inputs of shape (batch, self.input)

        Output:
        -------
            y_hat:  (batch of) predictions with shape (batch, self.output)
                    in the context of classification: y_hat are values in [0,1], indicating the confidence for class membership
        '''

        # avoid 1d-array -> reshape to 2d-array with batch-size 1
        if len(x.shape) == 1:
            x = x.reshape((1,-1))

        # check whether input consistent with architecture
        assert(x.shape[1] == self.input)
        
        # apply weight matrix for hidden layer
        net_h = np.dot(x, self.W0)
        # apply selected activation function for hidden layer
        a_h = phi(net_h, actv = self.hidden_activation)
        # apply the final weight matrix and a sigmoid activation (for classification)
        net_o = np.dot(a_h, self.W1)
        a_o = phi(net_o, actv = self.output_activation)

        # return dictionary of values which will be re-used during backpropagation
        return {'a_o': a_o, 'net_o': net_o, 'a_h': a_h, 'net_h': net_h, 'net_i': x}

    def backpropagation(self, t, cache):
        '''
        Compute delta_j values for all neurons to eventually obtain err_ij = dErr/dw_ij = delta_j*a_i.
        This function utilizes values that were computed during the forward-pass by the function feedforward (-> 'cache')

        Note: For now this functions will be run with a batch size (batch) of 1.

        Inputs:
        --------
            y:      true target values
            cache:  dictionary of values cached during forward-propagation, see output of feedforward function
                    use saved values to speed up computation

        Outputs:
        --------
            delta_o:    delta-value of output unit
            delta_h:    delta-values of hidden units
        '''

        # output neuron: \phi'(net_j)(y_j-t_j)  (slide 95)
        # shape delta_o: (1,batch)
        delta_o = np.transpose(dphi(cache['net_o'], actv = self.output_activation)*(cache['a_o']-t))


        # hidden neuron: \phi'(netj) \sum_k \delta_k w_jk
        # 1) hidden layer - output layer
        # shape delta_h: (self.no_neurons, batch)
        delta_h = np.transpose(dphi(cache['net_h'], actv = self.hidden_activation))*np.dot(self.W1, delta_o)


        # compute dErr/dw_ij for all weights
        # 1) hidden layer - output layer
        err_o = np.dot(delta_o, cache['a_h']).T
        # 2) input layer - hidden layer
        err_h = np.dot(delta_h, cache['net_i']).T

        return err_o, err_h

    def training(self, x_train, y_train, epochs = 10, learning_rate = 0.001):
        '''
        Combine forward- and back-propagation and calibrate the parameters of the network.

        Input:
        ------
            x,y:    np.arrays of input data incl. labels
            epochs: number of epochs we want to train
            learning_rate:  gradient descent parameter

        Outputs:
        --------
            None; model parameters will be adjusted in-place
        '''
        n_data = len(x_train)
        for e in range(epochs):
            print(f'Training: epoch {1+e}/{epochs}')
            pred = self.feedforward(x_train)['a_o']
            print(f'\t mse-loss: {np.mean((pred-y_train)**2): .6f}, accuracy: {np.mean((pred>=0.5)==y_train): .4f}')
            for k in range(n_data):
                # forward-pass
                cache = self.feedforward(x_train[k])
                # backpropagation
                err_o, err_h = self.backpropagation(t = y_train[k], cache= cache)
                # adjust weight matrices
                self.W0 -= learning_rate*err_h
                self.W1 -= learning_rate*err_o

    
    def predict(self, x):
        '''
        Returns the output-value of the network, without any cached values.
        Useful for running model on test data.
        '''
        return self.feedforward(x)['a_o']

    def predict_class(self, x, threshold = 0.5):
        '''
        Returns the class that network assigns to data x.
        Useful for running model on test data.
        '''
        return self.predict(x)>= threshold