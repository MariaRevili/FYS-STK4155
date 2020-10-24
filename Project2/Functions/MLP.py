import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import os
from numba import jit



class Layer:
    """

    Represents a layer (hidden or output) in our neural network.

    """
    
    def __init__(self, n_input, n_neurons, activation=None, alpha = 0.01, lam = 0.1):
        """

        :param int n_input: The input size (coming from the input layer or a previous hidden layer)

        :param int n_neurons: The number of neurons in this layer.

        :param str activation: The activation function to use (if any).

        :param weights: The layer's weights.

        :param bias: The layer's bias.

        """
        self.activation = activation
        self.alpha = alpha
        self.lam = lam
        self.last_activation = None
        self.error = None
        self.delta = None
        # self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)
        # self.activation = activation
        # self.bias = bias if bias is not None else np.random.rand(n_neurons)
        
        np.random.seed(1921)
        # Xavier initializations (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
        if self.activation == 'sigmoid' :
            r_inputs = np.sqrt(6.0 / (n_input + n_neurons))
            self.weights = np.random.uniform(-r_inputs, r_inputs, size=(n_input, n_neurons))
            #self.weights = np.random.rand(n_input, n_neurons)
            self.bias = np.random.rand(n_neurons)
            #self.v_reg = np.random.rand(n_neurons, 1)
            self.v_reg = np.random.uniform(-(6/(n_neurons+1)), (6/(n_neurons+1)), size=(n_neurons, 1))
            self.bias_reg = np.random.rand(1)
        
        elif self.activation == 'tanh' :
            r_inputs = 4.0 * np.sqrt(6.0 / (n_input + n_neurons))
            self.weights = np.random.uniform(-r_inputs, r_inputs, size=(n_input, n_neurons))
            #self.weights = np.random.rand(n_input, n_neurons)
            self.v_reg = np.random.uniform(-4.0 * np.sqrt(6.0 / (n_neurons+1)), 4.0 * np.sqrt(6.0 / (n_neurons+1)), size=(n_neurons, 1))
            #self.v_reg = np.random.rand(n_neurons, 1)
            #self.bias = np.zeros(shape=(n_neurons))
            self.bias = np.random.rand(n_neurons)
            self.bias_reg = np.random.rand(1) ##output bias

        # He initializations (https://arxiv.org/pdf/1502.01852.pdf).
        elif self.activation == 'relu' or self.activation == 'leaky_relu' or self.activation == 'elu' :
            self.weights = np.random.normal(size=(n_input, n_neurons)) * np.sqrt(2.0 / n_input)
            #self.weights = np.random.rand(n_input, n_neurons)
            #self.v_reg = np.random.rand(n_neurons, 1)
            self.v_reg = np.random.normal(size=(n_neurons,1)) * np.sqrt(2.0 / n_neurons)
            self.bias = np.random.rand(n_neurons)
            self.bias_reg = np.random.rand(1)

        else :
            self.weights = np.random.normal(size=(n_input, n_neurons))
            #self.weights = np.random.rand(n_input, n_neurons)
            #self.v_reg = np.random.rand(n_neurons, 1)
            self.v_reg = np.random.normal(size=(n_neurons, 1))
            self.bias = np.random.rand(n_neurons)
            self.bias_reg = np.random.rand(1)
            
    def activate(self, x):
        """

        Calculates the dot product of this layer.

        :param x: The input.

        :return: The result.

        """

        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation
    
    def _apply_activation(self, r):
        """

        Applies the chosen activation function (if any).

        :param r: The normal value.

        :return: The activated value.

        """

        # In case no activation function was chosen

        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return np.tanh(r)

        if self.activation == 'sigmoid':
            return self._sigmoid(r)
        
        if self.activation == 'relu':
            return self._relu(r)
        
        if self.activation == 'leaky_relu':
            return self._leakyrelu(r)
        
        if self.activation == 'elu':
            return self._elu(r)
        
        if self.activation == 'softmax':
            return self._softmax(r)
        
        if self.activation == 'identity':
            return self._identity(r)
            
        return r
    
    def apply_activation_derivative(self, r):
        """

        Applies the derivative of the activation function (if any).

        :param r: The normal value.

        :return: The "derived" value.

        """

        # We use 'r' directly here because its already activated, the only values that

        # are used in this function are the last activations that were saved.
        
        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)
        
        if self.activation == 'relu':
            
            r[r > 0] = self.lam
            return r
        
        if self.activation == 'leaky_relu':
            
            r[r > 0] = self.lam
            r[r <= 0] = self.lam * self.alpha
            return r
 
        if self.activation == 'identity':
            return 1
        
        if self.activation == 'elu':
      
            r[r > 0] = 1
            r[r <= 0] = r[r <= 0] + self.alpha
    
        
        return r
    
    
    def _sigmoid(self, x):
            return 1.0/ (1.0 + np.exp(-x))
    
    
    def _tanh(self, x):
        
        return np.tanh(x)
    
    
    def _relu(self, x):
    
        x = self.lam * x
        x[x <= 0] = 0

        return x
    

    def _leakyrelu(self, x) :
        
        x = self.lam * x
        x[x <= 0] = self.alpha * x[x <= 0]

        return x

    def _identity(self, x) :  ##linear activation function
        return x  
    
    
    def _elu(self, x) :
        neg = x<0.0
        x[neg] = self.alpha * (np.exp(x[neg]) - 1.0)
        
        return x 

    def _softmax(self, x) :
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0, keepdims=True) ## sum along the column
 
 
 
class NeuralNetwork:
    """

    Represents a neural network.

    """

    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        """

        Adds a layer to the neural network.

        :param Layer layer: The layer to add.

        """

        self._layers.append(layer)
        
        
    def feed_forward(self, X):
        """

        Feed forward the input through the layers.

        :param X: The input values.

        :return: The result.

        """

        for layer in self._layers:
            X = layer.activate(X)

        return X

        """

        N.B: Having a sigmoid activation in the output layer can be interpreted

        as expecting probabilities as outputs.

        W'll need to choose a winning class, this is usually done by choosing the

        index of the biggest probability.

        """
    def predict(self, X, net_type = 'regression', n_neurons = 3):
        """

        Predicts a class (or classes).

        :param X: The input values.

        :return: The predictions.

         """

        ff = self.feed_forward(X)
        
        if net_type == 'classification':
            
             # One row

            if ff.ndim == 1:
                pred = np.argmax(ff)
            else: 
                pred = np.argmax(ff, axis = 1)
                
        if  net_type == 'regression':
            
            pred = ff
        
        return pred
                

        # # Multiple rows

        # return np.argmax(ff, axis=1)
    
    
    def backpropagation(self, X, y, learning_rate, lmbd, net_type = 'classification'):
        """

        Performs the backward propagation algorithm and updates the layers weights.

        :param X: The input values.

        :param y: The target values.

        :param float learning_rate: The learning rate (between 0 and 1).

        """
        ntarget = y.size

        # Feed forward for the output

        output = self.feed_forward(X)
        
        # Loop over the layers backward

        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
                       
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
              
                
         # Update the weights

        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)

            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            
            layer.weights = layer.weights + layer.delta * input_to_use.T * learning_rate
            
            if lmbd > 0: ###adding L2 regularization
                
                layer.weights = layer.weights*(1-lmbd*learning_rate) + layer.delta * input_to_use.T * learning_rate
                
            layer.bias = layer.bias + layer.delta * learning_rate 
            
                        
            
    def train(self, X, y, learning_rate, max_epochs, net_type = 'regression', lmbd = 0):
        """

        Trains the neural network using backpropagation.

        :param X: The input values.

        :param y: The target values.

        :param float learning_rate: The learning rate (between 0 and 1).

        :param int max_epochs: The maximum number of epochs (cycles).

        :return: The list of calculated MSE errors.

        """
    
        mses = []
        
        for i in range(max_epochs):
            for j in range(len(X)): ##len(X) is rows
                 self.backpropagation(X[j], y[j], learning_rate, lmbd)
    
            # if i % 10 == 0: #At every 10th epoch, we will print out the Mean Squared Error and save it in mses which we will return at the end.
            #     nn = NeuralNetwork()
            #     mse = np.mean(np.square(y - nn.feed_forward(X)))
            #     mses.append(mse)
            #     print('Epoch: #%s, MSE: %f' % (i, float(mse)))
            # return mses
            
                    
    
    def MSE(self, y_pred, y_true):
        return (1/len(y_true))*np.sum((y_pred - y_true)**2)
    
       
    def accuracy(self, y_pred, y_true):
        """

        Calculates the accuracy between the predicted labels and true labels.

        :param y_pred: The predicted labels.

        :param y_true: The true labels.

        :return: The calculated accuracy.

        """

        return (y_pred == y_true).mean()
       
    def cal_err(self, y_pred, y_true, costf):
        
        if costf == "squared-error":
            err = np.sum((y_pred - y_true)**2)
        elif costf == "MSE":
            err = (1/len(y_true))*np.sum((y_pred - y_true)**2)

        return err
    
    def confusion_table(self, y_pred, y_true):
        
        conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        return conf
    
    def cal_r2(self, y_pred, y_true):
    
        mu = np.mean(y_true)
        SS_tot = np.sum((y_true - mu)**2)
        SS_res = np.sum((y_true - y_pred)**2)
        
        r2 = 1 - (SS_res/SS_tot)

        return r2
    
    



