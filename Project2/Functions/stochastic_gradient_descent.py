import os
import sys
import pytest
import numba
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import functools
import time
from numba import jit
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


# Add the Functions/ directory to the python path so we can import the code 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'Functions'))

from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix
from Bootstrap import Bootstrap


def compute_square_loss(X, y, theta):
    loss = 0 #Initialize the average square loss
    
    m = len(y)
    loss = (1.0/m)*(np.linalg.norm((X.dot(theta) - y)) ** 2)
    return loss


def gradient_ridge(X, y, beta, lambda_):
    return 2*(np.dot(X.T, (X.dot(beta) - y))) + 2*lambda_*beta

def gradient_ols(X, y, beta):
    m = X.shape[0]
    
    grad = 2/m * X.T.dot(X.dot(beta) - y)
    
    return grad

def learning_schedule(t):
    t0, t1 = 5, 50
    return t0/(t+t1)


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    
    """
    Slices the data into batches. 
    Arguments: inputs - numpy array type 
               targets - numpy array
               batchsize - the number of slices
               shuffle - if True, shuffles the data 
    
    Output: a batch of the original data
    
    """
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]


###sgd
def SGD(X, y, learning_rate = 0.02, n_epochs = 100, lambda_ = 0.01, batch_size = 20, method = 'ols'):
    num_instances, num_features = X.shape[0], X.shape[1]
    beta = np.random.randn(num_features) ##initialize beta
    
    for epoch in range(n_epochs+1):
        
        for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
             
            X_batch, y_batch = batch
            
            # for i in range(batch_size):
            #     learning_rate = learning_schedule(n_epochs*epoch + i)
            
            if method == 'ols':
                gradient = gradient_ols(X_batch, y_batch, beta)
                beta = beta - learning_rate*gradient
            if method == 'ridge':
                gradient = gradient_ridge(X_batch, y_batch, beta, lambda_ = lambda_)
                beta = beta - learning_rate*gradient
                
    mse_ols_train = compute_square_loss(X, y, beta) 
    mse_ridge_train = compute_square_loss(X, y, beta) + lambda_*np.dot(beta.T, beta)
            
    return beta

def compute_test_mse(X_test, y_test, beta, lambda_ = 0.01):
    mse_ols_test = compute_square_loss(X_test, y_test, beta) 
    mse_ridge_test = compute_square_loss(X_test, y_test, beta) + lambda_*np.dot(beta.T, beta)
    return mse_ols_test, mse_ridge_test          
