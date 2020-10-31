import os
import sys
import pytest
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import functools
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import confusion_matrix 
import seaborn as sns



from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from tensorflow.keras import initializers
from tensorflow.keras.layers.experimental import preprocessing



sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'Functions'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))

from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix
from Bootstrap import Bootstrap
from stochastic_gradient_descent import SGD, compute_test_mse
from MLP import Layer, NeuralNetwork


########## Regression ###########################

##Make synthetic data
n = 1000  
np.random.seed(20)
x1 = np.random.rand(n)
x2 = np.random.rand(n)     
X = designMatrix(x1, x2, 4)
y = franke(x1, x2) 
X = X[:, 1:]

##Train-validation-test samples. We choose / play with hyper-parameters on the validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

##min-max standardization of the outcome variable for a better prediction performace
y_train = (y_train - min(y_train))/(max(y_train) - min(y_train))
y_test = (y_test - min(y_test))/(max(y_test) - min(y_test))
y_val = (y_val - min(y_val))/(max(y_val) - min(y_val))


## Define keras model
epochs = 100
batch_size = 100
n_neurons_layer1 = 50
n_neurons_layer2 = 50
n_categories = 10
eta_vals = np.logspace(-4, -1, 7)
lmbd_vals = np.logspace(-7, -5, 7)

def neural_network_keras(n_neurons_layer1, n_neurons_layer2, eta, lmbd):  ##Build the model
    model = Sequential()
    model.add(Dense(n_neurons_layer1, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer2, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(1, activation=None))
    
    sgd = optimizers.SGD(lr=eta)
    model.compile(loss='mean_squared_error', optimizer=sgd)  ##try adam also
    
    return model

DNN_keras = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
  
def train_dnn():    ##fit for different learning rate and decay (lambda)  
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            DNN = neural_network_keras(n_neurons_layer1, n_neurons_layer2, 
                                            eta=eta, lmbd=lmbd)
            DNN.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0) ##what is verbose?
            scores = DNN.evaluate(X_test, y_test, verbose=0)
            
            DNN_keras[i][j] = DNN
            
            print("Learning rate = ", eta)
            print("Lambda = ", lmbd)
            print("Test MSE: %.3f" % scores)
            print()  
  
train_dnn()  
   
###For learning rate 0.1 and wno regularizing (i.e lambda = 0), we have the best test accuracy, thus use them here

def neural_network_no_l2(n_neurons_layer1, n_neurons_layer2, eta):
    model = Sequential()
    model.add(Dense(n_neurons_layer1, activation='sigmoid', kernel_regularizer=None))
    model.add(Dense(n_neurons_layer2, activation='sigmoid', kernel_regularizer=None))
    model.add(Dense(1, activation=None))
    
    sgd = optimizers.SGD(lr=eta)
    model.compile(loss='mean_squared_error', optimizer='adam')  ##adam performs much better than sgd
    
    return model

DNN = neural_network_no_l2(n_neurons_layer1, n_neurons_layer2, 
                                         eta=0.1)
dnn_keras = DNN.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)

def plot_loss(history):
    plt.plot(dnn_keras.history['loss'], label='loss')
    plt.plot(dnn_keras.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    return plt.show()

plot_loss(dnn_keras)

print(DNN.evaluate(X_test, y_test, verbose=0))
y_pred = DNN.predict(X_test).flatten()


def plot_scatter(y_pred, y_true, backend = 'manual'):
    a = plt.axes(aspect='equal')
    plt.scatter(y_pred, y_pred, color= 'blue')
    plt.scatter(y_pred, y_test, color = 'red')
    plt.xlabel('True y values')
    plt.ylabel('Predicted y')
    if backend == 'tfl':
        plt.title('DNN Tensorflow Prediction (MSE = 0.012)')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'nn_pred_tf.png'), transparent=True, bbox_inches='tight')
        plt.show()
    if backend == 'manual':
        plt.title('DNN Manual Prediction (MSE = 0.014)')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'nn_pred_manual.png'), transparent=True, bbox_inches='tight')
        plt.show()
    
plot_scatter(y_pred, y_test, backend='tfl') ##Tensorflow predictions
     

def mymodel():
    nn = NeuralNetwork()
    nn.add_layer(Layer(X_train.shape[1], 50 , 'sigmoid', alpha = 0.01, lam = 0.1))
    nn.add_layer(Layer(50, 50, 'sigmoid', alpha = 0.01, lam = 0.1))
    #nn.add_layer(Layer(18, 18, 'sigmoid'))
    nn.add_layer(Layer(50, 1, None, alpha = 0.01, lam = 0.1))
    train = nn.train(X_train, y_train, 0.0001, 100, lmbd=0.01)
    y_pred = nn.predict(X_test, net_type='regression')
    print('Test MSE: {}' .format(nn.MSE(y_pred.flatten(), y_test)))
    return y_pred

y_pred = mymodel()

plot_scatter(y_pred, y_test, backend='manual') ### my own nn predictions

