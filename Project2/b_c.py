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
sns.set()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'Functions'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))

from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix
from Bootstrap import Bootstrap
from stochastic_gradient_descent import SGD, compute_test_mse
from MLP import Layer, NeuralNetwork



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



##playing with activation functions, learning rate and stochastic gradient descent decay

def plot_nn_mse(act):
        
    if act == 'sigmoid':
        learning_rate = np.logspace(-5, -4, 5)
        lambda_ = np.logspace(-5, -2, 3)
    if act == 'elu':
        learning_rate = np.logspace(-10, -8, 5)
        lambda_ = np.logspace(-5, -2, 3)
    if act == 'relu':
        learning_rate = np.logspace(-6, -4, 3)
        lambda_ = np.logspace(-2, -1, 3)
    if act == 'leaky_relu':
        learning_rate = np.logspace(-5, -3.9, 5)
        lambda_ = np.logspace(-2, -1, 3)    
    if act == 'tanh':
        learning_rate = np.logspace(-5, -3, 5)
        lambda_ = np.logspace(-5, -2, 3)
        
    n_categories = 1
    
    
    validation_MSE = np.zeros((len(learning_rate), len(lambda_)))
    
    DNN_numpy = np.zeros((len(learning_rate), len(lambda_)), dtype=object)

    # grid search
    for i, eta in enumerate(learning_rate):
        
        for j, lam in enumerate(lambda_):
            
            nn = NeuralNetwork()
            n_hidden = 50
            nn.add_layer(Layer(X_train.shape[1], n_hidden, act))
            nn.add_layer(Layer(n_hidden, n_hidden, 'sigmoid'))
            nn.add_layer(Layer(n_hidden, n_categories, None))
            
            nn.train(X_train, y_train, eta, 100, lmbd=lam, net_type='regression')
            
            DNN_numpy[i][j] = nn
            
            y_pred = nn.predict(X_val, net_type='regression')
                        
            print("Learning rate  = {}, Lambda = {}, MSE = {} " .format(eta, lam, nn.MSE((y_pred.flatten()), y_val) ))
    
    
    
    
    for i in range(len(learning_rate)):
        
        for j in range(len(lambda_)):  
                
            dnn = DNN_numpy[i][j]
    
            y_pred = dnn.predict(X_val, net_type='regression')
        
            #train_accuracy[i][j] = nn.accuracy((y_pred.flatten()), y_train)
            validation_MSE[i][j] = nn.MSE((y_pred.flatten()), y_val)
                    
    
 
    fig, ax = plt.subplots(figsize = (10, 10))
    xlabels = ['{:3.3f}'.format(x) for x in lambda_]
    ylabels = ['{:3.4f}'.format(y) for y in learning_rate]      
    sns.heatmap(validation_MSE, xticklabels = xlabels, yticklabels = ylabels, annot=True, ax=ax)
        
    if act == 'sigmoid':
        ax.set_title("Validation MSE sigmoid")
     
    if act == 'elu':
        
        ax.set_title("Validation MSE elu") 
                
    if act == 'relu':
        
        ax.set_title("Validation MSE ReLu")
                
    if act == 'leaky_relu':
        
        ax.set_title("Validation MSE leaky ReLu")
                        
    if act == 'tanh':
        
        ax.set_title("Validation MSE Tanh")
        
    ax.set_ylabel(r"$\eta$")
    ax.set_xlabel(r"$\lambda$")     
            
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'MSE_reg_relu.png'), transparent=True, bbox_inches='tight')
    
    return plt.show()

plot_nn_mse(act = 'sigmoid')

plot_nn_mse(act = 'relu')

plot_nn_mse(act = 'leaky_relu')



### Play with hidden neurons and layers -18 neurons perform best with sigmoid activation

def play_with_neurons():
    n_hidden = [10, 15, 40, 50]
    MSE_hid = []

    for hidden in n_hidden:
        
        nn = NeuralNetwork()
        nn.add_layer(Layer(X_train.shape[1], hidden, 'sigmoid'))
        nn.add_layer(Layer(hidden, hidden, 'sigmoid'))
        #nn.add_layer(Layer(hidden, hidden, 'sigmoid'))
        nn.add_layer(Layer(hidden, 1, None))
        train = nn.train(X_train, y_train, 0.0001, 100, lmbd=0.01)
        y_pred = nn.predict(X_val, net_type='regression')
        MSE_hid.append(nn.MSE((y_pred.flatten()), y_val))
        print('For {} hidden neurons MSE is {} ' .format(hidden, nn.MSE((y_pred.flatten()), y_val)))
        

    plt.plot(n_hidden, MSE_hid, color = 'black', linestyle = 'dashed',linewidth = 1.5, marker = 'o', markersize=1)
    plt.xlabel('Hidden Neurons')
    plt.ylabel('Validation MSE (Sigmoid as an activation function)')
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'n_hidden_mse_sigm.png'), transparent=True, bbox_inches='tight')
    return plt.show()


play_with_neurons()


###Play with elu, relu, leaky_relu parameters
def play_with_activation():
    alpha = [0.01, 0.1]
    lambd = [0.01, 0.1]
    MSE_hid = []

    for act in ['relu', 'leaky_relu']:
        for a in alpha :
            for lambda_ in lambd:
                nn = NeuralNetwork()
                nn.add_layer(Layer(X_train.shape[1], 18 , act, alpha = a, lam = lambda_))
                nn.add_layer(Layer(18, 18, act, alpha = a, lam = lambda_))
                #nn.add_layer(Layer(hidden, hidden, 'sigmoid'))
                nn.add_layer(Layer(18, 1, None, alpha = a, lam = lambda_))
                train = nn.train(X_train, y_train, 0.001, 100, lmbd=0.001)
                y_pred = nn.predict(X_val, net_type='regression')
                MSE_hid.append(nn.MSE((y_pred.flatten()), y_val))
                print('For alpha {} and lambda {} in activation function {} MSE is {} ' .format(a, lambda_, act,  nn.MSE((y_pred.flatten()), y_val)))


play_with_activation()

def plot_MSE_ols_ridge_nn():
    MSE_ridge_val = []
    MSE_ols_val = []
    MSE_nn_val = []
    methods = ['ridge', 'ols', 'nn']
    for method in methods:
        
        if method == 'ridge':
            
            eta = np.logspace(-5, -3, 10) 
            
            for i in eta:            
                
                beta = SGD(X_train, y_train, learning_rate=i, lambda_ = 0.01, method = 'ridge')
                mse_ols_, mse_ridge_ = compute_test_mse(X_val, y_val, lambda_ = 0.01, beta = beta)
                MSE_ridge_val.append(mse_ridge_)
                
        if method == 'ols':
            
            eta = np.logspace(-5, -3, 10) 
            
            for i in eta:            
                
                beta = SGD(X_train, y_train, learning_rate=i, lambda_ = 0.01, method = 'ols')
                mse_ols_, mse_ridge_ = compute_test_mse(X_val, y_val, beta = beta)
                MSE_ols_val.append(mse_ols_)
        
        if method == 'nn':
            
            eta = np.logspace(-5, -3, 10) 
            
            for i in eta:
                
                nn = NeuralNetwork()
                nn.add_layer(Layer(X_train.shape[1], 18 , 'sigmoid', alpha = 0.1, lam = 0.1))
                nn.add_layer(Layer(18, 18, 'sigmoid', alpha = 0.1, lam = 0.1))
                #nn.add_layer(Layer(hidden, hidden, 'sigmoid'))
                nn.add_layer(Layer(18, 1, None, alpha = 0.1, lam = 0.1))
                train = nn.train(X_train, y_train, i, 100, lmbd=0.01)
                y_pred = nn.predict(X_val, net_type='regression')
                MSE_nn_val.append(nn.MSE((y_pred.flatten()), y_val))
        
    plot, ax = plt.subplots()
    plt.title('MSE for the OLS, Ridge and Neural Networks (validation data)')
    plt.semilogx(np.logspace(-5, -3, 10), MSE_ridge_val, 'k-o', label = 'Ridge')
    plt.semilogx(np.logspace(-5, -3, 10), MSE_ols_val, 'r-o', label = 'OLS')
    plt.semilogx(np.logspace(-5, -3, 10), MSE_nn_val, 'b-o', label = 'Neural Networks')
    plt.xlabel(r'Learning rate $\eta$')
    plt.ylabel('MSE')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2,bottom=0.2,right=0.9)
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'mse_all.png'), transparent=True, bbox_inches='tight')
    return plt.show()

    
plot_MSE_ols_ridge_nn()

    
