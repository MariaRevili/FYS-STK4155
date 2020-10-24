import os
import sys
import pytest
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import functools
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 
import seaborn as sns
from sklearn import datasets

sns.set()


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'Functions'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))

from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix
from Bootstrap import Bootstrap
from stochastic_gradient_descent import SGD, compute_test_mse
from MLP import Layer, NeuralNetwork




###Data pre-processing########

# ensure the same random numbers appear every time
np.random.seed(20)

# display images in notebook
plt.rcParams['figure.figsize'] = (12,12)


# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))


# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)

def plot_digits():
    for i, image in enumerate(digits.images[random_indices]):
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %d" % digits.target[random_indices[i]])
        plt.show()


###split into train - validation -test data. Choose hyper-parameters on the validation data

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

###scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


print("Number of training images: " + str(len(X_train)))
print("Number of test images: " + str(len(X_test)))

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

y_train_onehot, y_test_onehot, y_val_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test), to_categorical_numpy(y_val)


def plot_nn_accuracy(act):
        
    if act == 'sigmoid':
        learning_rate = np.logspace(-5, -1, 5)
        lambda_ = np.logspace(-5, -1, 3)
    if act == 'elu':
        learning_rate = np.logspace(-5, -1, 5)
        lambda_ = np.logspace(-5, -1, 3)
    if act == 'relu':
        learning_rate = np.logspace(-5, -1, 5)
        lambda_ = np.logspace(-5, -1, 3)
    if act == 'leaky_relu':
        learning_rate = np.logspace(-5, -1, 5)
        lambda_ = np.logspace(-5, -1, 3)
    if act == 'tanh':
        learning_rate = np.logspace(-5, -1, 5)
        lambda_ = np.logspace(-5, -1, 3)
        

    n_categories = np.max(labels) + 1
    
    
    train_accuracy = np.zeros((len(learning_rate), len(lambda_)))
    validation_accuracy = np.zeros((len(learning_rate), len(lambda_)))
    
    DNN_numpy = np.zeros((len(learning_rate), len(lambda_)), dtype=object)

    # grid search
    for i, eta in enumerate(learning_rate):
        
        for j, lam in enumerate(lambda_):
            
            nn = NeuralNetwork()
            n_hidden = 50
            nn.add_layer(Layer(X_train.shape[1], n_hidden, act))
            #nn.add_layer(Layer(n_hidden, n_hidden, 'sigmoid'))
            nn.add_layer(Layer(n_hidden, n_categories, 'softmax'))
            
            nn.train(X_train, y_train_onehot, eta, 100, lmbd=lam, net_type='classification')
            
            DNN_numpy[i][j] = nn
            
            y_pred = nn.predict(X_val, net_type='classification')
                        
            print("Learning rate  = {}, Lambda = {}, Accuracy = {} " .format(eta, lam, nn.accuracy((y_pred.flatten()), y_val) ))
    
    
    
    
    for i in range(len(learning_rate)):
        
        for j in range(len(lambda_)):  
                
            dnn = DNN_numpy[i][j]
    
            y_pred = dnn.predict(X_val, net_type='classification')
        
            #train_accuracy[i][j] = nn.accuracy((y_pred.flatten()), y_train)
            validation_accuracy[i][j] = nn.accuracy((y_pred.flatten()), y_val)
                    
    
 
    fig, ax = plt.subplots(figsize = (10, 10))
    xlabels = ['{:3.3f}'.format(x) for x in lambda_]
    ylabels = ['{:3.4f}'.format(y) for y in learning_rate]      
    sns.heatmap(validation_accuracy, xticklabels = xlabels, yticklabels = ylabels, annot=True, ax=ax, cmap='viridis')
        
    if act == 'sigmoid':
        ax.set_title("Validation Accuracy sigmoid")
        
    if act == 'elu':
        
        ax.set_title("Validation Accuracy elu")
              
    if act == 'relu':
        
        ax.set_title("Validation Accuracy ReLu")
                 
    if act == 'leaky_relu':
        
        ax.set_title("Validation Accuracy leaky ReLu")

    if act == 'tanh':
        
        ax.set_title("Validation Accuracy Tanh")
        
    ax.set_ylabel(r"$\eta$")
    ax.set_xlabel(r"$\lambda$")     
            
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'accuracy_valid_tanh.png'), transparent=True, bbox_inches='tight')
    
    return plt.show()


plot_nn_accuracy('sigmoid')
plot_nn_accuracy('relu') 
plot_nn_accuracy('elu')  
plot_nn_accuracy('leaky_relu')
plot_nn_accuracy('tanh')


##### Play with different hidden nodes

def accuracy_hidden():
    n_hidden = [10, 20, 30, 40, 50, 100]
    Accuracy_hid = []
    n_categories = np.max(labels) + 1

    for hidden in n_hidden:
        
        nn = NeuralNetwork()
        nn.add_layer(Layer(X_train.shape[1], hidden, 'sigmoid'))
        #nn.add_layer(Layer(n_hidden, n_hidden, 'sigmoid'))
        nn.add_layer(Layer(hidden, n_categories, 'softmax'))
        train = nn.train(X_train, y_train_onehot, 0.001, 100, lmbd=0.001, net_type='classification')
        y_pred = nn.predict(X_val, net_type='classification')
        Accuracy_hid.append(nn.accuracy((y_pred.flatten()), y_val))
        print('For {} hidden neurons accuracy is {} ' .format(hidden, nn.accuracy((y_pred.flatten()), y_val)))
        

    plt.plot(n_hidden, Accuracy_hid, color = 'black', linestyle = 'dashed',linewidth = 1.5, marker = 'o', markersize=1)
    plt.xlabel('Hidden Neurons')
    plt.ylabel('Validation Accuracy')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'n_hidden_accuracy_sigm.png'), bbox_inches='tight')
    return plt.show()

accuracy_hidden()  ##40 hidden layers is most optimal



###Test accuracy
nn = NeuralNetwork()
nn.add_layer(Layer(X_train.shape[1], 40, 'sigmoid'))
#nn.add_layer(Layer(n_hidden, n_hidden, 'sigmoid'))
nn.add_layer(Layer(40, 10, 'softmax'))
train = nn.train(X_train, y_train_onehot, 0.001, 300, lmbd=0.001, net_type='classification')
y_pred = nn.predict(X_test, net_type='classification')
print(nn.accuracy((y_pred.flatten()), y_test))