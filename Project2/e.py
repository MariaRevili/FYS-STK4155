import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LogisticRegression
sns.set()


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'Functions'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))

from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix
from Bootstrap import Bootstrap
from stochastic_gradient_descent import SGD, compute_test_mse
from MLP import Layer, NeuralNetwork
from multiclassLogisticReg import multiclassLogistic


### Multiclass Logistic Regression ###

#####Load the data #######
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

def append_intercept(X):
    xb = np.ones((X.shape[0], 1))
    return np.c_[xb, X]

X_train = append_intercept(X_train)
X_test = append_intercept(X_test)
X_val = append_intercept(X_val)


print("Number of training images: " + str(len(X_train)))
print("Number of test images: " + str(len(X_test)))


def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

y_train_onehot, y_test_onehot, y_val_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test), to_categorical_numpy(y_val)

learningRate = np.logspace(-3, 0, 4)
lmbd = np.logspace(-3, 0, 4) 


###Checking validation set accuracy for different values of learning rate and regularization

def plot_logistic_accuracy():
    learning_rate = np.logspace(-3, 0, 4)
    lambda_ = np.logspace(-3, 0, 4)
    n_categories = np.max(labels) + 1
    
    validation_accuracy = np.zeros((len(learning_rate), len(lambda_)))
    
    logistic_numpy = np.zeros((len(learning_rate), len(lambda_)), dtype=object)

    # grid search
    for i, eta in enumerate(learning_rate):
        
        for j, lam in enumerate(lambda_):
            
            logreg = multiclassLogistic(X_train, y_train, y_train_onehot, learning_rate=eta, lambda_ = lam)
            beta = logreg.sgd(X_train, y_train, y_train_onehot, 1000, lambda_ = eta, learning_rate= lam)
            
            validation_accuracy[i][j] = logreg.accuracy(X_val, y_val, beta)
            
                                        
            print("Learning rate  = {}, Lambda = {}, Accuracy = {} " .format(eta, lam, logreg.accuracy(X_val, y_val, beta)))

         
    fig, ax = plt.subplots(figsize = (10, 10))
    #xlabels = ['{:3.1f}'.format(x) for x in lambda_]
    #ylabels = ['{:3.1f}'.format(y) for y in learning_rate]       
    sns.heatmap(validation_accuracy, xticklabels = lambda_, yticklabels = learning_rate, annot=True, ax=ax, cmap="viridis")
        
    ax.set_title("Validation Accuracy Logistic Regression")
    ax.set_ylabel(r"$\eta$")
    ax.set_xlabel(r"$\lambda$")
    
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'logistic_accuracy.png'), transparent=True, bbox_inches='tight')

    return plt.show()
         

plot_logistic_accuracy()


    
#### Compare with scikit learn ####
        
lr = LogisticRegression(solver='lbfgs',
                                multi_class='multinomial',
                                penalty='l2',
                                max_iter=100,
                                random_state=42,
                                C=1e5)


##Validation accuracy scikit learn - same test data accuracy as my own

lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print('scikit learn: {}'.format(sum(pred == y_test)/(float(len(y_test))))) 


logreg = multiclassLogistic(X_train, y_train, y_train_onehot, learning_rate=0.01, lambda_ = 0.01)
beta = logreg.sgd(X_train, y_train, y_train_onehot, 1000, learning_rate=0.01, lambda_ = 0.01)
print('Manual accuracy: {}' .format(logreg.accuracy(X_test, y_test, beta)))


    


