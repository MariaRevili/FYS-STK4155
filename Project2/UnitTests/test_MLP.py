import os 
import numpy as np
import scipy 
import warnings
import sys 
import pytest
import copy


# print(os.getcwd())
# path = "C:/Users/Marie/OneDrive/Documents/Project2/UnitTests"

# ##Make a file
# try:
#     os.mkdir(path)
# except OSError:
#     print ("Creation of the directory %s failed" % path)
# else:
#     print ("Successfully created the directory %s " % path)

    
## Add Functions to the directory to import them

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Functions'))
from MLP import Layer, NeuralNetwork

def test_MLP_predict() :
    
    """
    Sets up scikit-learn neural networks system. Verifies that scikit-learn prediction with similar parameters
    roughly coincides with the manually setup neural networks prediction
    
    """
    
    
    # Lets set up a sci-kit learn neural network with similar parameter values

    from sklearn.neural_network import MLPRegressor

    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    X = np.array(X)
    print(X)
    y = [0, 2, 4, 6, 8, 10]
    y = np.array(y)
    mlp = MLPRegressor( solver              = 'sgd', 
                        alpha               = 0.0,
                        hidden_layer_sizes  = (3, 3), 
                        random_state        = 1,
                        activation          = 'relu' ,
                        max_iter= 200,
                        learning_rate_init=0.01)
    mlp.fit(X,y)
    pred_skl = mlp.predict(X)
    pred_skl = np.round(pred_skl, 0)
    print(pred_skl)



    nn = NeuralNetwork()
    nn.add_layer(Layer(1, 35 , 'sigmoid', alpha = 0.01, lam = 0.01))
    nn.add_layer(Layer(35, 35, 'sigmoid', alpha = 0.01, lam = 0.01))
    nn.add_layer(Layer(35, 1, None, alpha = 0.01, lam = 0.01))
    nn.train(X, y, 0.001, 200, lmbd=0.01)
    pred_manual = nn.predict(X, net_type='regression')
    pred_manual = pred_manual.flatten()
    pred_manual = np.round(pred_manual, 0)

    print(pred_manual)
    
    assert pred_skl == pytest.approx(pred_manual)


#cd "C:/Users/Marie/OneDrive/Documents/Project2" to run the test with pytest -v