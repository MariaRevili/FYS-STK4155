import os 
import numpy as np
import scipy 
import warnings
import sys 
import pytest
import copy

## Add Functions to the directory to import them
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Functions'))
from multiclassLogisticReg import multiclassLogistic



def test_multiclassLogisticReg():
    
    """
    Alternative implementation of scikit learn multinomial logistic regression. Verifies that 
    scikit learn and manually setup predictions coincide
    
    """
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.arange(2).reshape(-1, 1)
    y = np.array(np.arange(2))
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    def to_categorical_numpy(integer_vector):
        
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros((n_inputs, n_categories))
        onehot_vector[range(n_inputs), integer_vector] = 1
    
        return onehot_vector
    
    y_onehot = to_categorical_numpy(y)

        
    lr = LogisticRegression(solver='lbfgs',
                                    multi_class='multinomial',
                                    penalty='l2',
                                    max_iter=100,
                                    random_state=1,
                                    C=1e5)


    ##Validation accuracy scikit learn - same test data accuracy as my own
    lr.fit(x, y)
    pred = lr.predict(x)
    print('scikitlearn prediction {}' .format(pred))
    pred_skl = sum(pred == y)/(float(len(y)))

    
    logreg = multiclassLogistic(x, y, y_onehot, learning_rate=0.1, lambda_ = 0.1)
    beta_manual = logreg.sgd(x, y, y_onehot, 100, learning_rate=0.1, lambda_ = 0.1)
    pred_manual = logreg.accuracy(x, y, beta_manual)
    
    assert pred_skl == pytest.approx(pred_manual)

