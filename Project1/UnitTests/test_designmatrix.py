import os
import sys
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt

# Add the Functions/ directory to the python path so we can import the code 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Functions'))
from designMat import designMatrix

#cd "C:/Users/Marie/OneDrive/Documents/Project1" in the terminal and then run pytest -v


def test_designMatrix():
    """
    Tests the design matrix given in designMat.py
    
    The test is composed of the simple numpy vectors for which we can evaluate the function manually.
    """
    ##1 degree polynomial
    x = np.array([2.0])
    y = np.array([3.0])
    X = designMatrix(x, y, 1)
    X_true = np.array([[1.0,2.0,3.0]])
    assert X == pytest.approx(X_true, abs = 1e-15)
    
    ##2nd degree polynomial
    x = np.array([2.0])
    y = np.array([3.0])
    X = designMatrix(x, y, 2)
    X_true = np.array([[1.0,2.0,3.0, 4.0,6.0,9.0]])
    assert X == pytest.approx(X_true, abs = 1e-15)
    
    ##3rd degree polynomial
    x = np.array([2.0])
    y = np.array([3.0])
    X = designMatrix(x, y, 3)
    X_true = np.array([[1.0,2.0,3.0,4.0,6.0,9.0,8.0,12.0,18.0,27.0]])
    assert X == pytest.approx(X_true, abs = 1e-15)
    