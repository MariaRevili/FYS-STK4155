import os
import sys
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt


# Add the Functions/ directory to the python path so we can import the code 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Functions'))

from LinearReg import linregOwn, linregSKL
from designMat import designMatrix


def test_linregOwn_OLS():
    X = np.array([[np.cos(5), np.sin(5)], [-np.sin(5), np.cos(5)]])
    y= np.array([[7],[4]])
    beta_manual = np.array([[7*np.cos(5)-4*np.sin(5)], [7*np.sin(5)+4*np.cos(5)]])
    linreg = linregOwn(method = 'ols')
    beta = linreg.fit(X, y)
    assert beta == pytest.approx(beta_manual, abs = 1e-10)


def test_linregOwn_Ridge():
    X = np.array([[np.cos(5), np.sin(5)], [-np.sin(5), np.cos(5)]])
    y= np.array([[7],[4]])
    lambda_ = 0.1
    beta_ridge_manual = np.array([[(7*np.cos(5)-4*np.sin(5))/(1+lambda_)], 
                                  [(7*np.sin(5)+4*np.cos(5))/(1+lambda_)]])
    linreg = linregOwn(method = 'ridge')
    beta_ridge = linreg.fit(X, y, lambda_= 0.1)
    assert beta_ridge_manual == pytest.approx(beta_ridge, abs= 1e-10)


def test_MSE_R2():
    X = np.array([[np.cos(5), np.sin(5)], [-np.sin(5), np.cos(5)]])
    y= np.array([[7],[4]])
    MSE_manual = 0
    linreg = linregOwn()
    linreg.fit(X, y)
    linreg.predict(X)
    MSE = linreg.MSE(y)
    R2_manual = 1
    R2 = linreg.R2(y)
    assert MSE_manual == pytest.approx(MSE, abs = 1e-15)
    assert R2_manual == pytest.approx(R2, abs=1e-15)

    
    