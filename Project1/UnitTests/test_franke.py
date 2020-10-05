import os
import sys
import pytest
import numpy as np

# Add the Functions/ directory to the python path so we can import the code 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Functions'))

from Franke import franke

def test_franke():
    """
    Tests the franke function implemented in Franke.py. The test is composed of the alternative
    implementation of the franke function in R. 
    
    R code:
    ///
    install.packages("SigOptR")
    library(SigOptR)
    set.seed(5)
    x <- runif(4)
    y <- runif(4)
    z <- franke(x, y)
    ///
    
    References: https://rdrr.io/cran/SigOptR/man/franke.html
    """
    x = [0.2002145, 
         0.6852186, 
         0.9168758, 
         0.2843995]
    y = [0.1046501,
        0.7010575, 
        0.5279600, 
        0.8079352]
    z = [1.0877452, 
         0.1527279, 
         0.2304823,
        0.2306133]
    for i in range(4):
        assert franke(x[i], y[i]) == pytest.approx(z[i], abs=1e-7)
