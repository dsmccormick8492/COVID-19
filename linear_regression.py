#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: 
    
Purpose:
    
Description:
    
Comments:

TODO:
    
@author: dmccormick
Author: David S. McCormick
Created on Sat Mar 28 11:49:48 2020
"""
### import modules
import numpy as np
import scipy.stats as stats


def linear_regression(x: np.array, y: np.array) -> tuple:
    """
    

    Parameters
    ----------
    x : Numpy array
        array of x data values.
    y : TYPE
        array of y data values.

    Returns
    -------
    slope : float
        linear regression slope.
    intercept : TYPE
        linear regression y intercept.
    r_value : TYPE
        linear regression correlation coefficient.
    std_err : TYPE
        linear regression standard error.
    y_hat : TYPE
        prediction of y from linear regression.
        
    Usage
    -----
        slope, intercept, r_value, std_err, y_hat = linear_regression(x, y)

    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    y_hat = slope * x + intercept
    return slope, intercept, r_value, std_err, y_hat