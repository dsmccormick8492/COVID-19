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
Created on Fri Mar 27 10:35:34 2020
"""
### import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc3 as pm

#%% MCMC function
def mcmc_log_regression(x: np.array, y: np.array, y_hat: np.array, slope: float, data_type: str) -> None:
    """
    

    Parameters
    ----------
    x : np.array
        observed x values.
    y : np.array
        observed y values. These are assumed to be log10() of the original data.
    y_hat : np_array
        DESCRIPTION.
    slope : float
        DESCRIPTION.
    data_type : str
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """

    basic_model = pm.Model()
    
    ### model specification
    with basic_model:
        # Priors for unknown model parameters
        slope_mc = pm.Uniform('slope_mc', lower=0.1, upper=0.2)
        intercept_mc = pm.Uniform('intercept_mc', lower=1.5, upper=2)
        # sigma_mc = pm.Normal('sigma_mc', mu=0.05, sigma=0.02)
        # sigma_mc = pm.HalfNormal('sigma_mc', sigma=0.2)
        sigma_mc = pm.Uniform('sigma_mc', lower=0.001, upper=0.08)
    
        # Expected value of outcome
        expected_value = slope_mc * x + intercept_mc
    
        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('y_obs', mu=expected_value, sigma=sigma_mc, observed=y)
    
        # MAP estimate of model values
        map_estimate = pm.find_MAP(model=basic_model)
        print(map_estimate)
        
        # draw 500 posterior samples
        trace = pm.sample(500, target_accept=0.95)
        pm.traceplot(trace, compact=False)
        pm.plot_posterior(trace, round_to=4)
        plt.show()
        
    y_map = map_estimate['slope_mc'] * x + map_estimate['intercept_mc']
    
    ### plotting    
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, c='k', zorder=10)
    plt.plot(x, y_hat, c='r', lw=8, label=f"linear regression, slope={slope:0.3f}")
    plt.plot(x, y_map, c='b', lw=2, label=f"MCMC MAP linear regression, slope={map_estimate['slope_mc']:0.3f}")
    plt.legend()
    plt.show()
    
    ### posterior sampling of slope and intercept
    sample_count = 500
    # samples = pm.sample_posterior_predictive(trace=trace, samples=sample_count, model=basic_model, var_names=['slope_mc', 'intercept_mc'])
    samples = pm.sample_posterior_predictive(trace=trace, model=basic_model, var_names=['slope_mc', 'intercept_mc'])
    
    sample_count = len(samples['slope_mc'])
    
    # plt.figure(figsize=(10, 10))
    # for i in range(sample_count):
    #     y_sample = samples['slope_mc'][i] * x + samples['intercept_mc'][i]  
    #     plt.plot(x, y_sample, c='b', alpha=0.3)
    # plt.plot(x, y_hat, c='r', lw=2, label=f"linear regression, slope={slope:0.3f}", zorder=5)
    # plt.scatter(x, y, c='r', label='data', zorder=10)
    # plt.xlim(0, x.max())
    # plt.legend()
    # title_str = f"posterior predictive sampling of MCMC fit to log10(confirmed cases) since {confirmed_dates_start}"
    # plt.title(title_str)
    # plt.show()
    
    plt.figure(figsize=(10, 10))
    for i in range(sample_count):
        y_sample = samples['slope_mc'][i] * x + samples['intercept_mc'][i]
        plt.plot(x, 10**y_sample, c='b', alpha=0.3)
    plt.plot(x, 10**y_hat, c='r', lw=2, label=f"linear regression, slope={slope:0.3f}", zorder=5)
    plt.scatter(x, 10**y, c='r', label='data', zorder=10)
    plt.xlim(0, x.max())
    plt.yscale('log')
    plt.grid(which='both')
    plt.legend()
    title_str = f"{sample_count} posterior predictive samples of MCMC fit to log10({data_type}) since {confirmed_dates_start}"
    plt.title(title_str)
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.scatter(x, 10**y, c='r', label='data', zorder=10)
    plt.plot(x, 10**y_hat, c='r', lw=2, label=f"linear regression, slope={slope:0.3f}", zorder=5)
    for i in range(sample_count):
        y_sample = samples['slope_mc'][i] * x + samples['intercept_mc'][i]
        plt.plot(x, 10**y_sample, c='b', alpha=0.3)
    plt.xlim(0, x.max())
    plt.grid(which='both')
    plt.legend()
    title_str = f"{sample_count} posterior predictive samples of MCMC fit to log10({data_type}) cases) since {confirmed_dates_start}"
    plt.title(title_str)
    plt.show()
