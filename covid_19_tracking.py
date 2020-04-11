#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: 
    
Purpose:
    
Description:
    Data source: https://covidtracking.com/api/
    
Comments:

TODO:
    
@author: dmccormick
Author: David S. McCormick
Created on Sat Mar 21 07:57:42 2020
"""
### import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

# import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from datetime import datetime

### project imports
from dataframe_from_csv_url import dataframe_from_csv_url
import linear_regression as lr
import doubling as doubling

### constants
ENCODING_TYPE = 'utf-8'
FIGURE_SIZE = (10, 10)

#%% functions
def compute_lowess(x: np.array, y: np.array, is_logrithmic=True) -> np.array:
    """
    Compute a LOWESS (Locally Weighted Scatterplot Smoothing) estimate of 
    data. 
    
    see: https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html

    Parameters
    ----------
    x : np.array
        x data.
    y : np.array
        y data.
    is_logrithmic : bool, optional
        whether the y-data are logrithmically distributed. 
        If so, does the LOWESS estimate on log10(y) data. 
        The default is True.

    Returns
    -------
    y_lowess : np.array, shape (n, 2)
        LOWESS smoothed estimate of the input y data.

    """
    # transform the data to log10 space
    if is_logrithmic:
        y = np.log10(y)
        
    # lower frac uses less data for the fit, 
    y_lowess = lowess(y, x, frac=1./3.)
    # y_lowess = lowess(y, x)

    # back-transform the data to linear space
    if is_logrithmic:
        y_lowess[:, 1] = 10**y_lowess[:, 1]
        
    return y_lowess


def plot_daily_data_cum(df: pd.DataFrame, title_str: str, plot_trend=False) -> None:
    plt.figure(figsize=FIGURE_SIZE)
    # dates = df['date']
    dates = df['datetime']
    plt.plot(dates, df['total'], '-ob', label='total', zorder=10) # plot on top
    plt.plot(dates, df['positive'], '-ok', label='positive')
    # plt.plot(dates, df['negative'], '-og', label='negative')
    plt.plot(dates, df['death'], '-or', label='deaths')
    
    if plot_trend == True:
        # use LOWESS smoothing for defining trends
        positives_lowess = compute_lowess(dates, df['positive'], is_logrithmic=True)
        plt.plot(positives_lowess[:, 0], positives_lowess[:, 1], c='gray')
        totals_lowess = compute_lowess(dates, df['total'], is_logrithmic=True)
        plt.plot(totals_lowess[:, 0], totals_lowess[:, 1], c='gray')
        deaths_lowess = compute_lowess(dates, df['death'], is_logrithmic=True)
        plt.plot(deaths_lowess[:, 0], deaths_lowess[:, 1], c='gray')

    plt.yscale('log')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.title(f"COVID-19 statistics for {title_str}")
    plt.show()

    

def plot_daily_data_diff(df: pd.DataFrame, title_str: str, plot_trend=False) -> None:
    plt.figure(figsize=FIGURE_SIZE)
    # dates = df['date']
    dates = df['datetime']
    dates_diff = dates[1:]
    plt.plot(dates_diff, np.diff(df['total']), '--ob', label='total', zorder=10) # plot on top
    plt.plot(dates_diff, np.diff(df['positive']), '--ok', label='positive')
    # plt.plot(dates_diff, np.diff(df['negative']), '--og', label='negative')
    plt.plot(dates_diff, np.diff(df['death']), '--or', label='deaths')
    
    if plot_trend == True:
        # use LOWESS smoothing for defining trends
        positives_lowess = compute_lowess(dates, df['positive'], is_logrithmic=True)
        plt.plot(positives_lowess[:, 0], positives_lowess[:, 1], c='gray')
        totals_lowess = compute_lowess(dates, df['total'], is_logrithmic=True)
        plt.plot(totals_lowess[:, 0], totals_lowess[:, 1], c='gray')
        deaths_lowess = compute_lowess(dates, df['death'], is_logrithmic=True)
        plt.plot(deaths_lowess[:, 0], deaths_lowess[:, 1], c='gray')

    plt.yscale('log')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.title(f"Daily difference COVID-19 statistics for {title_str}")
    plt.show()
    

#%% fetch data
states_daily = "https://covidtracking.com/api/us/daily"
us_daily_csv = "https://covidtracking.com/api/us/daily.csv"
states_daily_csv = "https://covidtracking.com/api/states/daily.csv"

us_daily_df = dataframe_from_csv_url(us_daily_csv)
us_daily_df.sort_values(by='date', inplace=True)
states_daily_df = dataframe_from_csv_url(states_daily_csv)
us_daily_df.sort_values(by='date', inplace=True)

#%% US
datetime_strings = [s for s in us_daily_df['date']]
datetimes = [datetime.strptime(str(s), "%Y%m%d") for s in datetime_strings]
us_daily_df['datetime'] = datetimes

entity_codes = ['US']
entity_names = ['United States']
# entity_df_dict = {}
# entity_df_dict['US'] = us_daily_df
entity_df_dict = {'US' : us_daily_df}

#%% states and US
### create state tables
state_names_dict = {
    'MA' : 'Massachusetts',
    'NY' : 'New York',
    'NJ' : 'New Jersey',
    'CA' : 'California',
    'TX' : 'Texas',
    'LA' : 'Louisiana',
    'NH' : 'New Hampshire',
    'VT' : 'Vermont',
    }

state_codes = [k for k in state_names_dict.keys()]

datetime_strings = [s for s in states_daily_df['date']]
datetimes = [datetime.strptime(str(s), "%Y%m%d") for s in datetime_strings]
states_daily_df['datetime'] = datetimes

state_dfs = {}

states_gb = states_daily_df.groupby(by='state')

for state_code in state_codes:
    daily_gb = states_gb.get_group(state_code)
    daily_df = daily_gb.sort_values(by=['date'], ignore_index=True)
    state_dfs[state_code] = daily_df

#%% Add US and states into single entity 
# to simplify plotting and calcuations
entity_df_dict.update(state_dfs)
entity_codes.extend(state_codes)
entity_names.extend(list(state_names_dict.values()))
entity_names_dict = dict(zip(entity_codes, entity_names))

#%% plots!
for entity_code, entity_df in entity_df_dict.items():
    plot_daily_data_cum(entity_df, title_str=entity_names_dict[entity_code], plot_trend=False)
    plot_daily_data_diff(entity_df, title_str=entity_names_dict[entity_code], plot_trend=False)
    
#%% some regression tests
for entity_code, entity_df in entity_df_dict.items():
    df = entity_df[['datetime', 'positive']].copy().dropna()
    df = df[df['positive'] > 0].reset_index(drop=True)
    slope, intercept, r_value, std_err, y_hat = lr.linear_regression(df.index, np.log10(df['positive']))
    days2double = doubling.days_to_double(slope)
    print(f"Days for positive cases in {entity_names_dict[entity_code]} to double => {days2double:0.2f}")

    df = entity_df[['datetime', 'death']].copy().dropna().reset_index(drop=True)
    slope, intercept, r_value, std_err, y_hat = lr.linear_regression(df.index, np.log10(df['death']))
    days2double = doubling.days_to_double(slope)
    print(f"Days for deaths in {entity_names_dict[entity_code]} to double => {days2double:0.2f}")
    print()


