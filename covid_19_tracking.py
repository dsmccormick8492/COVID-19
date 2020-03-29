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

from datetime import datetime

### project imports
from dataframe_from_csv_url import dataframe_from_csv_url
import linear_regression as lr
import doubling

### constants
ENCODING_TYPE = 'utf-8'
FIGURE_SIZE = (10, 10)

#%% functions
def plot_daily_data(df: pd.DataFrame, title_str: str) -> None:
    plt.figure(figsize=FIGURE_SIZE)
    dates = df['date']
    dates = df['datetime']
    plt.plot(dates, df['total'], label='total')
    plt.plot(dates, df['positive'], label='positive')
    plt.plot(dates, df['negative'], label='negative')
    plt.plot(dates, df['death'], label='deaths')
    plt.yscale('log')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.title(f"COVID-19 statistics for {title_str}")
    

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

#%% states
state_codes = ['MA', 'NY', 'NJ', 'CA', 'TX', 'LA']
state_names = {
    'MA' : 'Massachusetts',
    'NY' : 'New York',
    'NJ' : 'New Jersey',
    'CA' : 'California',
    'TX' : 'Texas',
    'LA' : 'Louisiana',
    }

datetime_strings = [s for s in states_daily_df['date']]
datetimes = [datetime.strptime(str(s), "%Y%m%d") for s in datetime_strings]
states_daily_df['datetime'] = datetimes

state_dfs = {}

states_gb = states_daily_df.groupby(by='state')

for state_code in state_codes:
    daily_gb = states_gb.get_group(state_code)
    daily_df = daily_gb.sort_values(by=['date'], ignore_index=True)
    state_dfs[state_code] = daily_df

entity_codes = ['US']
entity_codes.extend(state_codes)
entity_dfs = {}
entity_dfs['US'] = us_daily_df
entity_dfs.update(state_dfs)

#%% plots!
plot_daily_data(us_daily_df, title_str='All US')
for state_code, state_df in state_dfs.items():
    plot_daily_data(state_df, title_str=state_names[state_code])


#%% some regression tests
df = us_daily_df[['datetime', 'positive']].copy().dropna()
df = df[df['positive'] > 0].reset_index(drop=True)
slope, intercept, r_value, std_err, y_hat = lr.linear_regression(df.index, np.log10(df['positive']))
days2double = doubling.days_to_double(slope)
print(f"Days for positive cases in US to double => {days2double:0.2f}")

df = state_df[['datetime', 'death']].copy().dropna().reset_index(drop=True)
slope, intercept, r_value, std_err, y_hat = lr.linear_regression(df.index, np.log10(df['death']))
days2double = doubling.days_to_double(slope)
print(f"Days for deaths in US to double => {days2double:0.2f}")
print()


for state_code, state_df in state_dfs.items():
    df = state_df[['datetime', 'positive']].copy().dropna()
    df = df[df['positive'] > 0].reset_index(drop=True)
    slope, intercept, r_value, std_err, y_hat = lr.linear_regression(df.index, np.log10(df['positive']))
    days2double = doubling.days_to_double(slope)
    print(f"Days for positive cases in {state_names[state_code]} to double => {days2double:0.2f}")

    df = state_df[['datetime', 'death']].copy().dropna().reset_index(drop=True)
    slope, intercept, r_value, std_err, y_hat = lr.linear_regression(df.index, np.log10(df['death']))
    days2double = doubling.days_to_double(slope)
    print(f"Days for deaths in {state_names[state_code]} to double => {days2double:0.2f}")
    print()
