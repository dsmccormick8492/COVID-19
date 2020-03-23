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

import urllib.request
from io import StringIO
from datetime import datetime


### constants
NEWLINE = '\n'
TAB = '\t'
SPACE = ' '
COMMA = ','
QUOTE = '"'
READ_MODE = 'r'
WRITE_MODE = 'w'
ENCODING_TYPE = 'utf-8'
EMPTY_STRING = ""

FIGURE_SIZE = (10, 10)

states_daily = "https://covidtracking.com/api/us/daily"
us_daily_csv = "https://covidtracking.com/api/us/daily.csv"
states_daily_csv = "https://covidtracking.com/api/states/daily.csv"


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


#%% US
with urllib.request.urlopen(us_daily_csv) as response:
   csv_bytes = response.read()

string = str(csv_bytes, ENCODING_TYPE)
data = StringIO(string)
us_daily_df = pd.read_csv(data)

datetime_strings = [s for s in us_daily_df['date']]
datetimes = [datetime.strptime(str(s), "%Y%m%d") for s in datetime_strings]
us_daily_df['datetime'] = datetimes

#%% states
with urllib.request.urlopen(states_daily_csv) as response:
   csv_bytes = response.read()

string = str(csv_bytes, ENCODING_TYPE)
data = StringIO(string)
states_df = pd.read_csv(data)

datetime_strings = [s for s in states_df['date']]
datetimes = [datetime.strptime(str(s), "%Y%m%d") for s in datetime_strings]
states_df['datetime'] = datetimes

states_gb = states_df.groupby(by='state')

ma_daily_gb = states_gb.get_group('MA')
ma_daily_df = ma_daily_gb.sort_values(by=['date'], ignore_index=True)

ny_daily_gb = states_gb.get_group('NY')
ny_daily_df = ny_daily_gb.sort_values(by=['date'], ignore_index=True)

nj_daily_gb = states_gb.get_group('NJ')
nj_daily_df = ny_daily_gb.sort_values(by=['date'], ignore_index=True)


#%% plots!

plot_daily_data(us_daily_df, title_str='All US')
plot_daily_data(ma_daily_df, title_str='MA')
plot_daily_data(ny_daily_df, title_str='NY')
plot_daily_data(nj_daily_df, title_str='NJ')
