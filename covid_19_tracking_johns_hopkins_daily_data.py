# -*- coding: utf-8 -*-
"""
Name:

Purpose: 

Description: Data source: Johns-Hopkins
    https://github.com/CSSEGISandData/COVID-19/

Comments:

TODO:
    
Created on Tue Mar  3 10:42:35 2020

@author: dmccormick
"""

### import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

from math import log, log10
from datetime import datetime

# from scipy.optimize import least_squares
import scipy.stats as stats
import pymc3 as pm

### project imports
from dataframe_from_csv_url import dataframe_from_csv_url

### some constants
DATE_COLUMN_START_INDEX = 4

figure_size = (10, 6.5)
rotation_angle = 0

COUNTRY_STR = 'Country/Region'
STATE_STR = 'Province/State'
ADMIN2 = 'Admin2'

#%% functions
def days_to_double(exponent: float) -> float:
    return log(2) / log(10**exponent)


def double_in_days_exponent(days_to_double: int) -> float:
    exponent = log10(2) / days_to_double
    
    return exponent

# tests
# d2d = 1.0
# exponent = double_in_days_exponent(d2d)
# print(f'days to double = {d2d} => exponent = {exponent}')

# d2d = days_to_double(exponent)
# print(f'exponent={exp_test} => days to double = {d2d}')

#%% data sources

daily_data_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_daily_reports/03-25-2020.csv"

#%% fetch data
daily_data_df = dataframe_from_csv_url(daily_data_url)


#%% parse data
### confirmed cases
confirmed_df[[ADMIN2, STATE_STR, COUNTRY_STR]] = daily_data_df[[ADMIN2, STATE_STR, COUNTRY_STR]].astype('str')

# confirmed_dates = df_confirmed.columns[DATE_COLUMN_START_INDEX:]
dates = confirmed_df.columns[DATE_COLUMN_START_INDEX:]
confirmed_dates = [datetime.strptime(s, "%m/%d/%y") for s in dates]
confirmed_totals = df_confirmed.iloc[:, DATE_COLUMN_START_INDEX:].sum()

confirmed_grouped_by_country = df_confirmed.groupby("Country/Region")
groups = dict(list(confirmed_grouped_by_country))
df_confirmed_italy = df_confirmed[df_confirmed[COUNTRY_STR] == 'Italy']
df_confirmed_france = df_confirmed[df_confirmed[COUNTRY_STR] == 'France']
df_confirmed_us = df_confirmed[df_confirmed[COUNTRY_STR] == 'US']

confirmed_totals_china = groups['China'].iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_other = confirmed_totals - confirmed_totals_china
confirmed_totals_italy = df_confirmed_italy.iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_france = df_confirmed_france.iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_us = df_confirmed_us.iloc[:, DATE_COLUMN_START_INDEX:].sum()

# 2020.03.26 state level data eliminated from Johns Hopkins dataset
# df_confirmed_ny = df_confirmed[df_confirmed[STATE_STR].str.contains("NY|New York", na=False)]
# df_confirmed_ma = df_confirmed[df_confirmed[STATE_STR].str.contains("MA|Massachusetts", na=False)]
# df_confirmed_nj = df_confirmed[df_confirmed[STATE_STR].str.contains("NJ|New Jersey", na=False)]
# confirmed_totals_ma = df_confirmed_ma.iloc[:, DATE_COLUMN_START_INDEX:].sum()
# confirmed_totals_ny = df_confirmed_ny.iloc[:, DATE_COLUMN_START_INDEX:].sum()
# confirmed_totals_nj = df_confirmed_nj.iloc[:, DATE_COLUMN_START_INDEX:].sum()
# ma_counties = df_confirmed_ma[STATE_STR].astype('str')
# grouped_by_ma_county = df_confirmed_ma.groupby(STATE_STR)
# groups_ma = dict(list(grouped_by_ma_county))

### deaths
df_deaths[[STATE_STR, COUNTRY_STR]] = df_deaths[[STATE_STR, COUNTRY_STR]].astype('str')
df_deaths_us = df_deaths[df_deaths[COUNTRY_STR] == 'US']

dates = df_deaths.columns[DATE_COLUMN_START_INDEX:]
deaths_dates = [datetime.strptime(s, "%m/%d/%y") for s in dates]
deaths_totals = df_deaths.iloc[:, DATE_COLUMN_START_INDEX:].sum()

deaths_grouped_by_country = df_deaths.groupby(COUNTRY_STR)
death_groups_countries = dict(list(deaths_grouped_by_country))
deaths_totals_china = death_groups_countries['China'].iloc[:, DATE_COLUMN_START_INDEX:].sum()
deaths_totals_other = deaths_totals - deaths_totals_china
deaths_totals_italy = death_groups_countries['Italy'].iloc[:, DATE_COLUMN_START_INDEX:].sum()
deaths_totals_france = death_groups_countries['France'].iloc[:, DATE_COLUMN_START_INDEX:].sum()
deaths_totals_us = death_groups_countries['US'].iloc[:, DATE_COLUMN_START_INDEX:].sum()

# 2020.03.26 state level data eliminated from Johns Hopkins dataset
# df_deaths_ny = df_deaths[df_deaths[STATE_STR].str.contains("NY|New York", na=False)]
# df_deaths_ma = df_deaths[df_deaths[STATE_STR].str.contains("MA|Massachusetts", na=False)]
# df_deaths_nj = df_deaths[df_deaths[STATE_STR].str.contains("NJ|New Jersey", na=False)]
# deaths_totals_ma = df_deaths_ma.iloc[:, DATE_COLUMN_START_INDEX:].sum()
# deaths_totals_ny = df_deaths_ny.iloc[:, DATE_COLUMN_START_INDEX:].sum()
# deaths_totals_nj = df_deaths_nj.iloc[:, DATE_COLUMN_START_INDEX:].sum()

# deaths_grouped_by_state = df_deaths.groupby(STATE_STR)
# death_groups_states = dict(list(deaths_grouped_by_state))

#%% Plots!
### confirmed cases
# plt.figure(figsize=figure_size)
# plt.scatter(confirmed_dates, confirmed_totals, label='Total')
# plt.scatter(confirmed_dates, confirmed_totals_china, c='m', label='Mainland China')
# plt.scatter(confirmed_dates, confirmed_totals_other, c='k', label='all rest of the world')
# plt.scatter(confirmed_dates, confirmed_totals_italy, c='g', label='Italy only')
# plt.scatter(confirmed_dates, confirmed_totals_france, c='r', label='France only')
# plt.scatter(confirmed_dates, confirmed_totals_us, c='b', label='US only')
# # plt.ylim(0, totals.max())
# plt.xticks(rotation=rotation_angle)
# plt.yscale('log')
# plt.xlim(confirmed_dates[0], confirmed_dates[-1])
# plt.ylim(1, 1e6)
# plt.grid(b=True, which='both', axis='both')
# plt.legend()
# plt.xlabel("date")
# plt.ylabel("number of confirmed cases")
# plt.title("Covid-19 confirmed cases")
# plt.show()

# ### deaths
# plt.figure(figsize=figure_size)
# plt.scatter(deaths_dates, deaths_totals, label='Total')
# plt.scatter(deaths_dates, deaths_totals_china, c='m', label='Mainland China')
# plt.scatter(deaths_dates, deaths_totals_other, c='k', label='all rest of the world')
# plt.scatter(deaths_dates, deaths_totals_italy, c='g', label='Italy only')
# plt.scatter(deaths_dates, deaths_totals_france, c='r', label='France only')
# plt.scatter(deaths_dates, deaths_totals_us, c='b', label='US only')
# # plt.ylim(0, totals.max())
# plt.xticks(rotation=rotation_angle)
# plt.yscale('log')
# plt.xlim(confirmed_dates[0], confirmed_dates[-1])
# plt.ylim(1, 2 * deaths_totals.max())
# plt.grid(b=True, which='both', axis='both')
# plt.legend()
# plt.xlabel("date")
# plt.ylabel("number of deaths")
# plt.title("Covid-19 deaths")
# plt.show()

# ### US
# plt.figure(figsize=figure_size)
# plt.plot(confirmed_dates, confirmed_totals_us, c='b', label="US confirmed cases")    
# plt.plot(deaths_dates, deaths_totals_us, c='r', label="US deaths")    
# plt.xticks(rotation=rotation_angle)
# # plt.yscale('log')
# plt.grid(b=True, which='both', axis='both')
# plt.legend()
# plt.yscale('log')
# plt.xlabel("date")
# plt.ylabel("numbers")
# plt.title("Covid-19 confirmed cases and deaths in US")
# plt.show()

### MA, NY, NJ
# 2020.03.26 state level data eliminated from Johns Hopkins dataset
# confirmed_dates_start = "3/1/20"
# confirmed_dates_start_dt = datetime.strptime(confirmed_dates_start, "%m/%d/%y")
# confirmed_dates_start_index = confirmed_dates.index(confirmed_dates_start_dt)

# plt.figure(figsize=figure_size)
# plt.plot(confirmed_dates, confirmed_totals_ma, c='b', marker='o', label="MA confirmed cases")    
# plt.plot(deaths_dates, deaths_totals_ma, c='c', marker='o', label="MA deaths")    
# plt.plot(confirmed_dates, confirmed_totals_ny, c='firebrick', marker='o', label="NY confirmed cases")    
# plt.plot(deaths_dates, deaths_totals_ny, c='r', marker='o', label="NY deaths")    
# plt.plot(confirmed_dates, confirmed_totals_nj, c='darkgreen', marker='o', label="NJ confirmed cases")    
# plt.plot(deaths_dates, deaths_totals_nj, c='limegreen', marker='o', label="NJ deaths")
# plt.xlim(confirmed_dates[confirmed_dates_start_index], confirmed_dates[-1])
# plt.ylim(1e0, 2 * confirmed_totals_ny.max())
# plt.yscale('log')
# plt.xticks(rotation=rotation_angle)
# plt.grid(b=True, which='both', axis='both')
# plt.legend()
# plt.xlabel("date")
# plt.ylabel("numbers")
# plt.title("Covid-19 confirmed cases and deaths in MA, NY, and NJ")
# plt.show()



