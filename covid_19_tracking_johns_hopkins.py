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

from math import log
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

#%% functions
def days_to_double(exponent: float) -> float:
    days = log(2) / log(10**exponent)
    
    return days


# def double_in_days_exponent(days_to_double: int) -> float:
#     a = days_to_double
#     # WolframAlpha solution
#     # y = (log(2^(1/a)))/log(10)
#     # exponent = (log(2**(1 / days_to_double))) / log(10)
#     exponent = log(2) / (days_to_double * log(10))
    
#     return exponent

# tests
# exp_test = 0.301
# d2d = days_to_double(exp_test)
# print(f'exponent={exp_test} => days to double = {d2d}')

# d2d = 2.0
# print(f'days to double = {d2d} => exponent = {double_in_days_exponent(d2d)}')


#%% data sources

confirmed_url = "https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
deaths_url = "https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

#%% fetch data
df_confirmed = dataframe_from_csv_url(confirmed_url)
df_deaths = dataframe_from_csv_url(deaths_url)

#%% parse data
### confirmed cases
df_confirmed[[STATE_STR, COUNTRY_STR]] = df_confirmed[[STATE_STR, COUNTRY_STR]].astype('str')

# confirmed_dates = df_confirmed.columns[DATE_COLUMN_START_INDEX:]
dates = df_confirmed.columns[DATE_COLUMN_START_INDEX:]
confirmed_dates = [datetime.strptime(s, "%m/%d/%y") for s in dates]
confirmed_totals = df_confirmed.iloc[:, DATE_COLUMN_START_INDEX:].sum()

confirmed_grouped_by_country = df_confirmed.groupby("Country/Region")
groups = dict(list(confirmed_grouped_by_country))
confirmed_totals_china = groups['China'].iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_other = confirmed_totals - confirmed_totals_china
df_confirmed_italy = df_confirmed[df_confirmed[COUNTRY_STR] == 'Italy']
df_confirmed_france = df_confirmed[df_confirmed[COUNTRY_STR] == 'France']
df_confirmed_us = df_confirmed[df_confirmed[COUNTRY_STR] == 'US']
df_confirmed_ny = df_confirmed[df_confirmed[STATE_STR].str.contains("NY|New York", na=False)]
df_confirmed_ma = df_confirmed[df_confirmed[STATE_STR].str.contains("MA|Massachusetts", na=False)]
df_confirmed_nj = df_confirmed[df_confirmed[STATE_STR].str.contains("NJ|New Jersey", na=False)]
confirmed_totals_italy = df_confirmed_italy.iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_france = df_confirmed_france.iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_us = df_confirmed_us.iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_ma = df_confirmed_ma.iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_ny = df_confirmed_ny.iloc[:, DATE_COLUMN_START_INDEX:].sum()
confirmed_totals_nj = df_confirmed_nj.iloc[:, DATE_COLUMN_START_INDEX:].sum()
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
df_deaths_ny = df_deaths[df_deaths[STATE_STR].str.contains("NY|New York", na=False)]
df_deaths_ma = df_deaths[df_deaths[STATE_STR].str.contains("MA|Massachusetts", na=False)]
df_deaths_nj = df_deaths[df_deaths[STATE_STR].str.contains("NJ|New Jersey", na=False)]
deaths_totals_ma = df_deaths_ma.iloc[:, DATE_COLUMN_START_INDEX:].sum()
deaths_totals_ny = df_deaths_ny.iloc[:, DATE_COLUMN_START_INDEX:].sum()
deaths_totals_nj = df_deaths_nj.iloc[:, DATE_COLUMN_START_INDEX:].sum()

deaths_grouped_by_state = df_deaths.groupby(STATE_STR)
death_groups_states = dict(list(deaths_grouped_by_state))

#%% Plots!
### confirmed cases
plt.figure(figsize=figure_size)
plt.scatter(confirmed_dates, confirmed_totals, label='Total')
plt.scatter(confirmed_dates, confirmed_totals_china, c='m', label='Mainland China')
plt.scatter(confirmed_dates, confirmed_totals_other, c='k', label='all rest of the world')
plt.scatter(confirmed_dates, confirmed_totals_italy, c='g', label='Italy only')
plt.scatter(confirmed_dates, confirmed_totals_france, c='r', label='France only')
plt.scatter(confirmed_dates, confirmed_totals_us, c='b', label='US only')
# plt.ylim(0, totals.max())
plt.xticks(rotation=rotation_angle)
plt.yscale('log')
plt.xlim(confirmed_dates[0], confirmed_dates[-1])
plt.ylim(1, 1e6)
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.xlabel("date")
plt.ylabel("number of confirmed cases")
plt.title("Covid-19 confirmed cases")
plt.show()

### deaths
plt.figure(figsize=figure_size)
plt.scatter(deaths_dates, deaths_totals, label='Total')
plt.scatter(deaths_dates, deaths_totals_china, c='m', label='Mainland China')
plt.scatter(deaths_dates, deaths_totals_other, c='k', label='all rest of the world')
plt.scatter(deaths_dates, deaths_totals_italy, c='g', label='Italy only')
plt.scatter(deaths_dates, deaths_totals_france, c='r', label='France only')
plt.scatter(deaths_dates, deaths_totals_us, c='b', label='US only')
# plt.ylim(0, totals.max())
plt.xticks(rotation=rotation_angle)
plt.yscale('log')
plt.xlim(confirmed_dates[0], confirmed_dates[-1])
plt.ylim(1, 2 * deaths_totals.max())
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.xlabel("date")
plt.ylabel("number of deaths")
plt.title("Covid-19 deaths")
plt.show()

### US
plt.figure(figsize=figure_size)
plt.plot(confirmed_dates, confirmed_totals_us, c='b', label="US confirmed cases")    
plt.plot(deaths_dates, deaths_totals_us, c='r', label="US deaths")    
plt.xticks(rotation=rotation_angle)
# plt.yscale('log')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.yscale('log')
plt.xlabel("date")
plt.ylabel("numbers")
plt.title("Covid-19 confirmed cases and deaths in US")
plt.show()


### MA, NY, NJ
confirmed_dates_start = "3/1/20"
confirmed_dates_start_dt = datetime.strptime(confirmed_dates_start, "%m/%d/%y")
confirmed_dates_start_index = confirmed_dates.index(confirmed_dates_start_dt)

plt.figure(figsize=figure_size)
plt.plot(confirmed_dates, confirmed_totals_ma, c='b', marker='o', label="MA confirmed cases")    
plt.plot(deaths_dates, deaths_totals_ma, c='c', marker='o', label="MA deaths")    
plt.plot(confirmed_dates, confirmed_totals_ny, c='firebrick', marker='o', label="NY confirmed cases")    
plt.plot(deaths_dates, deaths_totals_ny, c='r', marker='o', label="NY deaths")    
plt.plot(confirmed_dates, confirmed_totals_nj, c='darkgreen', marker='o', label="NJ confirmed cases")    
plt.plot(deaths_dates, deaths_totals_nj, c='limegreen', marker='o', label="NJ deaths")
plt.xlim(confirmed_dates[confirmed_dates_start_index], confirmed_dates[-1])
plt.ylim(1e0, 2 * confirmed_totals_ny.max())
plt.yscale('log')
plt.xticks(rotation=rotation_angle)
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.xlabel("date")
plt.ylabel("numbers")
plt.title("Covid-19 confirmed cases and deaths in MA, NY, and NJ")
plt.show()

#%% linear regression

### confirmed start date
confirmed_dates_start = "3/1/20"
confirmed_dates_start_dt = datetime.strptime(confirmed_dates_start, "%m/%d/%y")

confirmed_dates_start_index = confirmed_dates.index(confirmed_dates_start_dt)
confirmed_dates_subset = confirmed_dates[confirmed_dates_start_index:]
confirmed_dates_subset_count = len(confirmed_dates_subset)
x = np.arange(confirmed_dates_subset_count)
y = np.log10(confirmed_totals_us[confirmed_dates_start_index:])
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
y_hat = slope * x + intercept

doubling_days = days_to_double(slope)

#TODO: add lines for days to double
plt.figure(figsize=(10, 8))
plt.scatter(x, y, label='data')
plt.plot(x, y_hat, c='b', label=f'regression r={r_value:0.3f}, slope={slope:0.3f}, days to double={doubling_days:0.1f}')
plt.xlabel(f"days since {confirmed_dates_start}")
plt.ylabel("log10(confirmed cases)")
plt.legend()
plt.title(f"linear regression of log10(confirmed US cases) since {confirmed_dates_start}")
plt.show()

plt.figure(figsize=(10, 6.5))
plt.scatter(confirmed_dates_subset, 10**y, label='data')
plt.plot(confirmed_dates_subset, 10**y_hat, c='b', label=f'regression r={r_value:0.3f}, days to double={doubling_days:0.1f}')
plt.xlim(confirmed_dates_start_dt, confirmed_dates[-1])
plt.xlabel(f"days since {confirmed_dates_start}")
plt.xticks(rotation=rotation_angle)
plt.ylabel("confirmed cases")
plt.legend()
plt.title(f"regression of confirmed US cases since {confirmed_dates_start}")
plt.show()

### deaths start date
#TODO: regression on death rates
# deaths_dates_start = "3/3/20"
# deaths_dates_start_dt = datetime.strptime(deaths_dates_start, "%m/%d/%y")

# deaths_dates_start_index = deaths_dates.index(deaths_dates_start_dt)
# deaths_dates_subset = deaths_dates[deaths_dates_start_index:]
# deaths_dates_subset_count = len(deaths_dates_subset)
# x = np.arange(deaths_dates_subset_count)
# y = np.log10(deaths_totals_us[deaths_dates_start_index:])
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# y_hat = slope * x + intercept

# plt.figure(figsize=(10, 8))
# plt.scatter(x, y, c='r', label='data')
# plt.plot(x, y_hat, c='r', label=f'regression r={r_value:0.3f}, slope={slope:0.3f}')
# plt.xlabel(f"days since {deaths_dates_start}")
# plt.ylabel("log10(deaths)")
# plt.legend()
# plt.title(f"linear regression of log10(US deaths) since {deaths_dates_start}")
# plt.show()

# plt.figure(figsize=(10, 6.5))
# plt.scatter(deaths_dates_subset, 10**y, c='r', label='data')
# plt.plot(deaths_dates_subset, 10**y_hat, c='r', label=f'regression r={r_value:0.3f}')
# plt.xlim(deaths_dates_start_dt, deaths_dates[-1])
# plt.xlabel(f"days since {deaths_dates_start}")
# plt.xticks(rotation=rotation_angle)
# plt.ylabel("deaths")
# plt.legend()
# plt.title(f"regression of US deaths since {deaths_dates_start}")
# plt.show()

#%% pymc3 MCMC model for regression
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
title_str = f"{sample_count} posterior predictive samples of MCMC fit to log10(confirmed cases) since {confirmed_dates_start}"
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
title_str = f"{sample_count} posterior predictive samples of MCMC fit to log10(confirmed cases) since {confirmed_dates_start}"
plt.title(title_str)
plt.show()


