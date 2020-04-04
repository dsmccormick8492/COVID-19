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

### project imports
from dataframe_from_csv_url import dataframe_from_csv_url
import doubling
import mcmc_regression

### some constants
DATE_COLUMN_START_INDEX = 4

figure_size = (10, 6.5)
rotation_angle = 0

COUNTRY_STR = 'Country/Region'
STATE_STR = 'Province/State'

#%% data sources
confirmed_url = "https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
deaths_url = "https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
recovered_url = "https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

#%% fetch data
df_confirmed = dataframe_from_csv_url(confirmed_url)
df_deaths = dataframe_from_csv_url(deaths_url)
df_recovered = dataframe_from_csv_url(recovered_url)

#%% parse data
### confirmed cases
df_confirmed[[STATE_STR, COUNTRY_STR]] = df_confirmed[[STATE_STR, COUNTRY_STR]].astype('str')

# confirmed_dates = df_confirmed.columns[DATE_COLUMN_START_INDEX:]
dates = df_confirmed.columns[DATE_COLUMN_START_INDEX:]
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

#%% Plots!
### confirmed cases
plt.figure(figsize=figure_size)
plt.plot(confirmed_dates, confirmed_totals, label='Total')
plt.plot(confirmed_dates, confirmed_totals_china, c='m', label='Mainland China')
plt.plot(confirmed_dates, confirmed_totals_other, c='k', label='all rest of the world')
plt.plot(confirmed_dates, confirmed_totals_italy, c='g', label='Italy only')
plt.plot(confirmed_dates, confirmed_totals_france, c='r', label='France only')
plt.plot(confirmed_dates, confirmed_totals_us, c='b', label='US only')
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
plt.plot(deaths_dates, deaths_totals, label='Total')
plt.plot(deaths_dates, deaths_totals_china, c='m', label='Mainland China')
plt.plot(deaths_dates, deaths_totals_other, c='k', label='all rest of the world')
plt.plot(deaths_dates, deaths_totals_italy, c='g', label='Italy only')
plt.plot(deaths_dates, deaths_totals_france, c='r', label='France only')
plt.plot(deaths_dates, deaths_totals_us, c='b', label='US only')
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

#%% linear regression: confirmed cases
confirmed_dates_start = "3/1/20"
confirmed_dates_start_dt = datetime.strptime(confirmed_dates_start, "%m/%d/%y")

confirmed_dates_start_index = confirmed_dates.index(confirmed_dates_start_dt)
confirmed_dates_subset = confirmed_dates[confirmed_dates_start_index:]
confirmed_dates_subset_count = len(confirmed_dates_subset)
x_confirmed = np.arange(confirmed_dates_subset_count)
y_confirmed = np.log10(confirmed_totals_us[confirmed_dates_start_index:])
slope_confirmed, intercept, r_value, p_value, std_err = stats.linregress(x_confirmed,y_confirmed)
y_hat_confirmed = slope_confirmed * x_confirmed + intercept

doubling_days = doubling.days_to_double(slope_confirmed)
slope_2_days = doubling.double_in_days_exponent(2)
slope_3_days = doubling.double_in_days_exponent(3)

y_double_2_days = slope_2_days * x_confirmed + y_confirmed[0]
y_double_3_days = slope_3_days * x_confirmed + y_confirmed[0]

plt.figure(figsize=(10, 8))
plt.scatter(x_confirmed, y_confirmed, label='data')
plt.plot(x_confirmed, y_hat_confirmed, c='b', label=f'regresson r={r_value:0.3f}, slope={slope_confirmed:0.3f}, days to double={doubling_days:0.1f}')
plt.plot(x_confirmed, y_double_2_days, c='k', linestyle='--', label="double in 2 days")
plt.plot(x_confirmed, y_double_3_days, c='k', linestyle='-.', label="double in 3 days")
plt.xlabel(f"days since {confirmed_dates_start}")
plt.ylabel("log10(confirmed cases)")
plt.legend()
plt.title(f"linear regression of log10(confirmed US cases) since {confirmed_dates_start}")
plt.show()

plt.figure(figsize=(10, 6.5))
plt.scatter(confirmed_dates_subset, 10**y_confirmed, label='data')
plt.plot(confirmed_dates_subset, 10**y_hat_confirmed, c='b', label=f'regression r={r_value:0.3f}, days to double={doubling_days:0.1f}')
plt.plot(confirmed_dates_subset, 10**y_double_2_days, c='k', linestyle='--', label="double in 2 days")
plt.plot(confirmed_dates_subset, 10**y_double_3_days, c='k', linestyle='-.', label="double in 3 days")
plt.xlim(confirmed_dates_start_dt, confirmed_dates[-1])
plt.xlabel(f"days since {confirmed_dates_start}")
plt.xticks(rotation=rotation_angle)
plt.ylabel("confirmed cases")
plt.legend()
plt.title(f"regression of confirmed US cases since {confirmed_dates_start}")
plt.show()

#%% linear regression: deaths
# deaths_dates_start = "3/1/20"
deaths_dates_start = "3/15/20"
deaths_dates_start_dt = datetime.strptime(deaths_dates_start, "%m/%d/%y")

deaths_dates_start_index = deaths_dates.index(deaths_dates_start_dt)
deaths_dates_subset = deaths_dates[deaths_dates_start_index:]
deaths_dates_subset_count = len(deaths_dates_subset)
x_deaths = np.arange(deaths_dates_subset_count)
y_deaths = np.log10(deaths_totals_us[deaths_dates_start_index:])
slope_deaths, intercept, r_value, p_value, std_err = stats.linregress(x_deaths,y_deaths)
y_hat_deaths = slope_deaths * x_deaths + intercept

doubling_days = doubling.days_to_double(slope_deaths)
y_double_2_days = slope_2_days * x_deaths + y_deaths[0]
y_double_3_days = slope_3_days * x_deaths + y_deaths[0]

plt.figure(figsize=(10, 8))
plt.scatter(x_deaths, y_deaths, c='r', label='data')
plt.plot(x_deaths, y_hat_deaths, c='r', label=f'regression r={r_value:0.3f}, slope={slope_deaths:0.3f}, days to double={doubling_days:0.1f}')
plt.plot(x_deaths, y_double_2_days, c='k', linestyle='--', label="double in 2 days")
plt.plot(x_deaths, y_double_3_days, c='k', linestyle='-.', label="double in 3 days")
plt.xlabel(f"days since {deaths_dates_start}")
plt.ylabel("log10(deaths)")
plt.legend()
plt.title(f"linear regression of log10(US deaths) since {deaths_dates_start}")
plt.show()

plt.figure(figsize=(10, 6.5))
plt.scatter(deaths_dates_subset, 10**y_deaths, c='r', label='data')
plt.plot(deaths_dates_subset, 10**y_hat_deaths, c='r', label=f'regression r={r_value:0.3f}, days to double={doubling_days:0.1f}')
plt.plot(deaths_dates_subset, 10**y_double_2_days, c='k', linestyle='--', label="double in 2 days")
plt.plot(deaths_dates_subset, 10**y_double_3_days, c='k', linestyle='-.', label="double in 3 days")
plt.xlim(deaths_dates_start_dt, deaths_dates[-1])
plt.xlabel(f"days since {deaths_dates_start}")
plt.xticks(rotation=rotation_angle)
plt.ylabel("deaths")    
plt.legend()
plt.title(f"regression of US deaths since {deaths_dates_start}")
plt.show()

#%% pymc3 MCMC model for regression for deaths
mcmc_regression.mcmc_log_regression(x_deaths, y_deaths, y_hat_deaths, slope_deaths, "US deaths")

#%% pymc3 MCMC model for regression for confirmed cases
# mcmc_regression.mcmc_log_regression(x_confirmed, y_confirmed, y_hat_confirmed, slope_confirmed, "US confirmed cases")
