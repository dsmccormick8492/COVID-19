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
Created on Mon Mar 23 09:10:42 2020
"""
### import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

import urllib.request
from io import StringIO
from datetime import datetime

ENCODING_TYPE = 'utf-8'
date_column_start_index = 4


def dataframe_from_url(url_str: str) -> pd.DataFrame:
    with urllib.request.urlopen(url_str) as response:
       csv_bytes = response.read()
    
    string = str(csv_bytes, ENCODING_TYPE)
    data = StringIO(string)
    df = pd.read_csv(data)    
    
    return df

#%% data sources

confirmed_url = "https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
deaths_url = "https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
recovered_url = "https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"


#%% fetch data
confirmed_df = dataframe_from_url(confirmed_url)
deaths_df = dataframe_from_url(deaths_url)
recovered_df = dataframe_from_url(recovered_url)


#%% dates to datetimes

datetime_strings = [s for s in confirmed_df.columns[date_column_start_index:]]
datetimes = [datetime.strptime(str(s), "%m/%d/%y") for s in datetime_strings]
# confirmed_df['datetime'] = datetimes