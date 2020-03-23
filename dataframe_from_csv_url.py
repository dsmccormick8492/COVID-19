#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: dataframe_from_csv_url
    
Purpose: fetch a CSV file on the internet and return a Pandas DataFrame
    
Description:
    
Comments: Assumes that the URL points to a utf-8 encoded CSV file.

TODO:
    
@author: dmccormick
Author: David S. McCormick
Created on Mon Mar 23 09:46:31 2020
"""
### import modules
import pandas as pd
import urllib.request
from io import StringIO

### constants
ENCODING_TYPE = 'utf-8'

### functions
def dataframe_from_csv_url(url_str: str) -> pd.DataFrame:
    with urllib.request.urlopen(url_str) as response:
       csv_bytes = response.read()
    
    string = str(csv_bytes, ENCODING_TYPE)
    data = StringIO(string)
    df = pd.read_csv(data)    
    
    return df