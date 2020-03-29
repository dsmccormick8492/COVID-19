#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: days_to_double
    
Purpose:
    
Description:
    
Comments:

TODO:
    
@author: dmccormick
Author: David S. McCormick
Created on Sat Mar 28 13:08:48 2020
"""
### import modules
import math

#%% functions
def days_to_double(exponent: float) -> float:
    return math.log(2) / math.log(10**exponent)


def double_in_days_exponent(days_to_double: int) -> float:
    exponent = math.log10(2) / days_to_double
    
    return exponent


def main():
    d2d = 1.0
    exponent = double_in_days_exponent(d2d)
    print(f'days to double = {d2d} => exponent = {exponent}')
    
    d2d = days_to_double(exponent)
    print(f'exponent={exponent} => days to double = {d2d}')


if __name__ == '__main__':
	main()


