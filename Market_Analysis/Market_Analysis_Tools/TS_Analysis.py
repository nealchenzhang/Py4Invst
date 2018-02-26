#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
#
# Created on Fri Mar 31 15:57:14 2017

# @author: NealChenZhang

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################
"""
This module is based on Time-series Analysis.

Guidelines:
    1. Determine your goal
        * Cointegrated time-series or cross-sectional multiple regression
        * Trend Model
    2. Plot the values of the variable over time to look for characteristics
        * Heteroskedasticity
        * Non-constant mean
        * Seasonality
        * Structural change
            ** Run two different models:
                one incorporating the data before the date
                one after the data
    3. No seasonality or structural shift --> trend model
        * Upward or downward slope --> linear trend model
        * Curve --> log-linear trend model
    4. Run the trend analysis, compute the residuals and test for serial correlation using Durbin Watson test
        * No serial correlation --> use the model
        * Serial correlation --> another model (e.g., AR)
    5. Serial correlation exists:
        * Plot the data again and see if it's stationary
        * Not stationary:
            ** Linear trend --> first-difference the data
            ** Exponential trend --> first-difference the natural log of the data
            ** Structural shift --> run two separate models
            ** Seasonal component --> incorporate the seasonality in the AR model
    6. Covariance stationary --> AR(1)
        * Test for serial correlation and seasonality
        * No serial correlation --> use the model
        * Serial correlation (Seasonality) --> incorporate lagged values of the variables
    7. Test for ARCH. Regress the square of the residuals on squares of lagged values of the residuals and 
        test whether the resulting coefficient is significantly different from zero.
        * Coefficient is not significantly different from zero --> use the model
        * Coefficient is significantly different from zero --> ARCH exists, use Generalized Least Square(GLS)
    8. Compare the out-of-sample RMSE (the smaller, the better)
"""

print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class TS_Analysis(object):
    """
    This class is a process for analyzing a given time-series investment problem and justification.
    
    """
    
    def __init__(self, df_data):
        self.data = df_data.copy()
        self.data['Time'] = pd.Series(list(range(1, self.data.index.size+1))).values
        
    def TS_Plot(self, y_axis):
        self.data.plot(x='Time', y=y_axis)
        
    def linear_trend_model(self, Y):
        print('Constant increase: Linear trend model')
        self.data.plot(x='Time', y = Y)

    def log_linear_trend_model(self, Y):
        print('Constant rate: Log-Linear trend model')
        log_Y = 'Log Price'+ Y
        self.data[log_Y] = self.data[Y].apply(np.log)
        self.data.plot(x='Time', y=log_Y)
        
    def ols(self, X, Y):
        X = sm.add_constant(self.data[X])
        y = self.data[Y]
        model = sm.OLS(y, X)
        results = model.fit()
        #print(results.summary())
        return results
            
    def Durbin_Watson(self, resid):
        DW = (resid - resid.shift(1)).apply(np.square).sum() /  \
              resid.apply(np.square).sum()
        print('Durbin_Watson: ', DW)
        return DW
    #
    # def ADF_test(self):
    #     adfuller(data, 1)
    
if __name__ == '__main__':
    ex_a = TS_Analysis(df_data)
    ex_a.Durbin_Watson(ex_a.ols('X','Y').resid)
    
    