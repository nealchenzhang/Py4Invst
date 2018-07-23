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

import os
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import add_trend

import arch
from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import seaborn
from matplotlib import style
style.use('ggplot')


class TSAnalysis(object):
    """
    This class is a process for analyzing a given time-series investment problem and justification.
    
    """
    
    def __init__(self, df_data):
        """

        :param df_data: normally time-series data
        """
        self.data = df_data.copy()
        self.data = add_trend(self.data, trend='ct')

    def plot_linear_trend(self, y):
        seaborn.lmplot(x='trend', y=y, data=self.data)
        ax = plt.gca()
        ax.set_xlim(left=1)
        self.data.plot(x='trend', y=y, ax=ax)
        plt.title('Constant increase: Linear trend model')
        plt.show()

    def plot_log_linear_trend(self, y):
        log_y = 'Log_' + y
        self.data.loc[:, log_y] = self.data.loc[:, y].apply(np.log)
        seaborn.lmplot(x='trend', y=log_y, data=self.data)
        ax = plt.gca()
        ax.set_xlim(left=1)
        self.data.plot(x='trend', y=log_y, ax=ax)
        plt.title('Constant rate: Log-Linear trend model')
        plt.show()
        
    def ols(self, ls_x, y):
        x = sm.add_constant(self.data.loc[:, ls_x])
        y = self.data.loc[:, y]
        model = sm.OLS(y, x).fit()
        self.model = model
        print(model.summary())
        return model
            
    def durbin_watson(self, model):
        resid = model.resid
        DW = (resid - resid.shift(1)).apply(np.square).sum() / \
             resid.apply(np.square).sum()
        print('Durbin_Watson: ', DW)

    # def ADF_test(self, df_ts, lags=None):
    #     if lags == 'None':
    #         try:
    #             adf = ADF(df_ts)
    #         except:
    #             adf = ADF(df_ts.dropna())
    #     else:
    #         try:
    #             adf = ADF(df_ts)
    #         except:
    #             adf = ADF(df_ts.dropna())
    #         adf.lags = lags
    #     print(adf.summary().as_text())
    #     return adf
    #
    # def plot_acf_pacf(self, df_ts, lags=31):
    #     f = plt.figure(facecolor='white', figsize=(12, 8))
    #     ax1 = f.add_subplot(211)
    #     plot_acf(df_ts, lags=31, ax=ax1)
    #     ax2 = f.add_subplot(212)
    #     plot_pacf(df_ts, lags=31, ax=ax2)
    #     plt.show()

    
if __name__ == '__main__':
    # df_data = pd.read_excel('./Market_Analysis/Market_Analysis_Tools/cointegration.xls')
    df_data = pd.read_excel('cointegration.xls')
    df_data.set_index('Date', inplace=True)

    # Data Preparation
    df_ts = df_data.drop(columns='rb1701')

    # Establish TSAnalysis class
    ex_a = TSAnalysis(df_ts)

    # Plot Linear Trend Model
    ex_a.plot_linear_trend('i1701')
    # OLS for linear trend model
    l_model = ex_a.ols(ls_x=['trend'], y='i1701')
    ex_a.durbin_watson(model=l_model)
    print('-'*50)
    print('If DW(!=2) shows that autocorrelation exists, then try log_linear_trend model.')

    # Plot log_linear_trend model
    ex_a.plot_log_linear_trend('i1701')
    # OLS for log linear trend model
    lg_model = ex_a.ols(ls_x=['trend'], y='Log_i1701')
    ex_a.durbin_watson(model=lg_model)
    print('-' * 50)
    print('If DW(!=2) still shows that autocorrelation exists, then try autoregressive model.')

    # Covariance stationary test (ADF test)

    # Autoregressive Model AR(p)
