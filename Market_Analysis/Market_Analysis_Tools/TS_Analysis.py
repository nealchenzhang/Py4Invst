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
import re
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as st
from statsmodels.tsa.stattools import add_trend
from statsmodels.tsa.stattools import add_constant
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR

import arch
from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import prettytable as pt
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
        self.lag = None
        self.lagged_data = None
        self.sp_lag = None
        self.lagged_data_w_sp_lag = None

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
        print(model.summary())
        return model

    @staticmethod
    def durbin_watson(model):
        """

        :param model: fitted model from statsmodels library
        :return:
        """
        resid = model.resid
        DW = (resid - resid.shift(1)).apply(np.square).sum() / \
             resid.apply(np.square).sum()
        print('Durbin_Watson: ', DW)

    def add_lag(self, x, lag=1):
        lagged_data = pd.DataFrame(columns=[str(x)])
        lagged_data.loc[:, x] = self.data.loc[:, x].copy()
        ls_lags = []
        dic = {}
        for i in range(1, lag+1):
            dic['lag_' + str(i) + '_' + str(x)] = i

        for i in dic.keys():
            locals()[i] = dic[i]
            ls_lags.append(i)

        n = 1
        for i in ls_lags:
            lagged_data.loc[:, i] = self.data.loc[:, x].shift(n)
            n += 1

        lagged_data.dropna(inplace=True)
        self.lag = lag
        self.lagged_data = lagged_data
        return lagged_data

    def add_sp_lag(self, x, sp_lag=None):
        """

        :param x:
        :param sp_lag: one int number
        :return:
        """
        print('Current model is AR'+str(self.lag) + '.')
        lagged_data_w_sp_lag = self.add_lag(x, lag=self.lag).copy()
        ix_size = lagged_data_w_sp_lag.index.size

        ls_sp_lags = []

        dic = {}
        dic['sp_lag_' + str(sp_lag) + '_' + str(x)] = sp_lag
        for i in dic.keys():
            locals()[i] = dic[i]
            ls_sp_lags.append(i)

        for i in ls_sp_lags:
            lagged_data_w_sp_lag.loc[:, i] = self.data.loc[:, x].shift(dic[i]).iloc[-ix_size:]

        lagged_data_w_sp_lag.dropna(inplace=True)

        self.lagged_data_w_sp_lag = lagged_data_w_sp_lag
        self.sp_lag = sp_lag
        return lagged_data_w_sp_lag

    def AR_p(self, x, p=1, method='ols'):
        lagged_data = self.add_lag(x, lag=p).copy()
        lagged_data = add_constant(lagged_data)
        ls_x = []
        for i in range(1, p+1):
            ls_x.append('lag_'+str(i)+'_'+str(x))
        ls_x.append('const')
        self.lagged_data = lagged_data
        if method == 'ols':
            model = sm.OLS(endog=lagged_data.loc[:, x], exog=lagged_data.loc[:, ls_x]).fit()
        elif method == 'stats': # TODO check source code and algorithm
            model = ARMA(self.data.loc[:, x], order=(p, 0)).fit()
        # print(model.summary())
        return model
    # TODO: acf calculation and t-statistics

    @staticmethod
    def acf(model, maxlag=12):
        """

        :param model: fitted model from
        :param maxlag:
        :return:
        """
        df_res = pd.DataFrame(columns=['resid'])
        df_res.loc[:, 'resid'] = model.resid

        ls_lags = []
        dic = {}
        for i in range(1, maxlag + 1):
            dic['resid_lag' + str(i)] = i

        for i in dic.keys():
            locals()[i] = dic[i]
            ls_lags.append(i)

        n = 1
        for i in ls_lags:
            df_res.loc[:, i] = df_res.loc[:, 'resid'].shift(n)
            n += 1

        df_res.dropna(inplace=True)

        mean = df_res.loc[:, 'resid'].mean()
        dic = {}
        for i in ls_lags:
            dic[i] = ((df_res.loc[:, 'resid'] - mean) * (df_res.loc[:, i] - mean)).sum() /\
                     (df_res.loc[:, 'resid'] - mean).apply(np.square).sum()
        return dic

    def acf_table(self, model, maxlag=12):
        dic = self.acf(model, maxlag)
        T = model.resid.size
        tb = pt.PrettyTable()
        tb.set_style(pt.PLAIN_COLUMNS)
        tb.align = 'c'
        tb.field_names = ['Residual Lag', 'Autocorrelation', 't-Statistic']
        for i in dic.keys():
            tb.add_row([int(re.findall('\d+', i)[0]), dic[i], dic[i]/(1/np.sqrt(T))])
        print(tb)

    ##############################################################
    # @staticmethod
    # TODO less RMSE better predictive power
    # def RMSE(model, out_of_sample):
    #     RMSE = np.sqrt((real_y - predict_y).apply(np.square).sum())
    ##############################################################

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

    # def ARCH_test(self):


    
if __name__ == '__main__':
    # df_data = pd.read_excel('./Market_Analysis/Market_Analysis_Tools/cointegration.xls')
    df_data = pd.read_excel('cointegration.xls')
    df_data.set_index('Date', inplace=True)

    # Data Preparation
    df_ts = df_data.drop(columns='rb1701')

    # Establish TSAnalysis class
    ex_a = TSAnalysis(df_ts)

    # # Plot Linear Trend Model
    # ex_a.plot_linear_trend('i1701')
    # # OLS for linear trend model
    # l_model = ex_a.ols(ls_x=['trend'], y='i1701')
    # ex_a.durbin_watson(model=l_model)
    # print('-'*50)
    # print('If DW(!=2) shows that autocorrelation exists, then try log_linear_trend model.')
    #
    # # Plot log_linear_trend model
    # ex_a.plot_log_linear_trend('i1701')
    # # OLS for log linear trend model
    # lg_model = ex_a.ols(ls_x=['trend'], y='Log_i1701')
    # ex_a.durbin_watson(model=lg_model)
    # print('-' * 50)
    # print('If DW(!=2) still shows that autocorrelation exists, then try autoregressive model.')
    #
    # # Covariance stationary test (ADF test)
    #
    # # Autoregressive Model AR(p)
    # # Simple AR1 model
    AR_1_model = ex_a.AR_p(x='i1701', p=1)
    df_sp = ex_a.add_sp_lag(x='i1701', sp_lag=4)
    print(df_sp)
    # print(AR_1_model.summary())
    # ex_a.acf(AR_1_model)
    # ex_a.acf_table(AR_1_model, maxlag=12)


    # ############################################
    # dta = sm.datasets.sunspots.load_pandas().data
    # dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    # del dta["YEAR"]
    # arma_mod20 = sm.tsa.ARMA(dta, (2, 0)).fit(disp=False)
    # print(arma_mod20.params)
    # arma = TSAnalysis(dta)
    # ar = arma.AR_p(x='SUNACTIVITY', p=2)
    # ar.summary()
    # arma_mod20.summary()
    #
    # ############################################
    # df_ts_cap = pd.DataFrame(columns=['Capacity'])
    # da = np.array([82.4,81.5,80.8,80.5,80.2,80.2,80.5,80.9,81.3,81.9,81.7,80.3,77.9,76.4,76.4])
    # df_ts_cap.loc[:, 'Capacity'] = pd.Series(da)
    # df_diff = df_ts_cap.diff(1).dropna()
    # cf_test = TSAnalysis(df_diff)
    # mod = cf_test.AR_p(x='Capacity',p=1)
