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
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR

import arch
from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import prettytable as pt
import seaborn
from matplotlib import style
style.use('ggplot')


class TS_Analysis(object):
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

    @staticmethod
    def acf(model, max_lag=12):
        """

        :param model: fitted model from
        :param max_lag:
        :return:
        """
        df_res = pd.DataFrame(columns=['resid'])
        df_res.loc[:, 'resid'] = model.resid

        ls_lags = []
        dic = {}
        for i in range(1, max_lag + 1):
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

    @staticmethod
    def ADF_test(df_ts, lags=None):
        """
        ADF from arch
        formula:
        xt-xt-1 ~ b0 + (b1-1)*xt-1 + e
        test if b1-1 == 0 ~ DF statistics
        :param df_ts:
        :param lags:
        :return:
        """
        if lags == 'None':
            try:
                adf = ADF(df_ts)
            except:
                adf = ADF(df_ts.dropna())
        else:
            try:
                adf = ADF(df_ts)
            except:
                adf = ADF(df_ts.dropna())
            adf.lags = lags
        print(adf.summary().as_text())
        return adf

    @staticmethod
    def diff_selection(df, max_diff=12):
        dict_p = {}
        for i in range(1, max_diff+1):
            tmp = df.copy()
            tmp.loc[:, 'diff'] = tmp.loc[:, tmp.columns[0]].diff(i)
            tmp.dropna(inplace=True)
            pvalue = ADF(tmp.loc[:, 'diff']).pvalue
            dict_p[i] = pvalue
            df_p = pd.DataFrame.from_dict(dict_p, orient="index", columns=['p_value'])
        n = 0
        while n < len(df_p):
            if df_p.loc[:, 'p_value'].iloc[n] < 0.01:
                best_diff = i
                break
            n += 1
        return best_diff

    def plot_acf_pacf(self, df_ts, lags=31):
        f = plt.figure(facecolor='white', figsize=(12, 8))
        ax1 = f.add_subplot(211)
        plot_acf(df_ts, lags=31, ax=ax1)
        ax2 = f.add_subplot(212)
        plot_pacf(df_ts, lags=31, ax=ax2)
        plt.show()

    # def ARCH_test(self):

    ##############################################################
    # @staticmethod
    # TODO less RMSE better predictive power
    # def RMSE(model, out_of_sample):
    #     RMSE = np.sqrt((real_y - predict_y).apply(np.square).sum())
    ##############################################################


if __name__ == '__main__':
    ####################################################################
    #                      PART I Data Preparation                     #
    ####################################################################
    # Read data file
    df_data = pd.read_excel('./Market_Analysis/Market_Analysis_Tools/cointegration.xls')
    # df_data = pd.read_excel('cointegration.xls')
    df_data.set_index('Date', inplace=True)
    df_data = df_data.drop(columns='rb1701')

    # Remove outliers and missing data
    threshold = 300  # Depends on the data characteristics
    df_data.loc[:, 'rolling_median'] = df_data.loc[:, 'i1701'].\
        rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill')
    difference = np.abs(df_data.loc[:, 'i1701'] - df_data.loc[:, 'i1701'])
    outlier_idx = difference > threshold

    df_data.loc[:, 'i1701'] = df_data.loc[:, 'i1701'].fillna(method='bfill').fillna(method='ffill')

    ####################################################################
    #                PART II  Establish Time Series Model              #
    ####################################################################
    # Establish TS_Analysis class
    TS = TS_Analysis(df_data=df_data)

    # Plot Linear Trend Model
    TS.plot_linear_trend('i1701')
    # OLS for linear trend model
    l_model = TS.ols(ls_x=['trend'], y='i1701')
    TS.durbin_watson(model=l_model)
    print('-'*50)
    print('If DW(!=2) shows that autocorrelation exists, then try log_linear_trend model.')

    # Plot log_linear_trend model
    TS.plot_log_linear_trend('i1701')
    # OLS for log linear trend model
    lg_model = TS.ols(ls_x=['trend'], y='Log_i1701')
    TS.durbin_watson(model=lg_model)
    print('-' * 50)
    print('If DW(!=2) still shows that autocorrelation exists, then try autoregressive model.')

    # Covariance stationary test (ADF test)
    # Unit root test
    adf = TS.ADF_test(df_data)
    reg_res = adf.regression
    acf = pd.DataFrame(sm.tsa.stattools.acf(reg_res.resid), columns=['ACF'])
    fig = acf[1:].plot(kind='bar', title='Residual Autocorrelations')

    # If not stationary
    # Method 1: log
    df_log = pd.DataFrame(columns=['log_diff_i1701'])
    df_log = df_data.loc[:, 'i1701'].apply(np.log).dropna()

    # Method 2: diff
    # adf = ex_a.ADF_test(df_ts, lags=adf.lags)
    # reg_res = adf.regression
    acf = pd.DataFrame(sm.tsa.stattools.acf(reg_res.resid), columns=['ACF'])
    fig = acf[1:].plot(kind='bar', title='Residual Autocorrelations')

    # We try to use log-diff data
    df_data_new = df_data.loc[:, 'i1701'].apply(np.log).diff(1).dropna()
    adf = TS.ADF_test(df_data_new)
    reg_res = adf.regression
    acf = pd.DataFrame(sm.tsa.stattools.acf(reg_res.resid), columns=['ACF'])
    fig = acf[1:].plot(kind='bar', title='Residual Autocorrelations')
    # Pass the unit root test

    # TODO: Ljung-Box test
    # Check if acorr_ljungbox(ts, lags=1) source code


    # If AR model is needed and df_data is changed
    TS_new = TS_Analysis(df_data=df_data_new)

    # Autoregressive Model AR(p)
    AR_1_model = TS.AR_p(x='i1701', p=1)
    df_sp = TS.add_sp_lag(x='i1701', sp_lag=4)
    print(AR_1_model.summary())
    TS.acf(AR_1_model)
    TS.acf_table(AR_1_model, maxlag=12)

    # ARMA model
    best_order = st.arma_order_select_ic(df_data, max_ar=5, max_ma=5, ic=['aic', 'bic', 'hqic'])
    arma_model = ARMA(df_data, order=best_order.bic_min_order).fit(disp=-1, method='css')
    print(arma_model.summary())
    ####################################################################
    #              PART III  Model Selection and Prediction            #
    ####################################################################

    # # this is the nsteps ahead predictor function
    # from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
    #
    # res = sm.tsa.ARMA(y, (3, 2)).fit(trend="nc")
    # res = arma_model
    # # get what you need for predicting one-step ahead
    # params = res.params
    # residuals = res.resid
    # p = res.k_ar
    # q = res.k_ma
    # k_exog = res.k_exog
    # k_trend = res.k_trend
    # steps = 1
    #
    # _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=df_data, exog=None, start=len(df_data))
