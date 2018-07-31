# -*- coding: utf-8 -*-
__author__ = 'Neal Chen Zhang'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn
import statsmodels.api as sm

import os
import sys
import datetime

def MC_ARMA_GARCH_model(df_returns):
    """

    :param df: NAV time series with columns ['Date','NAV']
    :return:
    """


def MDD(df_data):
    """

    :param df_data: NAV data series
    :return:
    """
    n = df_data.size
    highwatermark = pd.Series(np.zeros(n),
                              index=df_data.index)
    drawdown = pd.Series(np.zeros(n),
                         index=df_data.index)
    drawdownduration = pd.Series(np.zeros(n),
                                 index=df_data.index)
    for i in range(1, n):
        highwatermark.iloc[i] = max(highwatermark.iloc[i - 1],
                                    float(df_data.iloc[i]))
        drawdown.iloc[i] = (highwatermark.iloc[i] -
                            float(df_data.iloc[i])) / highwatermark.iloc[i]
        if drawdown.iloc[i] == 0:
            drawdownduration.iloc[i] = 0
        else:
            drawdownduration.iloc[i] = drawdownduration.iloc[i - 1] + 1

    MaxDrawDown_end = drawdownduration.idxmax()
    MaxDrawDown_start = df_data.index[df_data.index.get_loc(MaxDrawDown_end) -
                                      int(max(drawdownduration))]
    return max(drawdown)

if __name__ == '__main__':
    print(os.getcwd())
    file_name = sys.argv[1]
    file = pd.read_excel('./NAV_files/'+str(file_name)+'.xlsx')
    file.columns = ['Date', 'NAV']
    file.set_index('Date', inplace=True)
    dt_start = datetime.datetime(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    dt_end = datetime.datetime(int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    df_data = file.loc[dt_start:dt_end, 'NAV']
    print('-'*20)
    # print(dt_start, dt_end)
    df_returns = df_data / df_data.shift(1) - 1

    num = input('Please enter the number of simulation:')

    dta = df_data.apply(np.log).diff(1).values.squeeze()

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df_data, lags=20, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df_data, lags=20, ax=ax2)

