import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
plt.style.use('ggplot')

import arch
from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import os
import datetime as dt


dff_df_R001 = df_R001.diff()
adf = ADF(df_R001.dropna())
print(adf.summary().as_text())
adf.lags=1
plot_acf(df_R001, lags=25, alpha=0.5)#自相关系数ACF图
plot_pacf(df_R001, lags=25, alpha=0.5)#偏相关系数PACF图

adf.lags = 4

reg_res = adf.regression
print(reg_res.summary().as_text())

type(reg_res)


class TSAnalysis(object):

    def plot_trend(self, df_ts, size):
        ax1 = plt.subplot()

        # 对size个数据进行移动平均
        rol_mean = df_ts.rolling(window=size).mean()
        # 对size个数据进行加权移动平均
        rol_weighted_mean = df_ts.ewm(span=size).mean()

        df_ts.plot(color='blue', label='Original', ax=ax1)
        rol_mean.plot(color='red', label='Rolling Mean', ax=ax1)
        rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean', ax=ax1)
        plt.legend(loc='best')
        plt.title('Rolling Mean')
        plt.show()

    def plot_ts(self, df_ts):
        ax1 = plt.subplot()
        df_ts.plot(color='blue', ax=ax1)
        plt.show()

    def ADF_test(self, df_ts):
        adf = ADF(df_ts)
        print(adf.summary().as_text())

    # 自相关和偏相关图，默认阶数为31阶
    def draw_acf_pacf(self, ts, lags=31):
        f = plt.figure(facecolor='white')
        ax1 = f.add_subplot(211)
        plot_acf(ts, lags=31, ax=ax1)
        ax2 = f.add_subplot(212)
        plot_pacf(ts, lags=31, ax=ax2)
        plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    R001 = pd.read_excel('R001.xlsx')
    R001.dropna(inplace=True)

    df = pd.DataFrame(columns=['Date', 'Close'])
    df.loc[:, 'Date'] = R001.loc[:, '日期']
    df.loc[:, 'Close'] = R001.loc[:, '收盘价']

    df.set_index('Date', inplace=True)

    df_R001 = df.loc[dt.datetime(2015, 7, 31):dt.datetime(2016, 7, 29), :]

    a = TSAnalysis()
    a.draw_trend(df, 5)
