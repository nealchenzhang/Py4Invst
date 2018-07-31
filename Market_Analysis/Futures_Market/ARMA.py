import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
plt.style.use('ggplot')

import arch
from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import datetime as dt

#
# dff_df_R001 = df.diff()
# adf = ADF(df_R001.dropna())
# print(adf.summary().as_text())
# adf.lags=1
# plot_acf(df_R001, lags=25, alpha=0.5)#自相关系数ACF图
# plot_pacf(df_R001, lags=25, alpha=0.5)#偏相关系数PACF图
#
# adf.lags = 4
#
# reg_res = adf.regression
# print(reg_res.summary().as_text())
#
# type(reg_res)


class TSAnalysis(object):

    def plot_trend(self, df_ts, size):
        ax = plt.subplot()

        # 对size个数据进行移动平均
        rol_mean = df_ts.rolling(window=size).mean()
        # 对size个数据进行加权移动平均
        rol_weighted_mean = df_ts.ewm(span=size).mean()

        df_ts.plot(color='blue', label='Original', ax=ax)
        rol_mean.plot(color='red', label='Rolling Mean', ax=ax)
        rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean', ax=ax)
        plt.legend(loc='best')
        plt.title('Rolling Mean')
        plt.show()

    def plot_ts(self, df_ts):
        ax = plt.subplot()
        df_ts.plot(color='blue', ax=ax)
        plt.show()

    def ADF_test(self, df_ts, lags=None):
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

    def plot_acf_pacf(self, df_ts, lags=31):
        f = plt.figure(facecolor='white', figsize=(12, 8))
        ax1 = f.add_subplot(211)
        plot_acf(df_ts, lags=31, ax=ax1)
        ax2 = f.add_subplot(212)
        plot_pacf(df_ts, lags=31, ax=ax2)
        plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    RU = pd.read_excel('/home/nealzc1991/PycharmProjects/Py4Invst/Fundamental/RU.xls')
    RU.dropna(inplace=True)

    df = pd.DataFrame(columns=['Date', 'Close'])
    df.loc[:, 'Date'] = RU.loc[:, 'Date']
    df.loc[:, 'Close'] = RU.loc[:, 'Close']

    df.set_index('Date', inplace=True)

    df_RU = df.loc[dt.datetime(2015, 7, 31):dt.datetime(2016, 7, 29), :]

    a = TSAnalysis()
    adf = a.ADF_test(df.apply(np.log).diff(1).dropna())

    a.plot_acf_pacf(df.apply(np.log).diff(1).dropna())

    df_diff_1 = df.apply(np.log).diff(1).dropna()

    from statsmodels.tsa.arima_model import ARMA
    model = ARMA(df_diff_1, order=(1, 1))
    result_arma = model.fit(disp=-1, method='css')
    predict_ts = result_arma.predict()
    diff_shift_ts = df_diff_1.shift(1).dropna()
    diff_recover_1 = predict_ts + diff_shift_ts.loc[:,'Close']