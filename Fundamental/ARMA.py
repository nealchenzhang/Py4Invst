import numpy as np
import pandas as pd
import arch

import os
import datetime as dt

from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

R001 = pd.read_excel('R001.xlsx')
R001.dropna(inplace=True)

df = pd.DataFrame(columns=['Date', 'Close'])
df.loc[:, 'Date'] = R001.loc[:, '日期']
df.loc[:, 'Close'] = R001.loc[:, '收盘价']

df.set_index('Date', inplace=True)

df_R001 = df.loc[dt.datetime(2015,7,31):dt.datetime(2016,7,29),:]

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

if __name__ == '__main__':
    print(os.getcwd())

