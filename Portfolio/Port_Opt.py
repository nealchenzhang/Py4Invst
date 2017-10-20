import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir('/home/nealzhangchen/Documents/Python/Py4Invst/Portfolio')

aapl = pd.read_csv('AAPL_CLOSE',index_col='Date',parse_dates=True)
cisco = pd.read_csv('CISCO_CLOSE',index_col='Date',parse_dates=True)
ibm = pd.read_csv('IBM_CLOSE',index_col='Date',parse_dates=True)
amzn = pd.read_csv('AMZN_CLOSE',index_col='Date',parse_dates=True)

stocks = pd.concat([aapl,cisco,ibm,amzn],axis=1)
stocks.columns = ['aapl','cisco','ibm','amzn']
mean_daily_ret = stocks.pct_change(1).mean()

stock_normed = stocks/stocks.iloc[0]
stock_normed.plot()
stock_daily_ret = stocks.pct_change(1)
stock_daily_ret.head()
log_ret = np.log(stocks/stocks.shift(1))

log_ret.hist(bins=100,figsize=(12,6));
plt.tight_layout()


