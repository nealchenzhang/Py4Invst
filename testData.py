import pandas as pd
import numpy as np
import datetime as dt
import mpl_finance as mpf
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt

# df = df.loc[:, ['Open', 'High', 'Low', 'Close']]
# Data_list = []
# for date, row in df.iterrows():
#     Date = date2num(date)
#     Open, High, Low, Close = row[:4]
#     Data = (Date, Open, High, Low, Close)
#     Data_list.append(Data)
#
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.2)
# # 设置X轴刻度为日期时间
# ax.xaxis_date()
# plt.xticks(rotation=45)
# plt.yticks()
# # mpf.candlestick_ohlc(ax,Data_list,width=0.00025,colorup='r',colordown='green')
# mpf.candlestick_ohlc(ax,Data_list,width=0.0025,colorup='r',colordown='green')
# plt.grid()

from Data.Futures_Data.MongoDB_manipulation import *
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# 研究时间从2017年11月28日（27日夜盘）开始至2017年12月15日收盘
asset = tickData('tick_rb', 'rb1801')
df_1m = asset.df_1min_fromMongoDB('1min_rb', 'rb1801')
start = dt.datetime(2017, 11, 27, 21, 00,00,000)
df_research_1min = df_1m.loc[start:].sort_index()
df_research_15min = asset.df_15min(df_research_1min)
# 均线系统策略
# 1min均线处理
fig, ax1 = plt.subplots()
df_research_1min['Close'].plot(ax=ax1)
MA_1min_20 = df_research_1min['Close'].rolling(window=20).mean()
MA_1min_20.plot(ax=ax1)
# 15min均线