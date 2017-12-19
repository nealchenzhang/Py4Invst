import pandas as pd
import numpy as np
import datetime as dt
import mpl_finance as mpf
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt
# df = df_rb1801_1m
df = df.loc[:, ['Open', 'High', 'Low', 'Close']]
Data_list = []
for date, row in df.iterrows():
    Date = date2num(date)
    Open, High, Low, Close = row[:4]
    Data = (Date, Open, High, Low, Close)
    Data_list.append(Data)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
# 设置X轴刻度为日期时间
ax.xaxis_date()
plt.xticks(rotation=45)
plt.yticks()
# plt.title("股票代码：002298两年K线图")
# plt.xlabel("时间")
# plt.ylabel("股价（元）")
# mpf.candlestick_ohlc(ax,Data_list,width=0.00025,colorup='r',colordown='green')
mpf.candlestick_ohlc(ax,Data_list,width=0.0025,colorup='r',colordown='green')
plt.grid()