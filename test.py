# import numpy as np
# import pandas as pd
#
# from Analyzers import Analyzer
# from Analyzers import Draw_Down
# from Analyzers import *
#
# filepath = "D:/Neal/Quant/PythonProject/ValuesFile/values1.csv"
#
# # a = Sharpe_Ratio(filepath, basis="Monthly")
# a = Sortino_Ratio(filepath)
# try:
#     print("MaxDuration: ", a.Maximum_Drawdown())
# except:
#     print("No drawdown")
#     pass
#
# print("Avg. Return: ", a.AvrgRet)
# print(a.describe)
# # print("Sharpe Ratio: ", a.SharpeRatio())
# print("Sortino Ratio: ", a.SortinoRatio())

# from Market_Analysis import Futures_Market
#
# #tmp = Futures_Market.SNR("ru")
# #
# #print(tmp.get_Asset_price())
# #print(tmp.rolling_SNR())
#
#
# from Market_Analysis import Futures_Market
# tmp = Futures_Market.AMH.MDI(markets=["ru","l"])
# print(tmp.Market_Divergence_Index())
# print(tmp.markets)
# print(tmp.get_df_Markets_SNRs())

# from Market_Analysis.Market_Analysis_Tools.Regression_Analysis import Regrs_Analysis

# x = R

import numpy as np
import pandas as pd
import datetime as dt
import mpl_finance as mpf
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt


from Data import MongoDB
stocklist = ['600614', '601118']
DB = MongoDB.MongoDBData()
dbnames = ['Stocks_Data', 'Futures_Data']
for stock in stocklist:
    DB.data2MongoDB('Stocks_Data', stock)

df = DB.datafromMongoDB('Stocks_Data', '600614')
df = df.loc[:, ['open', 'high', 'low', 'close']]

Data_list = []
for date, row in df.iterrows():
    Date = date2num(dt.datetime.strptime(date, "%Y-%m-%d"))
    Open, High, Low, Close = row[:4]
    Data = (Date, Open, High, Low, Close)
    Data_list.append(Data)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
# 设置X轴刻度为日期时间
ax.xaxis_date()
plt.xticks(rotation=45)
plt.yticks()
plt.title("股票代码：601558两年K线图")
plt.xlabel("时间")
plt.ylabel("股价（元）")
mpf.candlestick_ohlc(ax,Data_list,width=1.5,colorup='r',colordown='green')
plt.grid()