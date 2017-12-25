import pandas as pd
import numpy as np
import datetime
import mpl_finance as mpf
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt
from Data.Stocks_Data import MongoDB

# 选取的组合股票池
stocklist = ['002298', '002224', '002351']
DB = MongoDB.MongoDBData()
# Tushare数据源导入数据库
# for stock in stocklist:
#     DB.data2MongoDB('Stocks_Data', stock)

# 读取个股数据

df_zdxl = DB.datafromMongoDB('Stocks_Data', '002298')
df_sls = DB.datafromMongoDB('Stocks_Data', '002224')
df_mbz = DB.datafromMongoDB('Stocks_Data', '002351')

df_port_close = pd.DataFrame()
df_port_close['zdxl_close'] = df_zdxl.loc['2017-01-01':]['close']
df_port_close['zdxl_turnover'] = df_zdxl.loc['2017-01-01':]['turnover']

df_port_close['sls_close'] = df_sls.loc['2017-01-01':]['close']
df_port_close['sls_turnover'] = df_sls.loc['2017-01-01':]['turnover']

df_port_close['mbz_close'] = df_mbz.loc['2017-01-01':]['close']
df_port_close['mbz_turnover'] = df_mbz.loc['2017-01-01':]['turnover']

df_port_close = df_port_close.astype(np.float64)
df_port_close = df_port_close.fillna(method='ffill')

# 训练集
df_port_train = df_port_close.loc['2017-01-01':'2017-06-30', ['zdxl_close', 'sls_close', 'mbz_close']]
df_port_normal_train = df_port_train / df_port_train.iloc[0]
df_port_normal_ret_train =  df_port_normal_train / df_port_normal_train.shift(1)
df_port_normal_log_ret_train = df_port_normal_ret_train.apply(np.log)

# 测试集
df_port_test = df_port_close.loc['2017-07-01':, ['zdxl_close', 'sls_close', 'mbz_close']]
df_port_normal_test = df_port_test / df_port_test.iloc[0]
df_port_normal_ret_test =  df_port_normal_test / df_port_normal_test.shift(1)
df_port_normal_log_ret_test = df_port_normal_ret_test.apply(np.log)

# 2017年交易日
trading_dates = pd.to_datetime(pd.Series(df_port_close.index))

# 规则1：总账户39.52w 股价排序第一：5000股 第二：10000股 第三：15000股
n1 = 5000
n2 = 10000
n3 = 15000
init_port_value = 395200

# 参照组合
# 每个月调整两次仓位 分别在月初（1日）和月中（15日）如非交易日递沿
# 股数按照默认股价排序
trading = []

# 月初首个交易日
month = 1
for i in trading_dates:
    if month < 7:
        if i.month == month:
            # print(i)
            trading.append(i)
            month += 1
        pass

# 月中交易日15日前后
trading_mid_month = []
timedelta_tw = pd.Timedelta(days=14)
timedelta_oneday = pd.Timedelta(days=1)
for trading_month_start in trading:
    mid = trading_month_start + timedelta_tw
    if mid in list(trading_dates):
        trading_mid_month.append(mid)
    else:
        while mid.isoweekday() != 6 and mid.isoweekday() != 7 and mid in list(trading_dates):
            mid = mid + timedelta_oneday
        else:
            trading_mid_month.append(mid)

# 参照组交易日期
trading.extend(trading_mid_month)
trading.sort()

# 交易日股数调整
df_trade = pd.DataFrame(index=df_port_train.index, columns=df_port_train.columns)
for i in trading:
    dx = i.strftime('%Y-%m-%d')
    df_trade.loc[dx, (df_port_train.loc[dx].sort_values(ascending=False).index)[0]] = n1
    df_trade.loc[dx, (df_port_train.loc[dx].sort_values(ascending=False).index)[1]] = n2
    df_trade.loc[dx, (df_port_train.loc[dx].sort_values(ascending=False).index)[2]] = n3

df_trade = df_trade.fillna(method='ffill')

df_trade_equity = df_port_train.mul(df_trade, axis='index')
df_trade_equity['PortValue'] = df_trade_equity.sum(axis='columns')

dx1 = trading[0].strftime('%Y-%m-%d')
df_trade_equity.loc[dx1, 'Cash'] = init_port_value - df_trade_equity.loc[dx1, 'PortValue']

df_trade_equity['Cash'] = df_trade_equity['Cash'].fillna(0.0)
# 计算持仓差值
df_trade_diff = df_trade.diff()
for i in trading[1:]:
    dx = i.strftime('%Y-%m-%d')
    il = df_trade_equity.index.get_loc(dx)
    df_trade_equity.loc[dx, 'Cash'] = df_port_train.loc[dx].mul(-df_trade_diff.loc[dx], axis='index').sum() + df_trade_equity.iloc[il-1]['Cash']

df_trade_equity['Cash'] = df_trade_equity['Cash'].cumsum(axis='index', skipna=False)
df_trade_equity['Total'] = df_trade_equity['Cash'] + df_trade_equity['PortValue']


df_trade_equity['Total'].plot(sharex=True)
df_trade_equity['PortValue'].plot(sharex=True)
df_trade_equity['Cash'].plot(kind='bar', sharex=True)
plt.show()


# df = df.loc[:, ['open', 'high', 'low', 'close']]
# Data_list = []
# for date, row in df.iterrows():
#     Date = date2num(dt.datetime.strptime(date, "%Y-%m-%d"))
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
# plt.title("股票代码：002298两年K线图")
# plt.xlabel("时间")
# plt.ylabel("股价（元）")
# mpf.candlestick_ohlc(ax,Data_list,width=1.5,colorup='r',colordown='green')
# plt.grid()