import pandas as pd
import numpy as np
import datetime
import mpl_finance as mpf
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt
from Data.Stocks_Data import MongoDB

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC

import tushare as ts


# 选取的组合股票池
stocklist = ['hs300']
DB = MongoDB.MongoDBData()
# # Tushare数据源导入数据库
# for stock in stocklist:
#     DB.data2MongoDB('Stocks_Data', stock)

df_if300 = DB.datafromMongoDB('Stocks_Data', 'hs300')


def create_lagged_series(symbol, start_date, end_date, lags=5):
    """
    This creates a Pandas DataFrame that stores the
    percentage returns of the adjusted closing value of a
    stock obtained from Tushare, along with a number of lagged returns
    from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day,
    are also included.
    :param symbol:
    :param start_date:
    :param end_date:
    :param lags:
    :return:
    """
    # Obtain stock information from Tushare
    df_ts = ts.get_hist_data(symbol, start_date-pd.Timedelta(days=365),
                             end_date)

    df_ts = df_ts.reset_index()
    df_ts = df_ts.rename(columns={'index':'datetime'})
    df_ts['datetime'] = df_ts['datetime'].apply(pd.to_datetime)
    df_ts = df_ts.set_index('datetime')

    # Create the new lagged DataFrame
    df_tslag = pd.DataFrame(index=df_ts.index)
    df_tslag['Today'] = df_ts['close']
    df_tslag['Volume'] = df_ts['volume']

    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        df_tslag["Lag%s" % str(i+1)] = df_ts['Close'].shift(i+1)

    # Create the returns DataFrame
    df_tsret = pd.DataFrame(index=df_tslag.index)
    df_tsret['Volume'] = df_tslag['Volume']
    df_tsret['Today'] = df_tsret['Today'].pct_change()*100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with Quadratic Discriminant analysis model)

    for i, x in enumerate(df_tsret['Today']):
        if (abs(x) < 0.0001):
            df_tsret['Today'][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0, lags):
        df_tsret["Lag%s" % str(i+1)] = \
            df_tslag["Lag%s" % str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    df_tsret['Direction'] = np.sign(df_tsret['Today'])
    df_tsret = df_tsret[df_tsret.index >= start_date]

    return df_tsret