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
# stocklist = ['hs300']
# DB = MongoDB.MongoDBData()
# Tushare数据源导入数据库
# for stock in stocklist:
#     DB.data2MongoDB('Stocks_Data', stock)

# df_if300 = DB.datafromMongoDB('Stocks_Data', 'hs300')


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
    start_date = pd.datetime(int(start_date.split('-')[0]),
                             int(start_date.split('-')[1]),
                             int(start_date.split('-')[2]))
    date = start_date - pd.Timedelta(days=36)
    date = date.strftime("%Y-%m-%d")
    df_ts = ts.get_hist_data(symbol, date, end_date)

    df_ts = df_ts.reset_index()
    df_ts = df_ts.rename(columns={'date': 'datetime'})
    df_ts['datetime'] = df_ts['datetime'].apply(pd.to_datetime)
    # print(df_ts.head(2))
    df_ts = df_ts.set_index('datetime')

    # Create the new lagged DataFrame
    df_tslag = pd.DataFrame(index=df_ts.index)
    df_tslag['Today'] = df_ts['close']
    df_tslag['Volume'] = df_ts['volume']

    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        df_tslag["Lag%s" % str(i+1)] = df_ts['close'].shift(i+1)

    # Create the returns DataFrame
    df_tsret = pd.DataFrame(index=df_tslag.index)
    df_tsret['Volume'] = df_tslag['Volume']
    df_tsret['Today'] = df_tslag['Today'].pct_change()*100.0

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

if __name__ == "__main__":
    if300ret = create_lagged_series('hs300', "2015-01-05", "2017-12-31", lags=5)

    # Use the prior two days of returns as predictor
    # values, with direction as the response
    X = if300ret[["Lag1", "Lag2"]]
    y = if300ret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2016,1,1)

    # Create training and test sets
    X_train = X[(X.index >= pd.datetime(2015,1,5)) & (X.index < start_test)]
    y_train = y[(y.index >= pd.datetime(2015,1,5)) & (y.index < start_test)]
    X_test = X[(X.index >= start_test) & (X.index <= pd.datetime(2017, 12, 26))]
    y_test = y[(y.index >= start_test) & (y.index <= pd.datetime(2017, 12, 26))]

    # Create the parametrized models
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LogisticReg", LogisticRegression()),
              ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),
              ("QuadraticDiscriminantAnalysis", QuadraticDiscriminantAnalysis()),
              ("LinearSVC", LinearSVC()),
              ("RadialSupportVectorMachine", SVC(
                  C=1000000.0, cache_size=200, class_weight=None,
                  coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                  max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)
               ),
              ("RandomForest", RandomForestClassifier(
                  n_estimators=1000, criterion='gini',
                  max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, max_features='auto',
                  bootstrap=True, oob_score=False, n_jobs=1,
                  random_state=None, verbose=False
              ))]

    # Iterate through the models
    for m in models:

        # Train each of the models on the training set
        m[1].fit(X_train, y_train)

        # Make an array of predictions on the test set
        pred = m[1].predict(X_test)

        # Output the hit-rate and the confusion matrix for each model
        print("%s:\n%0.4f" % (m[0], m[1].score(X_test, y_test)))
        print("%s\n" % confusion_matrix(y_test, pred))
