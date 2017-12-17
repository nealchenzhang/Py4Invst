import pymongo
import numpy as np
import pandas as pd
import datetime
import pprint

class MongoDBData(object):

    def __init__(self, dbhost='localhost',
                 dbport=27017, dbusername=None,
                 dbpassword=None, *args, **kwargs):
        super(MongoDBData, self).__init__(*args, **kwargs)

        self.host = dbhost
        self.port = dbport
        self.username = dbusername
        self.password = dbpassword

    def _connect_mongo(self):
        # Connection to MongoDB
        if self.username and self.password:
            conn = pymongo.MongoClient(host=self.host,
                                       port=self.port,
                                       username=self.username,
                                       password=self.password)
        else:
            conn = pymongo.MongoClient(host=self.host,
                                       port=self.port)
        return conn

class tickData(object):

    timedelta_1s = pd.Timedelta(seconds=1)
    timedelta_1m = pd.Timedelta(minutes=1)
    timedelta_15m = pd.Timedelta(minutes=15)

    def __init__(self, asset, ticker):
        self.asset = asset
        self.ticker = ticker

    def tick2df(self, start, end):
        conn = MongoDBData(dbhost='localhost', dbport=27017)._connect_mongo()
        dbname = self.asset
        coll_name = self.ticker

        coll = conn[dbname][coll_name]
        print('{}数据库链接成功'.format(dbname))
        cursor = coll.find({'datetime': {'$gte': start, '$lte': end}})
        df = pd.DataFrame(list(cursor))
        df.set_index('datetime', inplace=True)
        df.drop(['exchange', 'time', '_id'], axis=1, inplace=True)
        print('{}从{}到{}的tick数据提取成功'.format(coll_name, start, end))
        return df







if __name__ == '__main__':

    # timedelta_1s = pd.Timedelta(seconds=1)
    # timedelta_1m = pd.Timedelta(minutes=1)
    # timedelta_15m = pd.Timedelta(minutes=15)

    start = datetime.datetime(2017, 11, 13, 9, 00, 00, 000) - pd.Timedelta(hours=8)
    end = datetime.datetime(2017, 11, 14, 15, 00, 00, 000)- pd.Timedelta(hours=8)
    start_night = datetime.datetime(2017, 11, 10, 21, 00, 00, 000)
    end_night = datetime.datetime(2017, 11, 10, 23, 30, 00, 000)

    # coll.create_index([('datetime', pymongo.ASCENDING)])

    i = tickData('tick_i', 'i1801')

    df_i1801 = i.tick2df(start, end)

    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'last'
    }

    day1_trade_time_index = pd.date_range(start=start+pd.Timedelta(hours=8),
                                          end=start+pd.Timedelta(hours=8)+pd.Timedelta(hours=1, minutes=14), freq='1min')
    day2_trade_time_index = pd.date_range(start=start+pd.Timedelta(hours=8)+pd.Timedelta(hours=1, minutes=30),
                                          end=start+pd.Timedelta(hours=8)+pd.Timedelta(hours=2, minutes=29), freq='1min')
    day3_trade_time_index = pd.date_range(start=start+pd.Timedelta(hours=8)+pd.Timedelta(hours=4, minutes=30),
                                          end=start+pd.Timedelta(hours=8)+pd.Timedelta(hours=5, minutes=59), freq='1min')

    day_trade_time_index = day1_trade_time_index.append(day2_trade_time_index).append(day3_trade_time_index)
    df_1m_day = pd.DataFrame(index=day_trade_time_index)
    for i in day_trade_time_index:
        tmp = df_i1801.loc[i-pd.Timedelta(hours=8): i-pd.Timedelta(hours=8) + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),['lastPrice', 'openInterest', 'volume']]
        df_1m_day.loc[i, 'Open'] = tmp.iloc[0]['lastPrice']
        df_1m_day.loc[i, 'Close'] = tmp.iloc[-1]['lastPrice']
        df_1m_day.loc[i, 'High'] = tmp['lastPrice'].max()
        df_1m_day.loc[i, 'Low'] = tmp['lastPrice'].min()
        df_1m_day.loc[i, 'Volume'] = tmp.iloc[-1]['volume']
        df_1m_day.loc[i, 'OI'] = tmp.iloc[-1]['openInterest']

    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'last',
        'OI': 'last'
    }

    df_15m_day = (df_1m_day.resample('15T', closed='left', label='left').apply(ohlc_dict)).dropna()

assets_list = ['i', 'au']