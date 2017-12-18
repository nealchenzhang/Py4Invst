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
    """
    input:
        asset: 资产名称,对应MongoDB数据库相应数据库名
        ticker: 合约名称,对应MongoDB数据库名中的CollectionName
        start: 遵从数据库时间标准,国际标准时间
        end: 遵从数据库时间标准,国际标准时间
    """
    timedelta_1s = pd.Timedelta(seconds=1)
    timedelta_1m = pd.Timedelta(minutes=1)
    timedelta_15m = pd.Timedelta(minutes=15)
    timedelta_utc = pd.Timedelta(hours=8)

    def __init__(self, asset, ticker):
        self.asset = asset
        self.ticker = ticker

    def tick2df(self, start, end):
        conn = MongoDBData(dbhost='localhost', dbport=27017)._connect_mongo()
        dbname = self.asset
        coll_name = self.ticker

        coll = conn[dbname][coll_name]
        # print('{}数据库链接成功'.format(dbname))
        cursor = coll.find({'datetime': {'$gte': start, '$lte': end}})
        df = pd.DataFrame(list(cursor))
        # return df
        df.set_index('datetime', inplace=True)
        df.drop(['exchange', 'time', '_id'], axis=1, inplace=True)
        # print('{}从{}到{}的tick数据提取成功'.format(coll_name, start, end))
        return df

    def tick2OneMinute_day_session(self, start, end):
        # 白天交易时间段
        day1_trade_time_index = pd.date_range(start=start+self.timedelta_utc,
                                              end=start+self.timedelta_utc+pd.Timedelta(hours=1, minutes=14),
                                              freq='1min')
        day2_trade_time_index = pd.date_range(start=start+self.timedelta_utc+pd.Timedelta(hours=1, minutes=30),
                                              end=start+self.timedelta_utc+pd.Timedelta(hours=2, minutes=29),
                                              freq='1min')
        day3_trade_time_index = pd.date_range(start=start+self.timedelta_utc+pd.Timedelta(hours=4, minutes=30),
                                              end=start+self.timedelta_utc+pd.Timedelta(hours=5, minutes=59),
                                              freq='1min')
        day_trade_time_index = day1_trade_time_index.append(day2_trade_time_index).append(day3_trade_time_index)
        df_1m_day = pd.DataFrame(index=day_trade_time_index)
        df = self.tick2df(start, end)
        for i in day_trade_time_index:
            tmp = df.loc[i-self.timedelta_utc: i-self.timedelta_utc+pd.Timedelta(minutes=1)-pd.Timedelta(milliseconds=1),
                  ['lastPrice', 'openInterest', 'volume']]
            df_1m_day.loc[i, 'Open'] = tmp.iloc[0]['lastPrice']
            df_1m_day.loc[i, 'Close'] = tmp.iloc[-1]['lastPrice']
            df_1m_day.loc[i, 'High'] = tmp['lastPrice'].max()
            df_1m_day.loc[i, 'Low'] = tmp['lastPrice'].min()
            df_1m_day.loc[i, 'Volume'] = tmp.iloc[-1]['volume']
            df_1m_day.loc[i, 'OI'] = tmp.iloc[-1]['openInterest']
        return df_1m_day

    def tick2OneMinute_night_session(self, start, end):
        night_trade_time_index = pd.date_range(start=start+self.timedelta_utc,
                                               end=end+self.timedelta_utc-pd.Timedelta(minutes=1),
                                               freq='1min')
        df_1m_night = pd.DataFrame(index=night_trade_time_index)
        df = self.tick2df(start, end)
        for i in night_trade_time_index:
            tmp = df.loc[i-self.timedelta_utc: i-self.timedelta_utc+pd.Timedelta(minutes=1)-pd.Timedelta(milliseconds=1),
                  ['lastPrice', 'openInterest', 'volume']]
            df_1m_night.loc[i, 'Open'] = tmp.iloc[0]['lastPrice']
            df_1m_night.loc[i, 'Close'] = tmp.iloc[-1]['lastPrice']
            df_1m_night.loc[i, 'High'] = tmp['lastPrice'].max()
            df_1m_night.loc[i, 'Low'] = tmp['lastPrice'].min()
            df_1m_night.loc[i, 'Volume'] = tmp.iloc[-1]['volume']
            df_1m_night.loc[i, 'OI'] = tmp.iloc[-1]['openInterest']
        return df_1m_night

    def df_1min_2MongoDB(self, tic):
        conn = MongoDBData(dbhost='localhost', dbport=27017)._connect_mongo()
        dbname = self.asset
        coll_name = self.ticker

        coll = conn[dbname][coll_name]

    def df_15min(self, df):
        """
        :param df: 1min DataFrame for a specific period
        :return df_15m_day: 15min candle bar DataFrame for one day
        """
        ohlcvoi_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'last',
            'OI': 'last'
        }
        df_15m = (df.resample('15T', closed='left', label='left').apply(ohlcvoi_dict)).dropna()
        return df_15m

if __name__ == '__main__':

    start = datetime.datetime(2017, 11, 13, 9, 00, 00, 000) - pd.Timedelta(hours=8)
    end = datetime.datetime(2017, 11, 14, 15, 00, 00, 000)- pd.Timedelta(hours=8)
    start_night = datetime.datetime(2017, 11, 10, 21, 00, 00, 000) - pd.Timedelta(hours=8)
    end_night = datetime.datetime(2017, 11, 10, 23, 30, 00, 000) - pd.Timedelta(hours=8)

    # coll.create_index([('datetime', pymongo.ASCENDING)])

    iron = tickData('tick_i', 'i1801')
    iron = tickData('test', 'i1801')

    df_i1801 = iron.tick2df(start, end)
    df_i1801_1m = iron.tick2OneMinute_day_session(start, end)

    # df_i1801_night = iron.tick2df(start_night, end_night)
    # df_i1801_1m = iron.tick2OneMinute_night_session(start_night, end_night)

    df_i1801 = df_i1801_1m.append(iron.tick2OneMinute_night_session(start_night, end_night)).sort_index()









# assets_list = ['i', 'au']