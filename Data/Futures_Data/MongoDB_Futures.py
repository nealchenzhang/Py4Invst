import pymongo
import numpy as np
import pandas as pd
import datetime
import pprint
import json


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


def df_fromMongoDB(dbname, coll_name, dbhost='localhost', dbport=27017,
                   dbusername=None, dbpassword=None):
    conn = MongoDBData(dbhost, dbport, dbusername, dbpassword)._connect_mongo()
    tmp = list(conn[dbname][coll_name].find())
    df = pd.DataFrame.from_dict(tmp).drop('_id', axis=1)
    df['datetime'] = df['datetime'].apply(pd.to_datetime)
    df.set_index('datetime', inplace=True)
    df = df.sort_index()
    return df

class tickData(object):
    """
    input:
        tick_DB: DB名称,对应MongoDB数据库相应数据库名
        ticker: 合约名称,对应MongoDB数据库名中的CollectionName
        start: 遵从数据库时间标准,国际标准时间
        end: 遵从数据库时间标准,国际标准时间
    """
    timedelta_1s = pd.Timedelta(seconds=1)
    timedelta_1m = pd.Timedelta(minutes=1)
    timedelta_15m = pd.Timedelta(minutes=15)
    timedelta_utc = pd.Timedelta(hours=8)

    def __init__(self, tick_DB, ticker):
        self.tick_DB = tick_DB
        self.ticker = ticker

    def tick2df(self, start, end):
        conn = MongoDBData(dbhost='localhost', dbport=27017)._connect_mongo()
        dbname = self.tick_DB
        coll_name = self.ticker

        coll = conn[dbname][coll_name]
        cursor = coll.find({'datetime': {'$gte': start, '$lte': end}})
        df = pd.DataFrame(list(cursor))
        # df['datetime'] = df['datetime'].apply(pd.to_datetime)
        df.set_index('datetime', inplace=True)
        df.drop(['exchange', 'time', '_id'], axis=1, inplace=True)
        # print('{}从{}到{}的tick数据提取成功'.format(coll_name, start, end))
        df = df.sort_index()
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
        # print(night_trade_time_index)
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

    def df_1min_2MongoDB(self, df, dbname, coll_name):
        conn = MongoDBData(dbhost='localhost', dbport=27017)._connect_mongo()
        df = df.reset_index().rename(columns={'index': 'datetime'})
        df['datetime'] = df['datetime'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        records = json.loads(df.T.to_json()).values()
        conn[dbname][coll_name].insert(records)
        print('Data for {} Stored!'.format(self.ticker))

    def df_1min_fromMongoDB(self, dbname, coll_name):
        conn = MongoDBData(dbhost='localhost', dbport=27017)._connect_mongo()
        tmp = list(conn[dbname][coll_name].find())
        df = pd.DataFrame.from_dict(tmp).drop('_id', axis=1)
        df['datetime'] = df['datetime'].apply(pd.to_datetime)
        df.set_index('datetime', inplace=True)
        return df

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

    asset = tickData('tick_rb', 'rb1801')
    # 品种不同 night_delta不同
    night_delta = pd.Timedelta(hours=2)
    start = '2017-11-27'
    end = '2017-12-15'
    research_daterange = pd.bdate_range(start, end)
    for i in research_daterange:
        if i.isoweekday() == 1:
            start_night_trade = datetime.datetime(i.year, i.month, i.day, 21, 00, 00, 000) -\
                                pd.Timedelta(days=3) - pd.Timedelta(hours=8)
            end_night_trade = datetime.datetime(i.year, i.month, i.day, 21, 00, 00, 000) - \
                              pd.Timedelta(days=3) + night_delta - pd.Timedelta(hours=8)
            start_day_trade = datetime.datetime(i.year, i.month, i.day, 9, 00, 00, 000) - \
                              pd.Timedelta(hours=8)
            end_day_trade = datetime.datetime(i.year, i.month, i.day, 15, 00, 00, 000) - \
                            pd.Timedelta(hours=8)
        else:
            start_night_trade = datetime.datetime(i.year, i.month, i.day, 21, 00, 00, 000) -\
                                pd.Timedelta(days=1) - pd.Timedelta(hours=8)
            end_night_trade = datetime.datetime(i.year, i.month, i.day, 21, 00, 00, 000) - \
                              pd.Timedelta(days=1) + night_delta - pd.Timedelta(hours=8)
            start_day_trade = datetime.datetime(i.year, i.month, i.day, 9, 00, 00, 000) - \
                              pd.Timedelta(hours=8)
            end_day_trade = datetime.datetime(i.year, i.month, i.day, 15, 00, 00, 000) - \
                            pd.Timedelta(hours=8)

        try:
            df_night = asset.tick2OneMinute_night_session(start_night_trade, end_night_trade)
            print('{}{}到{}夜盘数据已改成1min数据'.format('rb1801', start_night_trade+asset.timedelta_utc,
                                                end_night_trade+asset.timedelta_utc))
            df_day = asset.tick2OneMinute_day_session(start_day_trade, end_day_trade)
            print('{}{}到{}白天数据已改成1min数据'.format('rb1801', start_day_trade + asset.timedelta_utc,
                                                end_day_trade + asset.timedelta_utc))

            df_1m = df_night.append(df_day).sort_index()
            asset.df_1min_2MongoDB(df_1m, '1min_rb', 'rb1801')
        except:
            print(df_night)
            try:
                print(df_day)
            except: pass
            pass
        # print('{}到{}的数据已转成1min并导入{}中'.format(start, end, 'rb1801'))

        # df = asset.df_15min(df_1m)
