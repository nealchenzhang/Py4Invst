import pymongo
import numpy as np
import pandas as pd
import datetime
import pprint

number = '0123456789'

connection = pymongo.MongoClient('localhost', 27017)
# print all the collections in the DB
collections = connection['Futures_Data'].collection_names()


futures_asset = [i.rstrip(number) for i in collections]
futures_asset = list(set(futures_asset))
contracts = []
for i in collections:
    # print(i.rstrip(number))
    if i.rstrip(number) == futures_asset[0]:
        contracts.append(i)

jiao = 'j1709'

test_Coll = connection['Futures_Data'][jiao]

for index in test_Coll.index_information():
    pprint.pprint(index)

if __name__ == '__main__':
    connection = pymongo.MongoClient('localhost', 27017)
    coll = connection['test']['i1801']


    timedelta_1s = pd.Timedelta(seconds=1)
    timedelta_1m = pd.Timedelta(minutes=1)
    timedelta_15m = pd.Timedelta(minutes=15)

    start = datetime.datetime(2017, 11, 13, 9, 00, 00, 000) - pd.Timedelta(hours=8)
    end = datetime.datetime(2017, 11, 13, 15, 00, 00, 000)- pd.Timedelta(hours=8)
    start_night = datetime.datetime(2017, 11, 10, 21, 00, 00, 000)
    end_night = datetime.datetime(2017, 11, 10, 23, 30, 00, 000)

    coll.create_index([('datetime', pymongo.ASCENDING)])

    cursor_i1801 = coll.find({'datetime': {'$gte': start, '$lte': end}})
    df_i1801 = pd.DataFrame(list(cursor_i1801))
    df_i1801.set_index('datetime', inplace=True)
    df_i1801.drop(['exchange', 'time', '_id'], axis=1, inplace=True)

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