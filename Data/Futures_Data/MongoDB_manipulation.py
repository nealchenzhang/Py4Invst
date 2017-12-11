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
# .find({'date':"2017-08-11"})
dtstart = '13:30:00.0'
dtend = '13:40:00.0'

for index in test_Coll.index_information():
    pprint.pprint(index)

if __name__ == '__main__':
    connection = pymongo.MongoClient('localhost', 27017)
    coll = connection['test']['j1801']


    timedelta_1s = pd.Timedelta(seconds=1)
    timedelta_1m = pd.Timedelta(minutes=1)
    timedelta_15m = pd.Timedelta(minutes=15)
    start = datetime.datetime(2017, 8, 10, 1, 40, 49, 000)

    coll.create_index([('datetime', pymongo.ASCENDING)])

    for i in coll.find({'datetime': {'$gte': start,'$lte': start+timedelta_1s}}):
        pprint.pprint(i)

    query = [
        {
            '$limit':100
        },
        {
            '$project': {
                'High': { $max: "lastPrice"},
                'Low': { $min: "lastPrice"},
                'Open': {'datetime': {'$gte': start, '$lte': start + timedelta_1s}}
            }
        }
    ]
        'datetime': {'$gte': start, '$lte': start + timedelta_1m}
    }

    projection = {'lastPrice':}





for i in coll.find({'date':'2017-09-25'}):
    pprint.pprint(i)
    break
