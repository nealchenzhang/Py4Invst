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
