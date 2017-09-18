import pandas as pd
import tushare as ts
import pymongo
import json

dbhost = 'localhost'
dbport = 27017

conn = pymongo.MongoClient(host=dbhost, port=dbport)

df = ts.get_hist_data('600848')

conn['600848'].histdata.insert(json.loads(df.to_json(orient='records')))

collection = conn['600848'].histdata

data = pd.DataFrame(list(collection.find()))
