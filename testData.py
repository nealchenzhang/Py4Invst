import pymongo
import pandas as pd

connection = pymongo.MongoClient('localhost', 27017)
db = connection.CTP_tick

# zn01 = db['zn1708']
zn02 = db['zn1709']
zn03 = db['zn1710']
zn04 = db['zn1711']
#
# for i in collection.find_one({'date':'2017-08-10'}):
#     pprint.pprint(i)

df_zn = pd.DataFrame()

time = pd.date_range('2017-08-10 13:30:00.000', end='2017-08-10 15:00:00.000', freq='500ms')
dtstart = '13:30:00.0'
dtend = '13:40:00.0'

cursor_zn02 = zn02.find({'time':{'$gte':dtstart, '$lte':dtend}})
cursor_zn03 = zn03.find({'time':{'$gte':dtstart, '$lte':dtend}})
cursor_zn04 = zn04.find({'time':{'$gte':dtstart, '$lte':dtend}})


df_zn02 = pd.DataFrame(list(cursor_zn02))
df_zn03 = pd.DataFrame(list(cursor_zn03))
df_zn04 = pd.DataFrame(list(cursor_zn04))
df_zn =pd.DataFrame()

df_zn['1_ask'] = df_zn02['askPrice1']
df_zn['1_bid'] = df_zn02['bidPrice1']
df_zn['2_ask'] = df_zn03['askPrice1']
df_zn['2_bid'] = df_zn03['bidPrice1']
df_zn['3_ask'] = df_zn04['askPrice1']
df_zn['3_bid'] = df_zn04['bidPrice1']

df_zn['spread_long'] = 2 * df_zn['2_ask'] - 1 * df_zn['1_bid'] - 1* df_zn['3_bid']
df_zn['spread_short'] = - 2 * df_zn['2_bid'] + 1 * df_zn['1_ask'] + 1* df_zn['3_ask']
