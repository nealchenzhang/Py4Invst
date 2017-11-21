# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Tue Sept 26 15:00:19 2017

# @author: NealChenZhang

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################

import pandas as pd
import tushare as ts
import pymongo
import json


class TushareData(object):

    def __init__(self, tick, *args, **kwargs):
        super(TushareData, self).__init__(*args, **kwargs)

        self.tick = tick

    def get_tushare_data(self):
        df = ts.get_hist_data(self.tick)
        return df


class MongoDBData(object):

    def __init__(self, dbhost='localhost',
                 dbport=27017, dbusername=None,
                 dbpassword=None, *args, **kwargs):
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

    def data2MongoDB(self, dbname, tick):
        # Get Tushare Data for tick
        TD = TushareData(tick)
        df = TD.get_tushare_data()

        conn = self._connect_mongo()
        conn[dbname][tick].insert(json.loads(df.to_json(orient='index')))
        print('Data for {} Stored!'.format(tick))

    def datafromMongoDB(self, dbname, tick):
        conn = self._connect_mongo()
        collection = conn[dbname][tick]
        tmp = list(collection.find())
        data = pd.DataFrame.from_dict(tmp[0]).T.drop('_id')
        print('Data for {} Retrieved!'.format(tick))
        return data
