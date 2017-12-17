# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Mon Apr 10 13:59:19 2017

# @author: NealChenZhang

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################
import datetime as dt
import pymongo
import numpy as np
import pandas as pd

class FuturesData(object):

    def __init__(self, start_date, end_date, asset, basis='Daily'):
        self.start_date = start_date
        self.end_date = end_date
        self.asset = asset
        self.basis = basis

    def get_cc(self):
        """
        Futures contracts:
        the continuous contract for the underlying asset

        Method:
            Define the continuous contract as the trading volume become the
            first in all futures contracts of that underlying asset at
            particular date
        """

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

# dbhost = "192.168.1.58"
dbhost = 'localhost'
dbport = 27017

client = pymongo.MongoClient(host=dbhost, port=dbport)



