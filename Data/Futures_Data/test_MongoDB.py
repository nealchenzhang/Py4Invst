!/usr/bin/env python3
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

# import re

# import tushare as ts

# ts.set_token('d42b6bb78dae5ca4b400e2629071640bc458b0f1f21eeef71823447ef980fb94')


class FuturesData(object):

    def __init__(self, start_date, end_date, asset, basis='Daily'):
        self.start_date = start_date
        self.end_date = end_date
        self.asset = asset
        self.basis = basis

    def get_cc(self):
        """
        Futures contracts:
        :return: the continuous contract for the underlying asset

        Method:
            Define the continuous contract as the trading volume become the
            first in all futures contracts of that underlying asset at
            particular date
        """




# dbhost = "192.168.1.58"
dbhost = 'localhost'
dbport = 27017

client = pymongo.MongoClient(host=dbhost, port=dbport)


db = client.CTPMinuteDb



