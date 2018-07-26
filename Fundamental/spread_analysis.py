__author__ = 'Neal Chen Zhang'
import os
import datetime as dt

import numpy as np
import pandas as pd

from Market_Analysis.Market_Analysis_Tools.Regression_Analysis import Regression_Analysis as RA


class CommoditySpread(object):

    def __init__(self, asset, ls_str_month, ):
        """

        :param asset: using UpperCase for
        :param ls_str_month: Attention! e.g. for January, use 01 instead of 1
        """
        self.asset = asset
        self.ls_month = ls_str_month
        self.ls_int_rate = np.diff([int(i) for i in ls_str_month]).tolist()

    @staticmethod
    def _most_traded(a,b,c):




    def trading_contract(self, df_raw):
        """
        asset +
            _p as close price
            _po as position numbers
            _v as trading volume numbers
        :param df_raw:
        :return:
        """
        asset = self.asset
        columns = [str(asset) + i for i in self.ls_str_month]
        .apply(lambda x: function(x.city, x.year), axis=1)


df_RB = pd.read_csv('/home/nealzc1991/PycharmProjects/Py4Invst/Fundamental/RB.csv')
df_RB.set_index('Date', inplace=True)

if __name__ == '__main__':
    data = {'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou', 'Chongqing'],
            'year': [2016, 2016, 2015, 2017, 2016, 2016],
            'population': [2100, 2300, 1000, 700, 500, 500]}
    frame = pd.DataFrame(data, columns=['year', 'city', 'population', 'debt'])


    def active(a, b):
        if ('ing' in a) and (b == 2016):
            return 1
        else:
            return 0


    print(frame, '\n')
    frame['test'] = frame.apply(lambda x: function(x.city, x.year), axis=1)
    print(frame)
