__author__ = 'Neal Chen Zhang'
import os
import datetime as dt

import numpy as np
import pandas as pd

from Market_Analysis.Market_Analysis_Tools.Regression_Analysis import Regression_Analysis as RA


class CommoditySpread(object):

    def __init__(self, str_asset, ls_str_month):
        """

        :param str_asset: using UpperCase for underlying asset
        :param ls_str_month: Attention! e.g. for January, use 01 instead of 1
        """
        self.str_asset = str_asset
        self.ls_str_month = ls_str_month
        self.ls_int_rate = np.diff([int(i) for i in ls_str_month]).tolist()
        self.df_cc0 = None
        self.df_cc1 = None

    @staticmethod
    def _second_mostly_traded(a,b,c):
        ls = [a,b,c]
        ls.remove(np.max(ls))
        return np.max(ls)

    def _mostly_traded_contract(self, df_raw):
        """
        asset +
            _p as close price
            _po as position numbers
            _v as trading volume numbers
        :param df_raw:
        :return: dictionary of date of mostly traded contract
        """
        asset = self.str_asset
        columns = [str(asset) + i for i in self.ls_str_month]
        po_cols = [i+'_po' for i in columns]
        n = len(po_cols)
        if n == 3:
            df_raw['most_traded_po'] = df_raw.apply(
                lambda x: max(x[po_cols[0]], x[po_cols[1]], x[po_cols[2]]), axis=1
            )
        elif n == 2:
            df_raw['most_traded_po'] = df_raw.apply(
                lambda x: max(x[po_cols[0]], x[po_cols[1]]), axis=1
            )
        dic = {}
        for i in po_cols:
            tmp = df_test.loc[:, i] - df_test.loc[:, 'most_traded_po']
            tmp.dropna(inplace=True)
            ls_index = tmp.where(tmp == 0).dropna().index.tolist()
            dic[i.split('_')[0]] = ls_index
        return dic

    def _second_traded_contract(self, df_raw):
        """
        asset +
            _p as close price
            _po as position numbers
            _v as trading volume numbers
        :param df_raw:
        :return: dictionary of date of second mostly traded contract
        """
        asset = self.str_asset
        columns = [str(asset) + i for i in self.ls_str_month]
        po_cols = [i+'_po' for i in columns]
        n = len(po_cols)
        if n == 3:
            df_raw['second_traded_po'] = df_raw.apply(
                lambda x: self._second_mostly_traded(
                    x[po_cols[0]], x[po_cols[1]], x[po_cols[2]]
                ), axis=1
            )
        elif n == 2:
            df_raw['second_traded_po'] = df_raw.apply(
                lambda x: self._second_mostly_traded(
                    x[po_cols[0]], x[po_cols[1]]
                ), axis=1
            )
        dic = {}
        for i in po_cols:
            tmp = df_test.loc[:, i] - df_test.loc[:, 'second_traded_po']
            tmp.dropna(inplace=True)
            ls_index = tmp.where(tmp == 0).dropna().index.tolist()
            dic[i.split('_')[0]] = ls_index
        return dic

    def construct_active_contracts(self):


    # def plot_futures_curve(self):


if __name__ == '__main__':
    df_RB = pd.read_csv('/home/nealzc1991/PycharmProjects/Py4Invst/Fundamental/RB.csv')
    df_RB = pd.read_csv('/home/nealzc/PycharmProjects/Py4Invst/Fundamental/RB.csv')
    df_RB.set_index('Date', inplace=True)

    test = CommoditySpread('RB', ['01', '05', '10'])
    print(test.ls_int_rate)

    asset = test.str_asset
    columns = [str(asset) + i for i in test.ls_month]
    po_cols = [i + '_po' for i in columns]
    df_RB['most_traded_po'] =
    df_RB.apply(lambda x: max(x[po_cols[0]], x[po_cols[1]],  x[po_cols[2]]), axis=1)

    df_test = df_RB.loc['2010-01-04':'2011-01-01',:].copy()
    df_test['most_traded_po'] = df_test.apply(lambda x: max(x[po_cols[0]], x[po_cols[1]], x[po_cols[2]]), axis=1)
    for i in po_cols:
        tmp = df_test.loc[:, i] - df_test.loc[:, 'most_traded_po']
        tmp.dropna(inplace=True)
        tmp.where(tmp==0).dropna().index
        print(i,)


    test.trading_contract(df_raw=df_RB)
