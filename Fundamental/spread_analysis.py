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
        cols = [str(asset) + i for i in self.ls_str_month]
        po_cols = [i+'_po' for i in cols]
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
            tmp = df_raw.loc[:, i] - df_raw.loc[:, 'most_traded_po']
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
        cols = [str(asset) + i for i in self.ls_str_month]
        po_cols = [i+'_po' for i in cols]
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
            tmp = df_raw.loc[:, i] - df_raw.loc[:, 'second_traded_po']
            tmp.dropna(inplace=True)
            ls_index = tmp.where(tmp == 0).dropna().index.tolist()
            dic[i.split('_')[0]] = ls_index
        return dic

    def active_contract(self, df_raw):
        """

        :param df_raw: raw data of
        :return:
        """
        asset = self.str_asset
        df_cc0 = pd.DataFrame(columns=[asset+'cc0_p', asset+'cc0_v', asset+'cc0_po'])
        dic_active_contract = self._mostly_traded_contract(df_raw=df_raw)
        tmp = pd.DataFrame(columns=[asset+'cc0_p', asset+'cc0_v', asset+'cc0_po'])
        for i in dic_active_contract.keys():
            tmp.loc[:, asset+'cc0_p'] = df_raw.loc[dic_active_contract[i], i+'_p']
            tmp.loc[:, asset+'cc0_v'] = df_raw.loc[dic_active_contract[i], i+'_v']
            tmp.loc[:, asset+'cc0_po'] = df_raw.loc[dic_active_contract[i], i+'_po']
            df_cc0 = df_cc0.append(tmp)
            tmp = pd.DataFrame(columns=[asset + 'cc0_p', asset + 'cc0_v', asset + 'cc0_po'])
        df_cc0 = df_cc0.sort_index(ascending=True)
        self.df_cc0 = df_cc0
        return df_cc0

    def second_active_contract(self, df_raw):
        """

        :param df_raw: raw data of
        :return:
        """
        asset = self.str_asset
        df_cc1 = pd.DataFrame(columns=[asset+'cc0_p', asset+'cc0_v', asset+'cc0_po'])
        dic_second_active_contract = self._second_traded_contract(df_raw=df_raw)
        tmp = pd.DataFrame(columns=[asset+'cc0_p', asset+'cc0_v', asset+'cc0_po'])
        for i in dic_second_active_contract.keys():
            tmp.loc[:, asset+'cc0_p'] = df_raw.loc[dic_second_active_contract[i], i+'_p']
            tmp.loc[:, asset+'cc0_v'] = df_raw.loc[dic_second_active_contract[i], i+'_v']
            tmp.loc[:, asset+'cc0_po'] = df_raw.loc[dic_second_active_contract[i], i+'_po']
            df_cc1 = df_cc1.append(tmp)
            tmp = pd.DataFrame(columns=[asset + 'cc0_p', asset + 'cc0_v', asset + 'cc0_po'])
        df_cc1 = df_cc1.sort_index(ascending=True)
        self.df_cc1 = df_cc1
        return df_cc1

    @staticmethod
    def last_trading_date(ls_str_month, str_date):
        dt_trading_date = dt.datetime.strptime(str_date, '%Y-%m-%d')
        if dt_trading_date.month in [int(i) for i in ls_str_month]:
            if dt_trading_date.day == 15:
                return str_date
            # TODO how to postpone last trading date for delivery month
            elif dt_trading_date.weekday()
            return None


    #
    # def plot_futures_curve(self, df_raw):
    #     asset = self.str_asset
    #     ls_str_month = self.ls_str_month



if __name__ == '__main__':
    ####################################################################
    #                PART I  Commodity Data Preparation                #
    ####################################################################
    # Establish class object for CommoditySpread
    test = CommoditySpread('RB', ['01', '05', '10'])

    # Raw Data with price, volume, and posi
    df_RB = pd.read_csv('/home/nealzc1991/PycharmProjects/Py4Invst/Fundamental/RB.csv')
    # df_RB = pd.read_csv('/home/nealzc/PycharmProjects/Py4Invst/Fundamental/RB.csv')
    df_RB.set_index('Date', inplace=True)

    # Construct most and second most active contract data series
    df_CC1 = test.active_contract(df_raw=df_RB)
    df_CC2 = test.second_active_contract(df_raw=df_RB)

    # For continuous contracts analysis, use df_CC1
    # Further adjustment needed for df_CC1
    #

    df_test = df_RB.loc['2017-01-01':'2018-01-01',:]