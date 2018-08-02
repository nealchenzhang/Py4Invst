__author__ = 'Neal Chen Zhang'
import os
import datetime as dt

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style
style.use('ggplot')


class CommodityAnalysis(object):

    def __init__(self, str_asset, ls_str_month, last_trading_day=15):
        """

        :param str_asset: using UpperCase for underlying asset
        :param ls_str_month: Attention! e.g. for January, use 01 instead of 1
        """
        self.str_asset = str_asset
        self.ls_str_month = ls_str_month.copy()
        ls_str_month.append(ls_str_month[0])
        ls_month_diff = np.diff([int(i) for i in ls_str_month]).tolist()
        month_diff = np.array(ls_month_diff)
        month_diff[np.where(month_diff<0)] += 12
        self.month_diff = month_diff.copy()
        self.last_trading_day = last_trading_day
        self.ls_trading_date = None
        self.df_cc0 = None
        self.df_cc1 = None
        self.df_next_cc1 = None
        self.df_spread = None

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
        df_raw = df_raw.copy()
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
        df_raw = df_raw.copy()
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
                    x[po_cols[0]], x[po_cols[1]], c=0 # c is used to keep the func work
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

        :param df_raw: raw data
        :return: DataFrame of most active contract
        """
        asset = self.str_asset
        df_raw = df_raw.copy()
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
        self.ls_trading_date = df_cc0.index.tolist()
        return df_cc0

    def second_active_contract(self, df_raw):
        """

        :param df_raw: raw data
        :return: DataFrame of second active contract
        """
        asset = self.str_asset
        df_raw = df_raw.copy()
        df_cc1 = pd.DataFrame(columns=[asset+'cc1_p', asset+'cc1_v', asset+'cc1_po'])
        dic_second_active_contract = self._second_traded_contract(df_raw=df_raw)
        tmp = pd.DataFrame(columns=[asset+'cc1_p', asset+'cc1_v', asset+'cc1_po'])
        for i in dic_second_active_contract.keys():
            tmp.loc[:, asset+'cc1_p'] = df_raw.loc[dic_second_active_contract[i], i+'_p']
            tmp.loc[:, asset+'cc1_v'] = df_raw.loc[dic_second_active_contract[i], i+'_v']
            tmp.loc[:, asset+'cc1_po'] = df_raw.loc[dic_second_active_contract[i], i+'_po']
            df_cc1 = df_cc1.append(tmp)
            tmp = pd.DataFrame(columns=[asset + 'cc1_p', asset + 'cc1_v', asset + 'cc1_po'])
        df_cc1 = df_cc1.sort_index(ascending=True)
        self.df_cc1 = df_cc1
        return df_cc1

    def last_trading_date(self):
        """

        :return: list of last trading date for each active contract
        """
        ls_str_month = self.ls_str_month
        ls_trading_date = [dt.datetime.strptime(i, '%Y-%m-%d') for i in self.ls_trading_date]

        # Last trading date
        ls_years = list(set([i.year for i in ls_trading_date]))
        ls_month = [int(i) for i in ls_str_month]
        int_day = self.last_trading_day

        ls_last_trading_date = []
        for y in ls_years:
            for m in ls_month:
                dt_date = dt.datetime(year=y, month=m, day=int_day)
                if dt_date.strftime('%Y-%m-%d') in self.ls_trading_date:
                    ls_last_trading_date.append(dt_date)
                elif (dt_date+dt.timedelta(days=1)).strftime('%Y-%m-%d') in self.ls_trading_date:
                    ls_last_trading_date.append(dt_date+dt.timedelta(days=1))
                elif (dt_date+dt.timedelta(days=2)).strftime('%Y-%m-%d') in self.ls_trading_date:
                    ls_last_trading_date.append(dt_date + dt.timedelta(days=1))

        ls_last_trading_date.append(dt.datetime(2018,10,15))
        ls_last_trading_date.append(dt.datetime(2019, 1, 15))

        return ls_last_trading_date

    def plot_futures_curve(self, df_raw, year=2010):
        """
        Here, the futures curve is calculated based on the contract expiry date
        noramlly, for example, RB contract is expired on 15th of the month
        :param df_raw:
        :param year:
        :return:
        """
        asset = self.str_asset
        df_raw = df_raw.copy()
        ls_dates = self.last_trading_date()
        ls_dates.sort()

        df_raw.loc[:, 'Date'] = self.ls_trading_date
        df_raw.loc[:, 'Date'] = df_raw.loc[:, 'Date'].apply(pd.to_datetime)

        df_tmp = df_raw.where(
            df_raw.loc[:, 'Date'].apply(lambda i: i.year) == year
        ).dropna()

        fig, ax1 = plt.subplots(figsize=(12, 8))
        plt.ion()

        for i in df_tmp.loc[:, 'Date']:
            time_to_maturity = np.array(ls_dates) - i
            print(time_to_maturity)
            ix = np.where(
                (time_to_maturity < dt.timedelta(days=366)) &
                (dt.timedelta(days=0) < time_to_maturity)
            )
            t = [i.days for i in time_to_maturity[ix]]
            # print(t)
            y_month = [i.month for i in np.array(ls_dates)[ix]]
            str_month = [asset+('0'+str(i))[-2:]+'_p' for i in y_month]

            # Plotting
            plt.cla()
            ax1.set_xlim(0, 366)
            ax1.set_xlabel('Time to maturity')
            ax1.set_ylim(2000, 5000)
            ax1.set_ylabel('Prices')
            plt.title('Futures curve for year {}\nand date {}'.format(year, i.strftime('%Y-%m-%d')))
            ax1.plot(t, df_raw.loc[i.strftime('%Y-%m-%d'), str_month])

            plt.pause(0.1)

        plt.ioff()
        plt.show()

    def spread_cal(self, df_raw):
        """
        Spread analysis is based on this particular spread calculation.
        Rationale:
            spread is calculated based on the next most active contracts
            e.g., today is Dec 30
                if the most active contract is May contract,
                next most active contract is Oct contract,
                then spread = May - Oct
                instead of Jan - May

        :param df_raw:
        :return:
        """
        asset = self.str_asset
        df_raw = df_raw.copy()
        cols = [str(asset) + i for i in self.ls_str_month]
        position_cols = [i+'_po' for i in cols]

        df_cc0 = self.df_cc0
        df_cc1 = self.df_cc1

        df_next_cc1 = pd.DataFrame(columns=[asset+'nextcc1_p', asset+'nextcc1_v', asset+'nextcc1_po', 'month'])
        for i in position_cols:
            mon = int(i.split('_')[0][-2:])
            # find the period for the 'spread' when it can be traded
            ds_tmp = df_raw.loc[:, i] - df_cc1.loc[:, str(asset)+'cc1_po']
            ds_tmp = ds_tmp.where(ds_tmp == 0).dropna()
            ix = ds_tmp.index

            df_tmp= df_raw.loc[ix, [i.split('_')[0]+'_p', i.split('_')[0]+'_v', i.split('_')[0]+'_po']]
            df_tmp.columns = [(asset+'nextcc1_p'), (asset+'nextcc1_v'), (asset+'nextcc1_po')]
            df_tmp.loc[:, 'month'] = mon
            df_next_cc1 = df_next_cc1.append(df_tmp, sort=True)
        df_next_cc1 = df_next_cc1.sort_index(ascending=True)
        self.df_next_cc1 = df_next_cc1

        df_spread = pd.DataFrame(columns=['spread_price', 'diff_month'])
        df_spread.loc[:, 'spread_price'] = df_cc0.loc[:, asset+'cc0_p'] - df_next_cc1.loc[:, asset+'nextcc1_p']

        # find the appropriate month diff fro interest rates
        ls_month = [int(i) for i in self.ls_str_month]
        ls_month_diff = self.month_diff.copy()
        for month in ls_month:
            ix = ls_month.index(month)
            diff_month = ls_month_diff[ix-1]
            df_tmp = df_next_cc1.where(df_next_cc1.month == month).dropna()
            df_spread.loc[df_tmp.index, 'diff_month'] = diff_month

        return df_spread


if __name__ == '__main__':
    ####################################################################
    #                PART I  Commodity Data Preparation                #
    ####################################################################
    # Establish class object for CommoditySpread
    test = CommodityAnalysis('RB', ['01', '05', '10'])

    # Raw Data with price, volume, and positions
    df_RB = pd.read_csv('/home/nealzc1991/PycharmProjects/Py4Invst/Market_Analysis/Futures_Market/RB.csv')
    # df_RB = pd.read_csv('/home/nealzc/PycharmProjects/Py4Invst/Market_Analysis/Futures_Market/RB.csv')
    df_RB.set_index('Date', inplace=True)

    # Construct most and second most active contract data series
    df_cc0 = test.active_contract(df_raw=df_RB)
    df_cc1 = test.second_active_contract(df_raw=df_RB)

    # For continuous contracts analysis, use df_CC1
    # Further adjustment needed for df_CC1
    #

    # Dates for last trading dates
    last_dates = test.last_trading_date()
    # Plot futures curve
    # test.plot_futures_curve(df_raw=df_RB, year=2017)

    df_spread = test.spread_cal(df_raw=df_RB)
    print(df_spread)

    # #############################################################################
    # df_int3M = pd.read_csv('/home/nealzc/PycharmProjects/Py4Invst/Market_Analysis/Futures_Market/DR3M.csv')
    # # df_int3M = pd.read_csv('/home/nealzc1991/PycharmProjects/Py4Invst/Market_Analysis/Futures_Market/DR3M.csv')
    # df_int3M.dropna(inplace=True)
    # df_int3M.set_index('Date', inplace=True)
    #
    # df_result = pd.merge(df_RB, df_int3M, left_index=True, right_index=True, how='outer')
    #
    # df_result = df_result.dropna(thresh=2)
    # df_result.loc[:, 'DR3M'] = df_result.loc[:, 'DR3M'].fillna(method='bfill').fillna(method='ffill')


