# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Wed Jul 19 08:59:52 2017

# @author: nealcz @Aian_fund

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################
import numpy as np
import pandas as pd
#import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt

class PairsTrading(object):
    def __init__(self, stock_lists, stock_data):
        self.stock_lists = stock_lists
        self.stock_data = stock_data

    def normalize_data(self):
        data = self.stock_data
        normalized_data = data / data.iloc[0, :]
        return normalized_data

    # There should be difference signal
    # Here distance measure has been used
    def distance_measure(self, a, b):
        df_normalized_price = self.normalize_data()
        ans = (df_normalized_price[a] - df_normalized_price[b]).apply(np.square).sum()
        return ans
    
    def df_results(self):
        # Within one sector stock lists
        stock1 = []
        stock2 = []
        measure = []
        stock_lists = self.stock_lists
        for i in stock_lists:
            m = stock_lists.index(i)
            if m == len(stock_lists)-1:
                pass
            else:
                for j in stock_lists[m+1:]:
                    stock1.append(i)
                    stock2.append(j)
                    measure.append(self.distance_measure(i, j))
                    #print(i+' and '+j+' : ', self.distance_measure(self.data_train, i, j))
    
        df_result_train = pd.DataFrame()
        df_result_train['Measure'] = pd.Series(measure)
        df_result_train['Stock1'] = pd.Series(stock1)
        df_result_train['Stock2'] = pd.Series(stock2)
    
        df_result_train = df_result_train.sort_values('Measure')
        return df_result_train
    
    def visualization(self, nrows=2, ncolumns=5):

        df_output = self.df_results()
        Data = self.stock_data
        # Plot
        fig, ax = plt.subplots(nrows, ncolumns, sharex=True)

        for i in range(nrows * ncolumns):
            ix = np.unravel_index(i, ax.shape)
            p = df_output.iloc[0:10,:].loc[:,['Stock1','Stock2']].iloc[i].values
            Data[p.tolist()].plot(ax=ax[ix])

        plt.show()
        
if __name__=='__main__':

    start = dt.datetime(2016, 1, 1)
    end_1 = dt.datetime(2016, 12, 31)
    start_2 = dt.datetime(2017, 1, 1)
    end = dt.datetime(2017, 6, 30)

    # Step 1: List down all the stocks in the industry
    gold_industry = ['ABX', 'NEM', 'GG', 'AEM', 'GOLD',
                     'AU', 'KGC', 'BVN', 'AUY', 'GFI',
                     'EGO', 'AGI', 'IAG', 'SBGL', 'OR',
                     'HMY']
    gold_mining_industry = ['AG', 'CDE', 'FNV', 'HL',
                            'MUX', 'TAHO']

    # Step 2: Collect last one year stock price data
    gold = pd.DataFrame()
    gold_mining = pd.DataFrame()

    '''
    for i in gold_industry:
        try:
            tmp = pdr.get_data_google(i, start, end)
            gold[i] = tmp['Close']
        except:
            print(i + ' is not available')

    for i in gold_mining_industry:
        try:
            tmp = pdr.get_data_google(i, start, end)
            gold_mining[i] = tmp['Close']
        except:
            print(i + ' is not available')
    '''
    gold = pd.read_csv('gold_industry.csv')
    gold_mining = pd.read_csv('gold_mining_industry.csv')

    gold['Date'] = gold['Date'].apply(pd.to_datetime)
    gold_mining['Date'] = gold_mining['Date'].apply(pd.to_datetime)

    gold = gold.set_index('Date')
    gold_mining = gold_mining.set_index('Date')

    # Normalize Price
    gold = gold / gold.iloc[0, :]
    gold_mining = gold_mining / gold_mining.iloc[0, :]

    gold_train = gold.loc[start.strftime('%Y-%m-%d'):end_1.strftime('%Y-%m-%d'), :]
    gold_test = gold.loc[start_2.strftime('%Y-%m-%d'):end.strftime('%Y-%m-%d'), :]

    gold_mining_train = gold_mining.loc[start.strftime('%Y-%m-%d'):end_1.strftime('%Y-%m-%d'), :]
    gold_mining_test = gold_mining.loc[start_2.strftime('%Y-%m-%d'):end.strftime('%Y-%m-%d'), :]

    # Gold and Gold_Mining instance
    Gold = PairsTrading(gold_industry, gold_train)
    Gold.df_results()
    Gold.visualization()
    #
    # Gold_mining = PairsTrading(gold_mining_industry, gold_mining_train)
    # Gold_mining.df_results()
    # Gold_mining.visualization()
    #
    # stock1 = []
    # stock2 = []
    # measure = []
    # for i in Gold.stock_lists:
    #     for j in Gold_mining.stock_lists:
    #         stock1.append(i)
    #         stock2.append(j)
    #         measure.append((Gold.stock_data[i] - Gold_mining.stock_data[j]).apply(np.square).sum())
    #
    # df_result = pd.DataFrame()
    # df_result['Measure'] = pd.Series(measure)
    # df_result['Stock1'] = pd.Series(stock1)
    # df_result['Stock2'] = pd.Series(stock2)
    #
    # df_result = df_result.sort_values('Measure')

            