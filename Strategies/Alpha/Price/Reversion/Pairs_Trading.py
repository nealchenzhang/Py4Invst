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
    def __init__(self, stock_lists, stock_data, train_data, test_data):
        self.stock_lists = stock_lists
        self.stock_data = stock_data
        self.train_data = train_data
        self.test_data = test_data
    
    
    
    '''
    ###########################################################################
                                Data Handling
    ###########################################################################
    '''
    def normalize_data(self, data):
        normalized_data = data / data.iloc[0, :]
        return normalized_data

    '''
    ###########################################################################
                                Picking Pairs
    ###########################################################################
    '''
    # Picking to pairs
    # Different Measures should be tested
    # Here distance measure has been used
    def distance_measure(self, a, b, data):
        df_normalized_price = self.normalize_data(data)
        ans = (df_normalized_price[a] - df_normalized_price[b]).apply(np.square).sum()
        return ans
    
    def stat_results(self, a, b, data):
        df_normalized_price = self.normalize_data(data)
        mean = (df_normalized_price[a] - df_normalized_price[b]).mean()
        std = (df_normalized_price[a] - df_normalized_price[b]).std()
        return mean, std
    
    '''
    ###########################################################################
                                    Training
    ###########################################################################
    '''
    
    ## Picking Results
    def df_picking_results(self):
        # Within one sector stock lists
        stock1 = []
        stock2 = []
        measure = []
        mean = []
        std = []
        stock_lists = self.stock_lists
        for i in stock_lists:
            m = stock_lists.index(i)
            if m == len(stock_lists)-1:
                pass
            else:
                for j in stock_lists[m+1:]:
                    stock1.append(i)
                    stock2.append(j)
                    measure.append(self.distance_measure(i, j, self.train_data))
                    a, b = self.stat_results(i, j, self.train_data)
                    mean.append(a)
                    std.append(b)
    
        df_picking_result_train = pd.DataFrame()
        df_picking_result_train['Measure'] = pd.Series(measure)
        df_picking_result_train['Stock1'] = pd.Series(stock1)
        df_picking_result_train['Stock2'] = pd.Series(stock2)
        df_picking_result_train['Diff_Mean'] = pd.Series(mean)
        df_picking_result_train['Diff_Std'] = pd.Series(std)
    
        df_picking_result_train = df_picking_result_train.sort_values('Measure')
        
        return df_picking_result_train
    
    ## Training Results and visualization    
    def visualization(self, nrows=2, ncolumns=5):
        npairs = nrows * ncolumns

        df_output = self.df_picking_results()
        Train_Data = self.train_data
        
        # Plotting
        fig, ax = plt.subplots(nrows, ncolumns, sharex=True)

        for i in range(nrows * ncolumns):
            ix = np.unravel_index(i, ax.shape)
            p = df_output.iloc[0:npairs,:].loc[:,['Stock1','Stock2']].iloc[i].values
            Train_Data[p.tolist()].plot(ax=ax[ix])
        plt.show()
    
    '''
    ###########################################################################
                                    Testing
    ###########################################################################
    '''
    def back_testing(self, nrows=2, ncolumns=5):
        npairs = nrows * ncolumns
        df_output = self.df_picking_results()
        Test_Data = self.test_data
        # Plotting
        fig, ax = plt.subplots(nrows, ncolumns, sharex=True)
        
        for i in range(nrows * ncolumns):
            p = df_output.iloc[0:npairs,:].loc[:,['Stock1','Stock2']].iloc[i].values
            mean = df_output.iloc[0:npairs,:].loc[:,'Diff_Mean'].iloc[i]
            std = df_output.iloc[0:npairs,:].loc[:,'Diff_Std'].iloc[i]
            pairs = p.tolist()
            tmp = Test_Data[pairs]
            tmp['Spread'] = -tmp.diff(axis=1).iloc[:, -1]
            tmp['Z_score'] = (tmp['Spread'] - mean) / std
        
            # Entry Signal
            upper_b = 2
            bottom_b = 1
            # Z_score > 2 or Z_score < -2
            df_entry_sell = tmp.where((tmp['Z_score']>upper_b) & (tmp['Z_score'].diff() >=0))
            df_entry_buy = tmp.where((tmp['Z_score']<-upper_b) & (tmp['Z_score'].diff() <=0))
            
            # Exit Signal
            # Z_score < 1 or Z_score > -1
            df_exit_buy = tmp.where((tmp['Z_score']<bottom_b) & (0<tmp['Z_score']) & (tmp['Z_score'].diff() <=0))
            df_exit_sell = tmp.where((tmp['Z_score']>-bottom_b) & (tmp['Z_score']<0) & (tmp['Z_score'].diff() >=0))
            
            ix = np.unravel_index(i, ax.shape)

            x = ax[ix].plot(tmp.index, tmp.drop('Spread', axis=1))
            ax[ix].legend(x, tuple(pairs))
            
            ax[ix].scatter(df_entry_sell.index, df_entry_sell['Z_score'], marker='o', color='r')
            ax[ix].scatter(df_exit_buy.index, df_exit_buy['Z_score'], marker='x', color='r')
            ax[ix].plot(tmp.index, np.array([2]*tmp.index.size))
            ax[ix].plot(tmp.index, np.array([1]*tmp.index.size))
            
            ax[ix].scatter(df_entry_buy.index, df_entry_buy['Z_score'], marker='o', color='b')
            ax[ix].scatter(df_exit_sell.index, df_exit_sell['Z_score'], marker='x', color='b')
            ax[ix].plot(tmp.index, np.array([-2]*tmp.index.size))
            ax[ix].plot(tmp.index, np.array([-1]*tmp.index.size))
          
        plt.show()
        
if __name__=='__main__':

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

    # Choose the proper training and test dataset
    start = dt.datetime(2016, 1, 1)
    end_1 = dt.datetime(2016, 12, 31)
    start_2 = dt.datetime(2017, 1, 1)
    end = dt.datetime(2017, 6, 30)
        
    gold_train = gold.loc[start.strftime('%Y-%m-%d'):end_1.strftime('%Y-%m-%d'), :]
    gold_test = gold.loc[start_2.strftime('%Y-%m-%d'):end.strftime('%Y-%m-%d'), :]

    gold_mining_train = gold_mining.loc[start.strftime('%Y-%m-%d'):end_1.strftime('%Y-%m-%d'), :]
    gold_mining_test = gold_mining.loc[start_2.strftime('%Y-%m-%d'):end.strftime('%Y-%m-%d'), :]

    # Gold and Gold_Mining instancse
    ###########################################################################
    Gold = PairsTrading(gold_industry, gold, gold_train, gold_test)
#    Gold.df_picking_results()
#    Gold.visualization()
#    Gold.back_testing()
    
    Gold_mining = PairsTrading(gold_mining_industry, gold_mining, gold_mining_train, gold_mining_test)
    
    stock1 = []
    stock2 = []
    measure = []
    mean = []
    std = []
    for i in Gold.stock_lists:
        for j in Gold_mining.stock_lists:
            stock1.append(i)
            stock2.append(j)
            measure.append((Gold.stock_data[i] - Gold_mining.stock_data[j]).apply(np.square).sum())
            mean.append((Gold.stock_data[i] - Gold_mining.stock_data[j]).mean())
            std.append((Gold.stock_data[i] - Gold_mining.stock_data[j]).std())
            
    df_result = pd.DataFrame()
    df_result['Measure'] = pd.Series(measure)
    df_result['Stock1'] = pd.Series(stock1)
    df_result['Stock2'] = pd.Series(stock2)
    df_result['Diff_Mean'] = pd.Series(mean)
    df_result['Diff_Std'] = pd.Series(std)
    
    df_result = df_result.sort_values('Measure')
    ###########################################################################
    nrows= 2
    ncolumns = 5
    npairs = nrows * ncolumns
    df_output = df_result
    Test_Data = Gold.test_data.join(Gold_mining.test_data)
    # Plotting
    fig, ax = plt.subplots(nrows, ncolumns, sharex=True)
    
    for i in range(nrows * ncolumns):
        p = df_output.iloc[0:npairs,:].loc[:,['Stock1','Stock2']].iloc[i].values
        mean = df_output.iloc[0:npairs,:].loc[:,'Diff_Mean'].iloc[i]
        std = df_output.iloc[0:npairs,:].loc[:,'Diff_Std'].iloc[i]
        pairs = p.tolist()
        tmp = Test_Data[pairs]
        tmp['Spread'] = -tmp.diff(axis=1).iloc[:, -1]
        tmp['Z_score'] = (tmp['Spread'] - mean) / std
    
        # Entry Signal
        upper_b = 1.5
        bottom_b = 1
        # Z_score > 2 or Z_score < -2
        df_entry_sell = tmp.where((tmp['Z_score']>upper_b) & (tmp['Z_score'].diff() >=0))
        df_entry_buy = tmp.where((tmp['Z_score']<-upper_b) & (tmp['Z_score'].diff() <=0))
        
        # Exit Signal
        # Z_score < 1 or Z_score > -1
        df_exit_buy = tmp.where((tmp['Z_score']<bottom_b) & (0<tmp['Z_score']) & (tmp['Z_score'].diff() <=0))
        df_exit_sell = tmp.where((tmp['Z_score']>-bottom_b) & (tmp['Z_score']<0) & (tmp['Z_score'].diff() >=0))
        
        ix = np.unravel_index(i, ax.shape)

        x = ax[ix].plot(tmp.index, tmp.drop('Spread', axis=1))
        ax[ix].legend(x, tuple(pairs))
        
        ax[ix].scatter(df_entry_sell.index, df_entry_sell['Z_score'], marker='o', color='r')
        ax[ix].scatter(df_exit_buy.index, df_exit_buy['Z_score'], marker='x', color='r')
        ax[ix].plot(tmp.index, np.array([upper_b]*tmp.index.size))
        ax[ix].plot(tmp.index, np.array([bottom_b]*tmp.index.size))
        
        ax[ix].scatter(df_entry_buy.index, df_entry_buy['Z_score'], marker='o', color='b')
        ax[ix].scatter(df_exit_sell.index, df_exit_sell['Z_score'], marker='x', color='b')
        ax[ix].plot(tmp.index, np.array([-upper_b]*tmp.index.size))
        ax[ix].plot(tmp.index, np.array([-bottom_b]*tmp.index.size))
      
    plt.show()
    