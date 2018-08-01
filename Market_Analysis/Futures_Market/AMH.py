#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Mon Mar 20 17:00:38 2017

# @author: NealChenZhang

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################
"""
This module is based on Adaptive Markets Hypothesis.

Following functions are used to measuring market divergence:
SNR(signal-to-noise-ratio), MDI(market divergence index).
"""

# print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


class SNR(object):
    """
    The tools used to analyze trend within asset class is Signal_to_Noise,
    specifically for commodities.
    
    Functions:
    ===========================================================================
        Signal_to_Noise: function to calculate SNR
        rolling_SNR: return SNR series based on the lookback_period
        # Market_Divergence_Index
    
    Methods:
    ===========================================================================
        .get_Asset_price: get df_price based on asset, start_date, and end_date
        
    """
    # print(__doc__)
    __name = "SNR"
    __Url = "https://en.wikipedia.org/wiki/Signal-to-noise_ratio"

    def __init__(self, asset, start_date="2017-01-01", end_date="2017-06-30"):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date

    def get_Asset_price(self):
        """
        # get price series for this asset

        Have to know the start_date, end_date and path for the csv files
        """
        return pd.DataFrame.from_csv(self.asset+".csv").Close.rename(self.asset)

    def Signal_to_Noise(self, na_price):
        """
        Signal_to_Noise: the ratio of the overall trend to a series of price
        changes during the same period, or the ratio of magnitude of the trend 
        to the volatility around that trend.
            
        Input: 
            na_price: numpy array of some commodity prices
        
        Output:
            SNR: return an SNR value
            
        """
        SNR = np.absolute(na_price[-1]-na_price[0]) / np.absolute(pd.Series(na_price).diff()).sum()
        
        return SNR
    
    def rolling_SNR(self, lookback_period=10):
        """
        rolling_SNR is used to create a series of SNR based on lookback_period
        for a particular security or asset.
        
        """
        df_price = self.get_Asset_price()
        df_SNR = df_price.rolling(lookback_period).apply(self.Signal_to_Noise)
        return df_SNR

    def get_Name(self):
        return self.__name

    def get_Url(self):
        return self.__Url


# ###############################################################################
#
#                           Market Divergence Index
#
# ###############################################################################

class MDI(object):
    """
    The tools used to analyze trend across asset class is Market Divergence Index.

    Functions:
    ===========================================================================
        Market_Divergence_Index:

    Methods:
    ===========================================================================
        .get_MDI

    """
    # print(__doc__)
    __name = "MDI"
    # __Url = "https://en.wikipedia.org/wiki/Signal-to-noise_ratio"

    def __init__(self,
                 start_date="2017-01-01",
                 end_date="2017-06-30",
                 markets=["ru"],
                 lookback_period=10):
        self.start_date = start_date
        self.end_date = end_date
        self.markets = markets
        self.lookback_period = lookback_period

    def get_df_Markets_SNRs(self, lookback_period=10):
        ls_markets = self.markets
        df_Markets_SNRs = pd.DataFrame()
        for each in ls_markets:
            # print(each)
            tmp = SNR(each, self.start_date, self.end_date)\
                .rolling_SNR(lookback_period=lookback_period)
            # print(tmp)
            # print(type(tmp))
            # print("-"*30)
            df_Markets_SNRs[each] = tmp
        return df_Markets_SNRs


    def Market_Divergence_Index(self):
        """
        Market_Divergence_Index(MDI):
            The average of the relevant SNRs for a given observation period (n)
            and number of markets (M).

        Implication:
            The greater the MDI, the greater the trends are across markets.

        Input:
            df_markets_SNRs: DataFrame for SNRs of different markets
            # Asset class for different industries

        Output:
            MDI: return a series of MDI values

        """
        lookback_period = self.lookback_period
        df_Marets_SNRs = self.get_df_Markets_SNRs(lookback_period=lookback_period)
        # print(df_Marets_SNRs)
        MDI = df_Marets_SNRs.apply(np.mean, axis=1)
        MDI.plot()
        plt.show()
        return MDI