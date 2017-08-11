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

Managed Futures Portfolio Construction
Entry, Size, Exit, and Allocation.

"""

print(__doc__)
import pandas as pd
import numpy as np
import datetime as dt

class AMH(object):
    """
    This class is used to anlayze different asset class, specifically for
    commodities.
    
    The tools used to analyze trend are Signal_to_Noise and Market Divergence
    Index.
    
    Functions:
    ===========================================================================
        Signal_to_Noise
        rolling_SNR
        Market_Divergence_Index
    
    Methods:
    ===========================================================================
        .get_SNR
        .get_rolling_SNR
        .get_MDI
        
    """
    __Url = "https://en.wikipedia.org/wiki/Signal-to-noise_ratio"

    def __init__(self, ):



    def Signal_to_Noise(self, na_price):
        """
        Signal_to_Noise: the ratio of the overall trend to a series of price
        changes during the same period, or the ratio of magnitude of the trend 
        to the volatility around that trend.
            
        URL: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
            
        Input: 
            na_price: numpy array of some commodity prices
        
        Output:
            SNR: return an SNR value
            
        """
        SNR = np.absolute(na_price[-1]-na_price[0]) \
                         / np.absolute(pd.Series(na_price).diff()).sum()
        
        return SNR
    
    def rolling_SNR(df_price, lookback_period=10):
        """
        rolling_SNR is used to create a series of SNR based on lookback_period
        for a particular security or asset.
        
        """
        df_SNR = df_price.rolling(lookback_period).apply(Signal_to_Noise)
        
        return df_SNR
    
    def Market_Divergence_Index(df_markets_SNRs):
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
        MDI = df_markets_SNRs.apply(np.mean, axis=1)
        MDI.plot()
        return MDI

    def getUrl(self):
        return self.__Url