#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Wed Mar 22 14:40:13 2017

# @author: NealChenZhang

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################

import numpy as np

from Backtest_Futures.Analyzers.Analyzer import Analyzer

class Sortino_Ratio(Analyzer):
    """
    This analyzer calcuates the SortinoRatio of a strategy.

    Analyzer:
    ===========================================================================
        name: Sortino Ratio
        url: https://en.wikipedia.org/wiki/Sortino_ratio

    Methods:
    ===========================================================================
        getName: get Name
        getUrl: get Url of this analyzer
        SortinoRatio: calculate the Sortino Ratio during the period

    """
    __name = "Sortino Ratio"
    __Url = "https://en.wikipedia.org/wiki/Sortino_ratio"
    
    def __init__(self, str_filepath, basis='Daily'):
        Analyzer.__init__(self, str_filepath)
        #======================================================================
        self.MT = {'Daily':252, 'Weekly':52, 'Monthly':12}[basis]
        self.basis = basis
        self.describe = "The Sortino Ratio is {} based".format(basis.upper())

    def getName(self):
        return self.__name
    
    def getUrl(self):
        return self.__Url
        
    def SortinoRatio(self):
        """
        df_port["Returns"]: returns based on [basis] data
        """        
        bs = self.MT
        df_rets = self.get_df()["Returns"].copy()
        # print(df_rets)

        f_down_std = np.sqrt((df_rets[df_rets<self.AvrgRet]-
                              self.AvrgRet).apply(np.square).sum()
                             / df_rets.size)

        f_Sortino_Ratio = float(np.sqrt(bs) * self.AvrgRet / f_down_std)

        print('{} Sortino_Ratio from {} to {} is {:.2}.'
                  .format(self.basis.upper(), self.get_start_date(), self.get_end_date(), f_Sortino_Ratio))
        return f_Sortino_Ratio