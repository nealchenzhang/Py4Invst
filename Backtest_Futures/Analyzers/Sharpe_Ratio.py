#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Fri Mar 31 17:31:02 2017

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

class Sharpe_Ratio(Analyzer):
    """
    This analyzer calculates the SharpeRatio of a strategy.

    Analyzer:
    ===========================================================================
        basis: Sharpe Ratio calculation based on {Daily, Weekly, Monthly}
        name: Sharpe Ratio
        url: https://en.wikipedia.org/wiki/Sharpe_ratio


    Methods:
    ===========================================================================
        get_Name: get Name
        get_Url: get Url of this analyzer
        SharpeRatio: calculate the Sharpe Ratio during the period

    """
    __name = "Sharpe Ratio"
    __Url = "https://en.wikipedia.org/wiki/Sharpe_ratio"
    
    def __init__(self, str_filepath, basis='Daily'):
        Analyzer.__init__(self, str_filepath)
        #======================================================================
        self.MT = {'Daily': 252, 'Weekly': 52, 'Monthly': 12}[basis]
        self.basis = basis
        self.describe = "The Sharpe Ratio is {} based".format(basis.upper())

    def get_Name(self):
        return self.__name
    
    def get_Url(self):
        return self.__Url
        
    def SharpeRatio(self):
        """
        self.PortReturns: returns based on [MT] data
        """
        f_Sharpe_Ratio = np.sqrt(self.MT) * self.AvrgRet / self.StdRet
        print('{} Sharpe_Ratio from {} to {} is {:.4}.'
              .format(self.basis.upper(), self.get_start_date(), self.get_end_date(), f_Sharpe_Ratio))
        return f_Sharpe_Ratio