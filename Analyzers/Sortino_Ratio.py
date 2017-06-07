#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Wed Mar 22 14:40:13 2017

# @author: nealcz @Aian_fund

# This program is personal trading platform desiged when employed in 
# Aihui Asset Management as a quantatitive analyst.
# 
# Contact: 
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################
#print(__doc__)

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Analyzers.Analyzer import Analyzer

class Sortino_Ratio(Analyzer):
    '''
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
        
    '''
    __name = "Sortino Ratio"
    __Url = "https://en.wikipedia.org/wiki/Sortino_ratio"
    
    def __init__(self, str_valuesfile, basis='Daily'):
        Analyzer.__init__(self, str_valuesfile)
        #======================================================================
        self.basis = {'Daily':252, 'Weekly':52, 'Monthly':12}[basis]
        self.describe = "The Sortino Ratio is {} based".format(basis.upper())

    def getName(self):
        return self.__name
    
    def getUrl(self):
        return self.__Url
        
    def SortinoRatio(self):
        """
        self.PortReturns: returns based on [basis] data
        """        
        bs = self.basis
        df_rets = self.PortReturns.copy()
        f_down_std = float(((df_rets[df_rets<self.AvrgRet].dropna() - \
                 self.AvrgRet).apply(np.square).apply(np.sum) \
                / df_rets.size).apply(np.sqrt))
        
        f_Sortino_Ratio = np.sqrt(bs) * self.AvrgRet / f_down_std
        
        print('Sortino_Ratio from {} to {} is {:.2}.'\
                  .format(self.start_date, self.end_date, f_Sortino_Ratio))
        return f_Sortino_Ratio