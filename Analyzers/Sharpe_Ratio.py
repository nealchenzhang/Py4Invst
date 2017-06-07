#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Thu Mar  2 14:17:31 2017

# @author: nealcz @Aian_fund

# This program is personal trading platform desiged when employed in 
# Aihui Asset Management as a quantatitive analyst.
# 
# Contact: 
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################
from __future__ import print_function
#print(__doc__)

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Analyzers.Analyzer import Analyzer

class Sharpe_Ratio(Analyzer):
    '''
    This analyzer calcuates the SharpeRatio of a strategy.
    
    Analyzer:
    ===========================================================================
        name: Sharpe Ratio
        url: https://en.wikipedia.org/wiki/Sharpe_ratio
    
    Methods:
    ===========================================================================
        getName: get Name
        getUrl: get Url of this analyzer
        SharpeRatio: calculate the Sharpe Ratio during the period
        
    '''
    __name = "Sharpe Ratio"
    __Url = "https://en.wikipedia.org/wiki/Sharpe_ratio"
    
    def __init__(self, str_valuesfile, basis='Daily'):
        Analyzer.__init__(self, str_valuesfile)
        #======================================================================
        self.basis = {'Daily':252, 'Weekly':52, 'Monthly':12}[basis]
        self.describe = "The Sharpe Ratio is {} based".format(basis.upper())

    def getName(self):
        return self.__name
    
    def getUrl(self):
        return self.__Url
        
    def SharpeRatio(self):
        """
        self.PortReturns: returns based on [basis] data
        """
        f_Sharpe_Ratio = np.sqrt(self.basis) * self.AvrgRet / self.StdRet
        print('Sharpe_Ratio from {} to {} is {:.4}.'\
              .format(self.start_date, self.end_date, f_Sharpe_Ratio))        
        return f_Sharpe_Ratio