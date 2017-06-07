#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Wed Mar 22 13:57:13 2017

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

class Omega_Ratio(Analyzer):
    """
    This funciton is to calcuate the omega_ratio of returns of CTA investments.
    
    Analyzer:
    ===========================================================================
        name: Omega Ratio
        url: https://en.wikipedia.org/wiki/Omega_ratio
    
    Methods:
    ===========================================================================
        getName: get Name
        getUrl: get Url of this analyzer
        OmegaRatio: calculate the Omega Ratio during the period
        
    '''    
    """
    __name = "Omega Ratio"
    __Url = "https://en.wikipedia.org/wiki/Omega_ratio"
    
    def __init__(self, str_valuesfile, target_return=0.15, basis='monthly'):
        Analyzer.__init__(self, str_valuesfile)
        #======================================================================
        self.target_return = target_return
        self.basis = {'Daily':252, 'Weekly':52, 'Monthly':12}[basis]
        self.describe = "The Sharpe Ratio is {} based".format(basis.upper())
    
        if basis == 'monthly':
            self.target_return = self.target_return / 12
        else:
            print('Please re-enter the required returns data and target_return')
    
    def getName(self):
        return self.__name
    
    def getUrl(self):
        return self.__Url
        
    def OmegaRatio(self):
        """
        Omega_Ratio:
            The ratio of the average realized return in excess of a target 
        return (i.e., average upper partial moment) relative to the average 
        relaized loss below the same target return (i.e., average lower 
        partial moment)
        
        A better estiamte of downside risk and upside potential relative to the
        target return.
        
        """
        df = pd.DataFrame(self.PortReturns)
        target_return = self.target_return
        upper_partial_moment = float((df[df >= target_return] - \
                                      target_return).sum() / df.size)
        lower_partial_moment = float((-1 * df[df < target_return] + \
                                      target_return).sum() / df.size)
        
        Omega = upper_partial_moment / lower_partial_moment
        
        return Omega