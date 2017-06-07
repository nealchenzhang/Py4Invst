#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Fri Mar 31 17:31:02 2017

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

class Draw_Down(Analyzer):
    """
    This funciton is to calcuate the Draw_Down of portfolio values.
    
    Including: Maximum Drawdonw(MDD), High-water Mark(HWM),
               Maximum Drawdwon Duration(MDD Duration),
               MaxDrawDown_start, MaxDrawDown_end
    
    """
    __name = "Draw_Down"
    __Url = "https://en.wikipedia.org/wiki/Drawdown_(economics)"
    
    def __init__(self, str_valuesfile):
        Analyzer.__init__(self, str_valuesfile)
        #======================================================================
        normalized_values = self.NormPortValues.copy()
        n = normalized_values.size
        self.highwatermark = pd.Series(np.zeros(n),\
                                       index=normalized_values.index)
        self.drawdown = pd.Series(np.zeros(n), \
                                  index=normalized_values.index)
        self.drawdownduration = pd.Series(np.zeros(n), \
                                          index=normalized_values.index)
        for i in range(1, n):
            self.highwatermark.iloc[i] = max(self.highwatermark.iloc[i - 1], \
                                         float(normalized_values.iloc[i]))
            self.drawdown.iloc[i] = (self.highwatermark.iloc[i] - \
                                      float(normalized_values.iloc[i])) \
                                    / self.highwatermark.iloc[i]
            if self.drawdown.iloc[i] == 0:
                self.drawdownduration.iloc[i] = 0
            else:
                self.drawdownduration.iloc[i] = \
                                          self.drawdownduration.iloc[i - 1] + 1
        
        self.MaxDrawDown_end = self.drawdownduration.argmax()
        self.MaxDrawDown_start = normalized_values.index[\
                    normalized_values.index.get_loc(self.MaxDrawDown_end) - \
                                            int(max(self.drawdownduration))]
    
    def getName(self):
        return self.__name
    
    def getUrl(self):
        return self.__Url
        
    def Maximum_Drawdown(self):
        return max(self.drawdown)
    
    def Drawdown_Duration(self):
        return int(max(self.drawdownduration))
    
    def HWM(self):
        return max(self.highwatermark)
    
    def Drawdown_Duration_start(self):
        return self.MaxDrawDown_start
    
    def Drawdown_Duration_end(self):
        return self.MaxDrawDown_end