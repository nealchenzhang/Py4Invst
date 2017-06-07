#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Fri Mar 31 15:57:14 2017

# @author: nealcz @Aian_fund

# This program is personal trading platform desiged when employed in 
# Aihui Asset Management as a quantatitive analyst.
# 
# Contact: 
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class Analyzer(object):
    """
    A class for analyzing portfolio values
    
    Data:
    ===========================================================================    
        start_date:
        end_date:
        portfolio values:
        normalized portfolio values:
        portfolio returns:
    
        
    Portfolio Attributes: 
    ===========================================================================
        Normalized portfolio values
        Desriptive Statistics
        Draw a plot of normalized portfolio values
    
        
    Returns Attributes:
    ===========================================================================
        AvrgRet: return the average return
        CumRet: return the cumulative return
        StdRet: return the standard deviation
        NumRet: return the total number of returns
    
    Visualization Methods:
    ===========================================================================
        plot
    
    """
    
    #path = r"/media/nealcz/Data/Neal/Quant/PythonProject/ValuesFile"
    #os.chdir(path)
    
    def __init__(self, str_valuesfile):
        """
        Open valuesfile and read the portfolio values
        The default path for valuesfile = r"D:\\Neal\\Quant\\PythonProject\\ValuesFile\\"
        
        """
        reader = pd.read_csv(str_valuesfile, delimiter=',', header=None)
        ls_dates = []
        ls_port_values = []

        for i in reader.index:
            ls_dates.append(dt.datetime(int(reader.iloc[i][0]), 
                                        int(reader.iloc[i][1]),
                                        int(reader.iloc[i][2])))
            ls_port_values.append(float(reader.iloc[i][3]))
        
        
        print("Analyzing {} portfolio values." \
              .format(str_valuesfile.split('.')[0]))
        print("Backtesting starts from {} to {}." \
              .format(ls_dates[0].strftime('%Y-%m-%d'),
                      ls_dates[-1].strftime('%Y-%m-%d')))
        
        #======================Portfolio Attributes============================
        self.start_date = ls_dates[0].strftime('%Y-%m-%d')
        self.end_date = ls_dates[-1].strftime('%Y-%m-%d')
        self.PortValues = pd.DataFrame(ls_port_values, index=ls_dates)
        self.NormPortValues = self.PortValues \
                                    / self.PortValues.iloc[0]
        self.PortReturns = (self.NormPortValues \
                             / self.NormPortValues.shift(1) - 1)[1:]
        
        self.PortValues.plot()
        
        #=======================Return Attributes==============================
        self.AvrgRet = float(np.mean(self.PortReturns))
        self.CumRet = float(self.NormPortValues.iloc[-1])
        self.StdRet = float(np.std(self.PortReturns))
        self.NumRet = int(self.PortReturns.size)
