#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Fri Mar 31 15:57:14 2017

# @author: NealChenZhang

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")


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
        Descriptive Statistics
        Draw a plot of normalized portfolio values
    
        
    Returns Attributes:
    ===========================================================================
        AvrgRet: return the average return
        StdRet: return the standard deviation
        NumRet: return the total number of returns
    
    Visualization Methods:
    ===========================================================================
        plot
    
    """

    def __init__(self, str_filepath):

        df_port = pd.DataFrame.from_csv(str_filepath)
        df_port["Normalized Value"] = df_port["Values"] / df_port["Values"].iloc[0]
        df_port["Returns"] = (df_port["Normalized Value"] / df_port["Normalized Value"].shift(1) - 1)[1:]

        df_port["Normalized Value"].plot()
        plt.show()
        # ======================Portfolio Attributes============================
        self.__start_date = df_port.index[0].strftime('%Y-%m-%d')
        self.__end_date = df_port.index[-1].strftime('%Y-%m-%d')
        self.__df_port = df_port
        # =======================Return Attributes==============================
        self.AvrgRet = float(np.mean(df_port["Returns"]))
        self.StdRet = float(np.std(df_port["Returns"]))
        self.NumRet = int(df_port["Returns"].size)

    def get_start_date(self):
        return self.__start_date

    def get_end_date(self):
        return self.__end_date

    def description(self):
        # print("Analyzing {} portfolio values.".format(str_filepath.split('.')[0]))
        print("Backtesting starts from {} to {}.".format(self.__start_date, self.__end_date))

    def get_df(self):
        return self.__df_port


if __name__ == "__main__":
    x = Analyzer("D:/Neal/Quant/PythonProject/ValuesFile/values1.csv")
    print(dir(x))
    x.description()
    try:
        x.__start_date = '2017'
        print(x.__start_date)
        x.description()
    except:
       pass
