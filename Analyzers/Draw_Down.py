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
import pandas as pd
import numpy as np

from Analyzers.Analyzer import Analyzer


class Draw_Down(Analyzer):
    """
    This function is to calculate the Draw_Down of portfolio values.

    Analyzer:
    ===========================================================================
        name: Sharpe Ratio
        url: https://en.wikipedia.org/wiki/Sharpe_ratio


    Methods:
    ===========================================================================
        get_Name: get Name
        get_Url: get Url of this analyzer
        Maximum_Drawdown: calculate the MDD during the period
        Drawdown_Duration: return the DD Duration during the period
        HWM: return the HWM of the portfolio
        Drawdown_Duration_start: return when the drawdown duration starts
        Drawdown_Duration_end: return when the drawdown duration ends

    """
    __name = "Draw_Down"
    __Url = "https://en.wikipedia.org/wiki/Drawdown_(economics)"

    def __init__(self, str_filepath):
        Analyzer.__init__(self, str_filepath)

        # ======================================================================
        normalized_values = self.get_df()["Normalized Value"].copy()
        n = normalized_values.size
        highwatermark = pd.Series(np.zeros(n),
                                  index=normalized_values.index)
        drawdown = pd.Series(np.zeros(n),
                             index=normalized_values.index)
        drawdownduration = pd.Series(np.zeros(n),
                                     index=normalized_values.index)
        for i in range(1, n):
            highwatermark.iloc[i] = max(highwatermark.iloc[i - 1],
                                        float(normalized_values.iloc[i]))
            drawdown.iloc[i] = (highwatermark.iloc[i] -
                                float(normalized_values.iloc[i])) / highwatermark.iloc[i]
            if drawdown.iloc[i] == 0:
                drawdownduration.iloc[i] = 0
            else:
                drawdownduration.iloc[i] = drawdownduration.iloc[i - 1] + 1

        MaxDrawDown_end = drawdownduration.argmax()
        MaxDrawDown_start = normalized_values.index[normalized_values.index.get_loc(MaxDrawDown_end) -
                                                    int(max(drawdownduration))]

        # ======================================================================
        dic = {"drawdown": drawdown,
               "drawdownduration": drawdownduration,
               "highwatermark": highwatermark,
               "MaxDrawDown_start": MaxDrawDown_start,
               "MaxDrawDown_end": MaxDrawDown_end}

        for i in dic.keys():
            setattr(self, i, dic[i])

    def get_Name(self):
        return self.__name

    def get_Url(self):
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

# if __name__ == "__main__":
#     x = Draw_Down("D:/Neal/Quant/PythonProject/ValuesFile/values1.csv")
#     # print(dir(x))
#     # x.description()
#     # print(x.get_df())
#     # print(x.getName())
#     # print(x.getUrl())
#     print(x.drawdown, x.HWM(), x.highwatermark)