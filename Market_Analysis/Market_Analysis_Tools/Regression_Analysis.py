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
"""
This module is based on Regression Analysis.

Guidelines:
    1. Is the model correctly specified?
        No --> Correct the model mis-specification
        Yes --> 2.
    2. Are individual coefficients statistically significant? (t-test)
                            AND
       Is the model statistically significant? (F-test)
        No --> Begin with another model
        Yes --> 3.
    3. Is heteroskedasticity present?
        No --> 4.
        Yes --> Conditional? (Breusch-Pagan Chi-square test)
                Yes --> Use White-corrected standard errors --> 4.
                No --> 4.
    4. Is serial correlation present? (Durbin-Watson test)
        No --> 5.
        Yes --> Use Hansen method to adjust standard errors -->5.
    5. Does model have significant multi-collinearity?
        Yes --> Drop one of the correlated variables --> Use the model!
        No --> Use the model!
"""
from __future__ import division
from __future__ import print_function
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm  # Generally use sm.api
import scipy.stats as scs # Scipy stats

class Regrs_Analysis(object):
    def __init__(self, df_data):
        self.data = df_data.copy()
        
    def Scatter_Plot(self, x_axis, y_axis):
        """
        A scatter plot of the values of two variables: X/Y pair
        """
        plt.scatter(x=self.data[x_axis], y=self.data[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
    
    def Corr_Analysis(self, x_axis, y_axis):
        df_corr = self.data[[x_axis, y_axis]].corr()
        corr = df_corr[x_axis][y_axis]
        n = self.data[x_axis].size
        t = corr*np.sqrt(n-2) / np.sqrt(1 - np.square(corr))
        print('Correlation between {} and {} is {}'.format(x_axis, y_axis, corr))
        print('Significance test: two-tailed t-test\nH0: rho=0\ndegree of freedom: {}\nt-values is {}'.format(n, t))
        return df_corr

    #def Simple_Linear_Regrs(self, x_axis, y_axis):
        
        
if __name__ == '__main__':
    ex_a = Regrs_Analysis(df_data)
    ex_a.Scatter_Plot()
    ex_a.Corr_Analysis