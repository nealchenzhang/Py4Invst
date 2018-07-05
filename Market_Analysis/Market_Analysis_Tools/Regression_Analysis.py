# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Thur July 5 19:18:14 2018

# @author: NealChenZhang

# This program is personal financial market analysis platform designed
# when employed in Everbright Securities as a FoHF analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com
# Working E-mail: chenzhang@ebscn.com

###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

import seaborn
from matplotlib import style
style.use('ggplot')


class Regression_Analysis(object):
    """
    A class for analyzing multiple regression model

    """

    def __init__(self, df_Data):
        print("This is the class for regression analysis.")
        self.df_Data = df_Data

    def Correlation_test(self):


    def Misspecification_check(self):
        print('This is the check for misspecification.')
        a = 'Omitting a Variable?'
        b = 'Improperly Transform the Variable?'
        # log(Market_Cap) not Market_Cap
        c = 'Incorrectly Pooling the Data? \n(Structural Change?)'
        d = 'Using a Lagged Dependent Variable\nas an Independent Variable?'
        e = 'Forecasting the Past?\n(Independent Variable should be consistent with Dependent Variable)'
        f = 'Measuring Independent Variables with Error?\n(Proxy Variable)'
        for i in [a, b, c, d, e, f]:
            if input(i).lower() is ('yes' or 'y'):
                print('Model is misspecified.\nPlease change the model.')
                break
            else:
                print('Test for ' + i + '\nis passed')
                print('-'*40)
                pass




if __name__ == '__main__':
    test = Regression_Analysis('fa')
    test.Misspecification_check()