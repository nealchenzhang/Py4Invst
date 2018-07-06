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

import os

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

    def __init__(self, df_Price):
        print('This is the class for regression analysis.')
        print('Data input here should be price data with datetime index')
        self.df_Price = df_Price
        self.df_Return = df_Price / df_Price.shift(1) - 1

    def Correlation_test(self, x, y):
        """
        Correlation test is aimed to check the returns correlation
        between variable x and variable y.

        :param x: independent variable
        :param y: dependent variable
        :return:
        """
        # Error check
        if (x in self.df_Price.columns.tolist()) and \
            (y in self.df_Price.columns.tolist()):
            print('Correlation Test for {} and {} begins:'.format(x,y))
        else:
            print('Please change the input.')

        print('# 1. Scatter Plot')
        print('-' * 40)
        print('Any outliers?')
        print('Any possibility of Spurious Correlation?')
        print('Any possibility of Nonlinear Relationship?')

        seaborn.lmplot(x=x, y=y, data=self.df_Price)
        plt.title('Price plot between {} and {}'.format(x, y))
        plt.show()

        seaborn.lmplot(x=x, y=y, data=self.df_Return)
        plt.title('Return plot between {} and {}'.format(x, y))
        plt.show()
        # plt.show()

        # print('# 2. Population Correlation Coefficient Test')
        # print('-' * 40)
        # print('H0: rho = 0')
        # number = self.df_Price.loc[:, x].count()
        # sample_corr = self.df_Price.loc[:, [x, y]].corr(method='pearson')[x][y]
        # cal_ttest = (sample_corr * np.sqrt(number - 2)) / np.sqrt(1 - sample_corr ** 2)
        # # Critical t-value: 2-tailed
        # two_tailed_alpha = [0.1, 0.05, 0.02]
        # from scipy import stats
        # for i in two_tailed_alpha:
        #     print('two-tailed test critical t with {:.4f} level of significance with df={}: {:.4f}'
        #           .format(i, number - 2, stats.t.ppf(1 - i / 2, df=number - 2)))
        # print('-' * 20)
        # print('if calculated t-value={:.4f} is greater than critical value\n'
        #       'we conclude that we reject the H0: rho=0'.format(cal_ttest))

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
            ans = input(i).lower()
            if (ans == 'yes') or (ans =='y'):
                print('ERROR '*3)
                print('Model is misspecified.\nPlease change the model.')
                break
            else:
                print('Test for ' + i + '\nis passed')
                print('-'*40)
                pass




if __name__ == '__main__':
    print('Current working directory is:')
    print(os.getcwd())
    df = pd.read_excel(os.getcwd()+'/cointegration.xls')
    df.set_index('Date', inplace=True)
    test = Regression_Analysis(df_Price=df)
    test.Correlation_test(x='i1701', y='rb1701')
    # test.Misspecification_check()