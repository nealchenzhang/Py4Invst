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

    def __init__(self, y, ls_x, data_set):
        print('This is the class for regression analysis.')
        print('Data input here should be data with datetime index')
        self.data_set = data_set
        self.y = y
        self.ls_x = ls_x
        # self.df_Return = df_Price / df_Price.shift(1) - 1

    def Correlation_Test(self, x, y):
        """
        Correlation test is aimed to check the returns correlation
        between variable x and variable y.

        :param x: independent variable
        :param y: dependent variable
        :return:
        """
        # Error check
        if (x in self.data_set.columns.tolist()) and \
            (y in self.data_set.columns.tolist()):
            print('Correlation Test for {} and {} begins:'.format(x,y))
        else:
            print('Please change the input.')

        print('# 1. Scatter Plot')
        print('-' * 40)
        print('Any outliers?')
        print('Any possibility of Spurious Correlation?')
        print('Any possibility of Nonlinear Relationship?')

        seaborn.lmplot(x=x, y=y, data=self.data_set)
        plt.title('Price plot between {} and {}'.format(x, y))
        plt.show()

        # seaborn.lmplot(x=x, y=y, data=self.data_set)
        # plt.title('Return plot between {} and {}'.format(x, y))
        # plt.show()

        print('# 2. Population Correlation Coefficient Test')
        print('-' * 40)
        print('H0: rho = 0')
        number = self.data_set.loc[:, x].count()
        sample_corr = self.data_set.loc[:, [x, y]].corr(method='pearson')[x][y]
        cal_ttest = (sample_corr * np.sqrt(number - 2)) / np.sqrt(1 - sample_corr ** 2)
        # Critical t-value: 2-tailed
        two_tailed_alpha = [0.1, 0.05, 0.01]
        from scipy import stats
        print('-' * 40)
        print('Calculated t_value is {}.\nWith df = {}'.format(cal_ttest, number-2))
        for i in two_tailed_alpha:
            c_t = stats.t.ppf(1 - i / 2, df=number - 2)
            if (cal_ttest < -c_t) or (cal_ttest > c_t):
                print('Reject the null hypothesis at the {:.2%} level of significance'.format(i))
            else:
                print('We failed to reject the null hypothesis at the {:.2%} level of significance'.format(i))

    def Misspecification_Check(self):
        print('This is the check for misspecification.')
        a = 'Omitting a Variable?'
        b = 'Improperly Transform the Variable?'
        # log(Market_Cap) not Market_Cap
        c = 'Incorrectly Pooling the Data? \n(Structural Change?)'
        d = 'Using a Lagged Dependent Variable\nas an Independent Variable?'
        e = 'Forecasting the Past?\n(Independent Variable should be consistent with Dependent Variable)'
        f = 'Measuring Independent Variables with Error?\n(Proxy Variable)'
        num = 0
        for i in [a, b, c, d, e, f]:
            ans = input(i).lower()
            if (ans == 'yes') or (ans =='y'):
                print('ERROR '*3)
                print('Model is misspecified.\nPlease change the model.')
                break
            else:
                print('Test for ' + i + '\nis passed')
                print('-'*40)
                num += 1
                pass
        if num == 6:
            return 1
        else:
            pass

    def Multiple_Regression(self):
        """

        :param y:
        :param ls_x:
        :param data_set:
        :return:
        """
        print('# 3. Model Construction')
        y = self.y
        ls_x = self.ls_x
        data_set = self.data_set
        # Scatter Plot
        seaborn.pairplot(data_set, x_vars=ls_x, y_vars=y, kind='scatter')
        plt.show()

        # Model specification and OLS
        str_model_spec = str(y) + ' ~ '
        num = len(ls_x)
        for i in range(num):
            if i == 0:
                str_model_spec += ls_x[i]
            else:
                str_model_spec += ' + ' + str(ls_x[i])
        print(str_model_spec)
        lm_model = smf.ols(formula=str_model_spec, data=data_set).fit()
        return lm_model

    def Heteroskadasticity_Check(self, model):
        """

        :param model:
        :return:
        """
        # print('The effects of Heteroskedasticity:')
        # print('The standard errors are usually unreliable estimates.\n'
        #       "The coefficient estimates aren't affected.\n"
        #       "If the standard errors are too small, but the coefficient estimates"
        #       " themselves are not affected,\nthe t-statistics will be too large and"
        #       " the null hypothesis of no statistical significance is rejected too"
        #       " often.\n"
        #       "The F-test is also unreliable.")
        print('# 4. Heteroskedasticity')
        print('-' * 40)
        print('H0: no conditional heteroskedasticity')
        ds_res = model.resid
        ds_res.name = 'resid'
        data_res = pd.concat([self.data_set.loc[:, self.ls_x], ds_res], axis=1)

        # Resid model specification and OLS
        str_resid_model_spec = 'resid' + ' ~ '
        k = len(self.ls_x)
        for i in range(k):
            if i == 0:
                str_resid_model_spec += self.ls_x[i]
            else:
                str_resid_model_spec += ' + ' + str(self.ls_x[i])
        resid_reg = smf.ols(formula=str_resid_model_spec, data=data_res).fit()
        BP = resid_reg.rsquared * ds_res.size

        # Hypothesis test
        one_tailed_alpha = [0.1, 0.05, 0.01]
        from scipy import stats
        print('-' * 40)
        print('Calculated BP value is {}.\nWith df = {}'.format(BP, k))
        num = 0
        for i in one_tailed_alpha:
            c_chi = stats.chi2.ppf(q=1-i, df=k)
            if BP > c_chi:
                print('Reject the null hypothesis at the {:.2%} level of significance'.format(i))
                print('#'*80)
                print('May have a problem with conditional heteroskedasticity.')
            else:
                print('We failed to reject the null hypothesis at the {:.2%} level of significance'.format(i))
                num += 1
        if num == 3:
            return 1
        else:
            return 0

    def Autocorrelation_Check(self, model):
        """

        :param model:
        :return:
        """
        print('# 5. Autocorrelation')
        print('-' * 40)
        print('H0: no positive serial correlation')
        model.resid.plot()
        plt.show()
        DW = sm.stats.durbin_watson(model.resid)
        print('The calculated DW is {}'.format(DW))
        print('DW~=2(1-r)')
        print('r~={}'.format(1-DW/2))
        k = len(self.ls_x)
        n = self.data_set.loc[:, self.y].size
        print('Degree of freedom is {} with {} independent variables.'.format(n, k))
        print('If {} < DL, we reject the null hypothesis and '
              'conclude that Positive Serial Correlation exists.'.format(DW))
        print('If {} > DU, we failed to reject the null hypothesis.'.format(DW))

    def Multicollinearity_Check(self, model):
        """

        :return:
        """
        print('# 6. Multicollinearity')
        print('-' * 40)
        print("F_pvalue is {:.4f}".format(model.f_pvalue))
        print("R-squared is {:.4f}".format(model.rsquared))
        print("Adj R-squared is {:.4f}".format(model.rsquared_adj))
        print(model.pvalues)
        print("If F_test is statistically significant\nand R-squared is high,"
              "while t-tests indicate that\nnone of the individual coefficients is"
              "significantly different thant zero, multicollinearity may exist.")

    def MultipleRegression_Assessment(self):
        print('=' * 80)
        print('MultipleRegression_Assessment')
        print('=' * 80)
        if self.Misspecification_Check() == 1:
            print('Pass the model misspecification check.')
        else:
            print('Please correct the model misspecification.')
        print('=' * 80)
        lm_model = self.Multiple_Regression()
        print(lm_model.summary())
        print('Are individual coefficients statistically significant?')
        print('Is the model statistically significant?')
        print('=' * 80)
        print('Is heteroskedasticity present?')
        if self.Heteroskadasticity_Check(lm_model) == 1:
            print('Conditional heteroskedasticity is not found.')
        else:
            print('Try use White-corrected standard errors.\nor GLS model.')
        print('=' * 80)
        # TODO change the criteria with DW use the econometrics material MFIN 701
        self.Autocorrelation_Check(lm_model)
        self.Multicollinearity_Check(lm_model)


if __name__ == '__main__':
    print('Current working directory is:')
    print(os.getcwd())
    df = pd.read_excel(os.getcwd()+'/cointegration.xls')
    df.set_index('Date', inplace=True)
    df.i1701 = np.log(df.i1701)
    df.rb1701 = np.log(df.rb1701)
    test = Regression_Analysis(ls_x=['i1701'], y='rb1701',data_set=df)
    # test.Correlation_Test(x='i1701', y='rb1701')
    # test.Misspecification_Check()
    # model = test.Multiple_Regression();

    test.MultipleRegression_Assessment()

    # data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
    # test = Regression_Analysis(y='sales', ls_x=['TV', 'radio', 'newspaper'], data_set=data)
    # test.MultipleRegression_Assessment()

    # df_data = pd.read_excel('./Market_Analysis/Market_Analysis_Tools/cointegration.xls')
    # df_data.set_index('Date', inplace=True)
    # df_data = df_data.loc[:,'i1701']

