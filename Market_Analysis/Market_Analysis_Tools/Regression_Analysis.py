import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

import seaborn

# Correlation & Regression
# Input: pandas DataFrame with price or return data series
data = pd.read_excel('./Market_Analysis/Market_Analysis_Tools/cointegration.xls')
# 1. Scatter Plot
seaborn.lmplot(x='i1701', y='rb1701', data=data)
print('Any outliers?')
print('Any possibility of Spurious Correlation?')
print('Any possibility of Nonlinear Relationship?')

# 2. Population Correlation Coefficient Test
# H0: rho = 0
number = data['i1701'].count()
sample_corr = data[['i1701', 'rb1701']].corr(method='pearson')['i1701']['rb1701']
cal_ttest = (sample_corr * np.sqrt(number-2)) / np.sqrt(1 - sample_corr**2)
# Critical t-value: 2-tailed
two_tailed_alpha = [0.1, 0.05, 0.02]
from scipy import stats
for i in two_tailed_alpha:
    print('two-tailed test critical t with {:.4f} level of significance with df={}: {:.4f}'
          .format(i, number-2, stats.t.ppf(1-i/2, df=number-2)))
print('-'*20)
print('if calculated t-value={:.4f} is greater than critical value\n'
      'we conclude that we reject the H0: rho=0'.format(cal_ttest))

# 3. Regression Analysis
import statsmodels.api as sm
import statsmodels.formula.api as smf
# 3.1 Regression Equation and Interpretation
# data preparation
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# determine y_vars & x_vars
# scatter plot
seaborn.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', kind='scatter')

# model specification
print('sales~TV+radio+newspaper')
model_spec = 'sales~TV+radio+newspaper'
lm_model = smf.ols(formula=model_spec, data=data).fit()
print(lm_model.summary())
