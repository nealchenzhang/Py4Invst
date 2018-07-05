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
# Statsmodel
print('sales~TV+radio+newspaper')
model_spec = 'sales~TV+radio+newspaper'
lm_model = smf.ols(formula=model_spec, data=data).fit()
print(lm_model.summary())

# scikit-learn model
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data['sales']

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X, y)

print(lm.intercept_)
zip(feature_cols, lm.coef_)
lm.score(X, y)

# Dummy Variable
# set a seed for reproducibility
np.random.seed(12345)

# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(data))
mask_large = nums > 0.5

# initially set size to small, then change roughly half to be large
data['size'] = 'small'
data.loc[mask_large, 'size'] = 'large'
data.head()
# create a new Series called IsLarge
data['IsLarge'] = data['size'].map({'small':0, 'large':1})
data.head()

# create X and y
feature_cols = ['TV', 'radio', 'newspaper', 'IsLarge']
X = data[feature_cols]
y = data['sales']

# instantiate, fit
lm = LinearRegression()
lm.fit(X, y)
print(lm.coef_)

# set a seed for reproducibility
np.random.seed(123456)

# assign roughly one third of observations to each group
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area'] = 'rural'
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'
data.head()
# create three dummy variables using get_dummies, then exclude the first dummy column
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]

# concatenate the dummy variable columns onto the original DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, area_dummies], axis=1)
data.head()

# create X and y
feature_cols = ['TV', 'radio', 'newspaper', 'IsLarge', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data['sales']

# instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print coefficients
print(lm.coef_)
