import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts

df_data = pd.read_excel('./Market_Analysis/Market_Analysis_Tools/cointegration.xls')
df_data.set_index('Date', inplace=True)
ts.adfuller(df_data['i1701'], maxlag=10, regression='c', autolag='AIC')

ts.adfuller(df_data['i1701'].diff(1).dropna(), maxlag=10, regression='c')
ts.adfuller(df_data['rb1701'].diff(1).dropna(), maxlag=10, regression='c')

import statsmodels.api as sm

model = sm.OLS(df_data['rb1701'], sm.add_constant(df_data['i1701'])).fit()
print(model.summary())