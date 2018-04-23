import numpy as np
import pandas as pd

ru_data = pd.read_excel('./Market_Analysis/Market_Analysis_Tools/RU.xls')
ru_data.set_index('Date', inplace=True)

data = ru_data.resample('M')
sample_data = ru_data.iloc[-100:]

cl = sample_data['Close'].copy()
lag = cl.shift(-1).copy()
mul = lag / cl

mul.iloc[-1] = cl.iloc[-1]

def multiplicative_adjustment4continuousactivefuturescontracts():

df_ru = pd.DataFrame({'Close': cl, 'Lag':lag})

