import numpy as np
import pandas as pd

ru_data = pd.read_excel('./Market_Analysis/Market_Analysis_Tools/RU.xls')
ru_data.set_index('Date', inplace=True)

data = ru_data.resample('M')