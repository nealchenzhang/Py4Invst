import numpy as np
import pandas as pd

from Analyzers import Analyzer
from Analyzers import Draw_Down
from Analyzers import *

filepath = "D:/Neal/Quant/PythonProject/ValuesFile/values1.csv"

# a = Sharpe_Ratio(filepath, basis="Monthly")
a = Sortino_Ratio(filepath)
try:
    print("MaxDuration: ", a.Maximum_Drawdown())
except:
    print("No drawdown")
    pass

print("Avg. Return: ", a.AvrgRet)
print(a.describe)
# print("Sharpe Ratio: ", a.SharpeRatio())
print("Sortino Ratio: ", a.SortinoRatio())