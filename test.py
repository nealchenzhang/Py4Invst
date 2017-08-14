# import numpy as np
# import pandas as pd
#
# from Analyzers import Analyzer
# from Analyzers import Draw_Down
# from Analyzers import *
#
# filepath = "D:/Neal/Quant/PythonProject/ValuesFile/values1.csv"
#
# # a = Sharpe_Ratio(filepath, basis="Monthly")
# a = Sortino_Ratio(filepath)
# try:
#     print("MaxDuration: ", a.Maximum_Drawdown())
# except:
#     print("No drawdown")
#     pass
#
# print("Avg. Return: ", a.AvrgRet)
# print(a.describe)
# # print("Sharpe Ratio: ", a.SharpeRatio())
# print("Sortino Ratio: ", a.SortinoRatio())

from Market_Analysis import Futures_Market

#tmp = Futures_Market.SNR("ru")
#
#print(tmp.get_Asset_price())
#print(tmp.rolling_SNR())


from Market_Analysis import Futures_Market
tmp = Futures_Market.AMH.MDI(markets=["ru","l"])
print(tmp.Market_Divergence_Index())
# print(tmp.markets)
# print(tmp.get_df_Markets_SNRs())
