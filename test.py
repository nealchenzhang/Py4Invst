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

# from Market_Analysis import Futures_Market
#
# #tmp = Futures_Market.SNR("ru")
# #
# #print(tmp.get_Asset_price())
# #print(tmp.rolling_SNR())
#
#
# from Market_Analysis import Futures_Market
# tmp = Futures_Market.AMH.MDI(markets=["ru","l"])
# print(tmp.Market_Divergence_Index())
# print(tmp.markets)
# print(tmp.get_df_Markets_SNRs())

# from Market_Analysis.Market_Analysis_Tools.Regression_Analysis import Regrs_Analysis

# x = R

from Data import MongoDB
stocklist = ['600614', '601118']
DB = MongoDB.MongoDBData()
dbnames = ['Stocks_Data', 'Futures_Data']
for stock in stocklist:
    DB.data2MongoDB('Stocks_Data', stock)

df1 = DB.datafromMongoDB('Stocks_Data', '600614')
