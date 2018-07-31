import numpy as np
import pandas as pd

import os
import datetime as dt

print(os.getcwd())
os.chdir(os.getcwd()+'/Fundamental')

tmp = pd.read_excel('DR3M.xlsx')
df_DR3M = pd.DataFrame(columns=['Date', 'DR3M'])
df_DR3M.loc[:,'Date'] = tmp.loc[:, '日期']
df_DR3M.loc[:, 'DR3M'] = tmp.loc[:, '收盘价']
df_DR3M.set_index('Date', inplace=True)

df_RB00 = pd.DataFrame(columns=['Date', 'RB00_p', 'RB00_v', 'RB00_po'])
df_RB01 = pd.DataFrame(columns=['Date', 'RB01_p', 'RB01_v', 'RB01_po'])
df_RB05 = pd.DataFrame(columns=['Date', 'RB05_p', 'RB05_v', 'RB05_po'])
df_RB10 = pd.DataFrame(columns=['Date', 'RB10_p', 'RB10_v', 'RB10_po'])


tmp00 = pd.read_excel('./Data/RB00.xlsx')
tmp01 = pd.read_excel('./Data/RB01.xlsx')
tmp05 = pd.read_excel('./Data/RB05.xlsx')
tmp10 = pd.read_excel('./Data/RB10.xlsx')

tmp00.dropna(inplace=True)
tmp01.dropna(inplace=True)
tmp05.dropna(inplace=True)
tmp10.dropna(inplace=True)

df_RB00.loc[:, 'Date'] = tmp00.loc[:, '日期']
df_RB00.loc[:, 'RB00_p'] = tmp00.loc[:, '收盘价(元)']
df_RB00.loc[:, 'RB00_v'] = tmp00.loc[:, '成交量']
df_RB00.loc[:, 'RB00_po'] = tmp00.loc[:, '持仓量']
df_RB00.set_index('Date', inplace=True)

df_RB01.loc[:, 'Date'] = tmp01.loc[:, '日期']
df_RB01.loc[:, 'RB01_p'] = tmp01.loc[:, '收盘价(元)']
df_RB01.loc[:, 'RB01_v'] = tmp01.loc[:, '成交量']
df_RB01.loc[:, 'RB01_po'] = tmp01.loc[:, '持仓量']
df_RB01.set_index('Date', inplace=True)

df_RB05.loc[:, 'Date'] = tmp05.loc[:, '日期']
df_RB05.loc[:, 'RB05_p'] = tmp05.loc[:, '收盘价(元)']
df_RB05.loc[:, 'RB05_v'] = tmp05.loc[:, '成交量']
df_RB05.loc[:, 'RB05_po'] = tmp05.loc[:, '持仓量']
df_RB05.set_index('Date', inplace=True)

df_RB10.loc[:, 'Date'] = tmp10.loc[:, '日期']
df_RB10.loc[:, 'RB10_p'] = tmp10.loc[:, '收盘价(元)']
df_RB10.loc[:, 'RB10_v'] = tmp10.loc[:, '成交量']
df_RB10.loc[:, 'RB10_po'] = tmp10.loc[:, '持仓量']
df_RB10.set_index('Date', inplace=True)

df_DR3M.to_csv('DR3M.csv')
df_RB00.to_csv('RB00.csv')
df_RB01.to_csv('RB01.csv')
df_RB05.to_csv('RB05.csv')
df_RB10.to_csv('RB10.csv')

df_RB = pd.merge(df_RB01, df_RB05, left_index=True, right_index=True, how='outer')
df_RB = pd.merge(df_RB, df_RB10,left_index=True, right_index=True, how='outer')
df_RB = pd.merge(df_RB, df_RB00,left_index=True, right_index=True, how='outer')
# last row data is wrong drop
df_RB.drop(df_RB.index[[-1]], inplace=True)
df_RB = df_RB.iloc[190:]
df_RB.to_csv('Data.csv')