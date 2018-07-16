import numpy as np
import pandas as pd

import os

print(os.getcwd())
os.chdir(os.getcwd()+'/Market_Analysis/Futures_Market/Fundamental')

tmp = pd.read_excel('DR3M.xlsx')
df_DR3M = pd.DataFrame(columns=['Date', 'DR3M'])
df_DR3M.loc[:,'Date'] = tmp.loc[:, '日期']
df_DR3M.loc[:, 'DR3M'] = tmp.loc[:, '收盘价']
df_DR3M.set_index('Date', inplace=True)

df_RB = pd.DataFrame(columns=['Date', 'RB00_p', 'RB01_p', 'RB05_p', 'RB10_p',
                              'RB00_v', 'RB01_v', 'RB05_v', 'RB10_v',
                              'RB00_po', 'RB01_po', 'RB05_po', 'RB10_po',])
for file in os.listdir('./RB'):
    tmp = pd.read_excel(file)
    str_price = file.split('.')[0] + '_p'
    str_v = file.split('.')[0] + '_v'
    str_po = file.split('.')[0] + '_po'
    df_RB.loc[:, str_price] = pd.read_excel(file)