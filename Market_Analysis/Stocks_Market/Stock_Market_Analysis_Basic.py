import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import os
os.chdir('/home/nealzc/Documents/Python/Py4Invst/Market_Analysis/Stocks_Market')

# Part 1: Getting the Data
df_Stock1 = pd.read_csv("Tesla_Stock.csv", index_col='Date', parse_dates=True)

start_date = dt.datetime(2012,1,1)
end_date = dt.datetime(2012,12,31)

df_Stock1_Analysis = df_Stock1.loc[start_date: end_date]


# Part 2: Visualizing the Data
df_Stock1['Open'].plot(label='Tesla', title='Open', figsize=(12,8))
plt.legend()

df_Stock1['Volume'].plot(label='Tesla', title='Volume', figsize=(12,8))
plt.legend()

# Most Traded Date
most_traded_Date = df_Stock1.sort_values('Volume', ascending=False)['Volume'][:5]

# Market Cap visualization
# ['Mkt Cap']

print('Find out what happened on these date.')
# # Check other business website for further information
# import webbrowser
# # Dictionary for news source
# #  {'business week', 'sina finance'}
# webbrowser.open('https://www.bing.com/search?q=2013-05-14+Tesla')


# Technical Indicators
#########################################################
#
# Moving Average: MA20 MA50 MA200
df_Stock1['MA20'] = df_Stock1['Close'].rolling(window=20).mean()
df_Stock1[['Close', 'MA20']].plot(label='Tesla', title='Moving Average', figsize=(12,8));


from pandas.plotting import scatter_matrix
industry_comp = pd.concat([tesla['Close'], gm['Close']], axis=1)
scatter_matrix(industry_comp, figsize=(8,8), alpha=0.2, hist_kwds={'bins': 50});



# Candlestick plot
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY

# Rest the index to get a column of January Dates
ford_reset = ford.loc['2012-01':'2012-01'].reset_index()

# Create a new column of numerical "date" values for matplotlib to use
ford_reset['date_ax'] = ford_reset['Date'].apply(lambda date: date2num(date))
ford_values = [tuple(vals) for vals in ford_reset[['date_ax', 'Open', 'High', 'Low', 'Close']].values]

mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12

#Plot it
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)

candlestick_ohlc(ax, ford_values, width=0.6, colorup='g',colordown='r');


class Stock_Analysis(object):
    def __init__(self, stock_name, ticker, *args, **kwargs):
        self.name = stock_name
        # self.ticker = ticker

    def get_data(self, start_date='2012-01-01', end_date='2012-12-31'):
        df = pd.read_csv("Tesla_Stock.csv", index_col='Date', parse_dates=True)
        return df[start_date:end_date]

class Visual_Analysis(Stock_Analysis):
    def __init__(self, stock_list, *args, **kwargs):
        Stock_Analysis.__init__(self)
        self.stock_list = stock_list
        # self.ticker = ticker

    def price_plot(self):
        df = self.get_data(start_date='2012-01-01', end_date='2012-12-31')

        fig, axes = plt.subplots(2,1, figsize=(12,12))
        axes[0].plot(df.index, df['Open'])
        axes[0].set_title('Open Price')
        # axes[0].set_label('Tesla')

        axes[1].plot(df.index, df['Volume'])
        axes[1].set_title('Volume')



