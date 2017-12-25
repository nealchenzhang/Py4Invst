import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
# Candlestick plot
try:
    from mpl_finance import candlestick_ohlc
except:
    from matplotlib.finance import candlestick_ohlc

from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY

import matplotlib
matplotlib.style.use('ggplot')
from pandas.plotting import scatter_matrix

import urllib.parse
import webbrowser


import os
os.chdir('/home/nealzhangchen/Documents/Python/Py4Invst/Market_Analysis/Stocks_Market')

# # Technical Indicators
# #########################################################
# #
# # Moving Average: MA20 MA50 MA200
# df_Stock1['MA20'] = df_Stock1['Close'].rolling(window=20).mean()
# df_Stock1[['Close', 'MA20']].plot(label='Tesla', title='Moving Average', figsize=(12,8));


class Stock_Analysis(object):
    def __init__(self, stock_name, *args, **kwargs):
        self.name = stock_name

    def get_prices_data(self, start_date='2012-12-01', end_date='2012-12-31'):
        df = pd.read_csv(self.name+"_Stock.csv", index_col='Date', parse_dates=True)
        return df[start_date:end_date]

    def get_returns_data(self, start_date='2012-12-01', end_date='2012-12-31'):
        df = self.get_prices_data(start_date, end_date)
        df['returns'] =(df['Close'] / df['Close'].shift(1)) - 1
        return df[start_date:end_date]

    def returns_plot(self, start_date='2012-01-01', end_date='2012-12-31'):
        df = self.get_returns_data(start_date, end_date)
        df.reset_index(inplace=True)

        fig = plt.figure(figsize=(12,8))
        fig.suptitle('{}: Returns plot during {} and {}'.format(self.name,
                                                                start_date,
                                                                end_date),
                     fontsize=16)
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        df['returns'].plot.hist(bins=50, color='b', ax=ax1, alpha=0.5)
        df['returns'].plot.kde(ax=ax1)
        ax1.set_title('Histogram')

        df['returns'].plot.box(colormap='jet', ax=ax2)
        ax2.set_title('Box Plot')

        ax3.scatter(x=df.index, y=df['returns'], alpha=0.5)
        ax3.set_title('Scatter Plot')

        # fig.savefig("{}_returns_{}_{}.png".format(self.name, start_date, end_date))
        plt.show()

    def k_volume_plot(self, start_date='2012-01-01', end_date='2012-12-31'):
        df = self.get_prices_data(start_date, end_date)
        df.reset_index(inplace=True)
        df['date_ax'] = df['Date'].apply(lambda date: date2num(date))
        df['Date'] = df['Date'].apply(pd.to_datetime)
        df.set_index('Date', inplace=True)
        candle_values = [tuple(vals) for vals in df[['date_ax', 'Open', 'High', 'Low', 'Close']].values]

        mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
        alldays = DayLocator()  # minor ticks on the days
        weekFormatter = DateFormatter('%y %b %d')  # e.g., 12 Jan 12
        dayFormatter = DateFormatter('%d')  # e.g., 12

        fig, axes = plt.subplots(2,1, figsize=(12,12), sharex=True)
        # fig.subplots_adjust(hspace=0.2)
        fig.suptitle(self.name, fontsize=16)

        # Plot candlestick
        axes[0].xaxis.set_major_locator(mondays)
        axes[0].xaxis.set_minor_locator(alldays)
        axes[0].xaxis.set_major_formatter(weekFormatter)
        candlestick_ohlc(axes[0], candle_values, width=0.6, colorup='r', colordown='g');
        axes[0].set_title('Candlestick')

        # Plot volume
        axes[1].bar(df['date_ax'], df['Volume'])
        axes[1].set_title('Volume')

        for label in axes[1].get_xticklabels():
            label.set_rotation(45)

        # fig.savefig("{}_K_{}_{}.png".format(self.name, start_date, end_date))
        plt.show()

    def volume_news(self, start_date='2012-12-01', end_date='2012-12-31'):
        df = self.get_prices_data(start_date, end_date)
        most_volume_dates = df.sort_values('Volume', ascending=False)['Volume'][:10]
        print(most_volume_dates)
        print('Find out what happened on most volume day.')
        str_keyword = datetime.datetime.strftime(most_volume_dates.argmax(), '%Y-%m-%d')+\
                      ' ' + self.name

        sina_keyword = urllib.parse.quote(str_keyword, encoding='gb2312')
        baidu_keyword = urllib.parse.quote(str_keyword, encoding='utf-8')
        finance_sina_url = 'http://search.sina.com.cn/?q={}&range=all&c=news&sort=time'.format(sina_keyword)
        baidu_url = 'https://www.baidu.com/s?wd={}&ie=UTF-8'.format(baidu_keyword)
        bing_url = 'https://www.bing.com/search?q={}'.format(str_keyword.replace(' ', '+'))

        web_lists = [finance_sina_url, baidu_url, bing_url]

        for web in web_lists:
            webbrowser.open(web)


class Industry_Analysis(object):
    def __init__(self, industry_list):
        self.industry_list = industry_list

    def analysis(self, start_date='2012-01-01', end_date='2012-12-31'):
        industry_list = self.industry_list
        for stock in industry_list:
            single_analysis = Stock_Analysis(stock)
            single_analysis.returns_plot(start_date, end_date)
            single_analysis.k_volume_plot(start_date, end_date)

    def plot_analysis(self, start_date='2012-01-01', end_date='2012-12-31'):
        matplotlib.style.use('classic')
        industry_list = self.industry_list
        df_returns = pd.DataFrame()
        col_list = []
        for stock in industry_list:
            single_analysis = Stock_Analysis(stock)
            df_returns = pd.concat([df_returns,
                                    single_analysis.get_returns_data(start_date, end_date)['returns']],
                                   axis=1)
            col_list.append(single_analysis.name)
        df_returns.columns = col_list

        # Plot scatter_matrix
        scatter_matrix(df_returns, figsize=(12, 8),
                       alpha=0.5, hist_kwds={'bins': 30})
        fig = plt.gcf()
        fig.suptitle('Industry Analysis for \n{}'.format(" ".join(str(i) for i in industry_list)))

        plt.show()





if __name__ == '__main__':
    # stock_list = ['Ford', 'GM', 'Tesla']
    # a = Industry_Analysis(stock_list)
    # a.plot_analysis()
    a = Stock_Analysis('GM')
    a.volume_news()
    # a.returns_plot()
    # a.k_volume_plot()


