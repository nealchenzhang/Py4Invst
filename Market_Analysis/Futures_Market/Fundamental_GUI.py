# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Mon Sep 18 16:03:45 2017

# @author: NealChenZhang

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################

__author__ = 'NealChenZhang'

import sys

import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import mpl_finance as mpf
import pylab as pl

pl.mpl.rcParams['font.sans-serif'] = ['SimHei']
pl.mpl.rcParams['axes.unicode_minus'] = False

import pandas as pd
import datetime as dt

from Data.Stocks_Data import MongoDB

class MyMplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, *args, **kwargs):

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.initial_figure()

        super(MyMplCanvas, self).__init__(fig, *args, **kwargs)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def initial_figure(self):
        pass

class MyCandleStick(MyMplCanvas):
    tick = '600614'
    dbname = 'Stocks_Data'
    start_date = '2017-01-03'
    end_date = '2017-09-26'

    def __init__(self, *args, **kwargs):
        super(MyCandleStick, self).__init__(*args, **kwargs)

    def initial_figure(self):

        DB = MongoDB.MongoDBData()
        df = DB.datafromMongoDB(self.dbname, self.tick)
        df = df.loc[self.start_date:self.end_date, ['open', 'high', 'low', 'close']]

        Data_list = []
        for date, row in df.iterrows():
            Date = date2num(dt.datetime.strptime(date, "%Y-%m-%d"))
            Open, High, Low, Close = row[:4]
            Data = (Date, Open, High, Low, Close)
            Data_list.append(Data)

        self.axes.xaxis_date()
        # self.axes.plot(df['open'], df['close'])
        # fig, ax = plt.subplots()
        # fig.subplots_adjust(bottom=0.2)
        # 设置X轴刻度为日期时间
        # ax.xaxis_date()
        # plt.xticks(rotation=45)
        # plt.yticks()
        # plt.title("股票代码：601558两年K线图")
        # plt.xlabel("时间")
        # plt.ylabel("股价（元）")
        plt.tight_layout()
        mpf.candlestick_ohlc(self.axes, Data_list, width=1.5, colorup='r', colordown='green')
        # plt.grid()


class MyMainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MyMainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Plot")

        self.main_widget = QWidget(self)

        gridlayout = QGridLayout(self.main_widget)
        gridlayout.addWidget(MyCandleStick(self.main_widget, width=5, height=4, dpi=100))
        gridlayout.addWidget(QLabel("Hello"))

        self.setCentralWidget(self.main_widget)


app = QApplication(sys.argv)

window = MyMainWindow()
window.show() # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec_()
