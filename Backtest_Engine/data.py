# -*- coding: utf-8 -*-

# data.py

from abc import ABCMeta, abstractmethod
import datetime as dt
import os

import numpy as np
import pandas as pd

from Backtest_Engine.event import MarketEvent

class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OHLCVOI) for each symbol requested.

    This will replace how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        :param symbol:
        :return:
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars updated.
        :param symbol:
        :param N:
        :return:
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        :param symbol:
        :return:
        """
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        from the last bar.
        :param symbol:
        :param val_type:
        :return:
        """
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        :param symbol:
        :param val_type:
        :return:
        """
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bars to the bars_queue for each symbol
        in a tuple OHLCVOP format: (datetime, Open, High, Low,
        Close, Volume, OpenInterest).
        :return:
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricalMongoDataHandler(DataHandler):
    """
    HistoricalMongoDataHandler is designed to read historical data for
    each requested symbol from MongoDB and provide an interface to obtain
    the "latest" bar in a manner identical to a live trading interface.
    """

    def __init__(self, events, dbname, symbol_list):
        """
        Initializes the historic data handler by requesting the Mongo
        DataBase and a list of symbols.

        It will be assumed that all database name are of the form of
        "1min_CTP", "tick_CTP" or other time period.

        :param events: The Event Queue.
        :param dbname: The Database name for different time period.
        :param symbol_list: A list of symbol strings
        """
        self.events = events
        self.dbname = dbname
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True

        TODO: self._open_convert_csv_files()