# -*- coding: utf-8 -*-

# data.py

from abc import ABCMeta, abstractmethod
import datetime
import os

import numpy as np
import pandas as pd

from Data.Futures_Data.MongoDB_Futures import tickData

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

        self._retrieve_mongodb_data()

    def _retrieve_mongodb_data(self):
        """
        Retrieves the mongodb from the DB, converting them into pandas
        DataFrames within a symbol dictionary.

        For this handler it will be assumed that the database structure
        is designed as Data Directory.

        :return:
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the data from symbol database for specific time period,
            # indexed on datetime
            self.symbol_data[s] = tickData(self.dbname, s).df_1min_fromMongoDB(self.dbname, s)

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].\
                reindex(index=comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        :param symbol:
        :return:
        """
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        :param symbol:
        :return:
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        :param symbol:
        :return:
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        :param symbol:
        :return:
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        :param symbol:
        :param val_type:
        :return:
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        :param symbol:
        :param val_type:
        :param N:
        :return:
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list)

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())