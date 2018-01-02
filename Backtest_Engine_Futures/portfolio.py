# -*- coding: utf-8 -*-

# portfolio.py

import datetime
import queue
import os

import numpy as np
import pandas as pd

from Backtest_Engine_Futures.event import FillEvent, OrderEvent
from Backtest_Engine_Futures.performance import create_sharpe_ratio, create_drawdowns


class Portfolio(object):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, 5-min, 30-min, 60 min or EOD.

    The positions DataFrame stores a time-index of the 
    quantity of positions held. 

    The holdings DataFrame stores the cash and total market
    holdings value of each symbol for a particular 
    time-index, as well as the percentage change in 
    portfolio total across bars.
    """

    def __init__(self, bars, events, start_date, initial_capital=100000.0):
        """
        Initialises the portfolio with bars and an event queue. 
        Also includes a starting datetime index and initial capital .

        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital

        self.d_pos_0 = self.initial_symbol_position()

        self.all_positions = self.construct_all_positions()
        self.current_positions = dict((k, v) for k, v in [(s, self.d_pos_0) for s in self.symbol_list])

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def initial_symbol_position(self):
        """
        d_pos: positions for particular direction
                Symbol_positions for particular symbol,
                including quantity, position_T0,
                position_T-1, position_avlbl, price_open,
                price_settle, P/L, UPL, and margin.
        d_p:  'LONG' or 'SHORT' positions details, and margin details.

        """
        d_pos = dict()
        d_pos['quantity'] = 0         # Number of position
        d_pos['position_TO'] = 0      # Today OPEN
        d_pos['position_T-1'] = 0     # Yesterday OPEN
        d_pos['position_avlbl'] = 0   # Position available to trade
        d_pos['price_open'] = 0.0     # Open price
        d_pos['price_pos'] = 0.0      # Avg. position price held
        d_pos['P/L'] = 0.0            # Realized P/L
        d_pos['UPL'] = 0.0            # MTM P/L
        d_pos['margin'] = 0.0         # Initial Margin TODO: will change after settlement
        d_p = dict()
        d_p['LONG'] = [d_pos]         # LONG
        d_p['SHORT'] = [d_pos]        # SHORT
        d_p['Margin'] = 0.0           # Total margin for this symbol
        return d_p

    def construct_all_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = dict((k, v) for k, v in [(s, self.d_pos_0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = dict((k, v) for k, v in [(s, self.d_pos_0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        d['preBalance'] = self.initial_capital
        d['Total_margin'] = 0.0
        d['Fund_occpd'] = 0.0
        d['Fund_avail'] = self.initial_capital
        d['Commission'] = 0.0
        d['Risk_degree'] = 0.0
        d['Balance'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        dh_pos = self.d_pos_0
        d = dict((k, v) for k, v in [(s, dh_pos) for s in self.symbol_list])
        d['datetime'] = self.start_date
        d['preBalance'] = self.initial_capital
        d['Total_margin'] = 0.0
        d['Fund_occpd'] = 0.0
        d['Fund_avail'] = self.initial_capital
        d['Commission'] = 0.0
        d['Risk_degree'] = 0.0
        d['Balance'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current 
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCVOI).

        Makes use of a MarketEvent from the events queue.
        """
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])

        # Symbol positions
        d_pos = self.d_pos_0

        # Update positions
        # ================
        dp = dict((k, v) for k, v in [(s, d_pos) for s in self.symbol_list])
        dp['datetime'] = latest_datetime

        for s in self.symbol_list:
            dp[s] = self.current_positions[s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = dict((k, v) for k, v in [(s, self.current_holdings[s]) for s in self.symbol_list])
        dh['datetime'] = latest_datetime
        dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        # ToDO# dh['datetime'] = latest_datetime
        # dh['equity_T-1'] = self.current_holdings['equity_T-1']
        # dh['long_quantity'] = self.current_holdings['long_quantity']
        # dh['short_quantity'] = self.current_holdings['short_quantity']
        # dh['long_margin'] = self.current_holdings['long_margin']
        # dh['short_margin'] = self.current_holdings['short_margin']
        # dh['commission'] = self.current_holdings['commission']
        # dh['equity_T0'] = self.current_holdings['equity_T-1']

        # TODO: self.d_pos
        for s in self.symbol_list:
            dh['preBalance'] = 0 # TODO:self.current_holdings[s]['price_settlement']
            # TODO: check the margin for intraday/interday trading
            dh['Total_margin'] += self.current_holdings[s]['margin']
            dh['Fund_avail'] += dh[s]['UPL']
            dh['Commission'] += self.current_holdings[s]['commission']
            dh['Balance'] = dh['Fund_avail'] + dh['Total_margin']

        # Append the current holdings
        self.all_holdings.append(dh)

    # ======================
    # FILL/POSITION HANDLING
    # ======================
#########################################################################
    def update_positions_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to
        reflect the new position.

        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Update positions list with Fill object
        multiplier = 10
        fill_pos = ''
        if fill.direction == 'BUY' & fill.position_type == 'OPEN':
            fill_pos = 'LONG'
        if fill.direction == 'SELL'& fill.position_type == 'OPEN':
            fill_pos = 'SHORT'

        if fill.direction == 'BUY' & fill.position_type == 'CLOSE':
            fill_pos = 'SHORT'
        if fill.direction == 'SELL'& fill.position_type == 'CLOSE':
            fill_pos = 'LONG'

        if fill.position_type == 'OPEN':
            self.current_holdings[fill.symbol][fill_pos]['quantity'] += fill.quantity
            self.current_holdings[fill.symbol][fill_pos]['position_T0'] += fill.quantity
            self.current_holdings[fill.symbol][fill_pos]['open_price'] = np.average(
                [
                    self.current_holdings[fill.symbol][fill_pos]['open_price'], fill.fill_price],
                weights=[
                    self.current_holdings[fill.symbol][fill_pos]['quantity'], fill.quantity
                ]
            )
        # TODO: track every fill and calculate different position price and close pnl

        if fill.position_type == 'CLOSE':
            # self.current_holdings[fill.symbol]['position_T0'] -= fill.quantity
            self.current_holdings[fill.symbol][fill_pos]['quantity'] -= fill.quantity
            self.current_holdings[fill.symbol][fill_pos]['P/L'] += (
                    self.bars.get_latest_bar_value(fill.symbol, "Close") -
                    fill.fill_price) * fill.quantity * multiplier

    ########################################################################
    def update_holdings_from_fill(self, fill):
        """
        Takes a Fill object and updates the holdings matrix to
        reflect the holdings value.

        Parameters:
        fill - The Fill object to update the holdings with.
        """
        # Check whether the fill is a long or short
        fill_dir = 0
        if fill.direction == 'LONG':
            fill_dir = 1
        if fill.direction == 'SHORT':
            fill_dir = -1

        fill_type = 0
        if fill.position_type == 'OPEN':
            fill_type = 1
        if fill.position_type == 'CLOSE':
            fill_type = -1
        if fill.position_type == 'CLOSE_T0':
            fill_type = -1

        fill_dir = 0
        if self.current_holdings[s]['direction'] == 'LONG':
            fill_dir = 1
        if self.current_holdings[s]['direction'] == 'SHORT':
            fill_dir = -1

        # ToDO: Multiplier
        multiplier = 10

        # Calculate UPL based on open_price
        dh[s]['UPL'] = (self.bars.get_latest_bar_value(s, "Close") -
                        self.current_positions[s]['price_open']) * \
                       dir * multiplier * self.current_positions[s]['quantity']

        # Update holdings list with new quantities
        fill_price = self.bars.get_latest_bar_value(
            fill.symbol, "Close"
        )
        # self.current_holdings[fill.symbol]['price_open']
        # TODO: 合约基本信息
        multiplier = 10
        equity = fill_price * fill.quantity * multiplier
        long_margin = equity * fill.margin_rate
        short_margin = equity * fill.margin_rate

        self.current_holdings[fill.symbol]['margin'] += np.max(long_margin, short_margin)
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= np.max(long_margin, short_margin)
        self.current_holdings['total'] -= (cost + fill.commission)

    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)

    def generate_naive_order(self, signal):
        """
        Simply files an Order object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.

        Parameters:
        signal - The tuple containing Signal information.
        """
        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength

        mkt_quantity = 1
        cur_quantity = self.current_positions[symbol]['quantity']
        order_type = 'MKT'

        if direction == 'BUY' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY', 'OPEN')
        if direction == 'SELL' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL', 'OPEN')
    
        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL', 'CLOSE')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY', 'CLOSE')
        return order

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)

    # ========================
    # POST-BACKTEST STATISTICS
    # ========================

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0+curve['returns']).cumprod()
        self.equity_curve = curve

    def output_summary_stats(self):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns, periods=252*60*6.5)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        self.equity_curve.to_csv('equity.csv')
        print(os.getcwd())
        return stats
