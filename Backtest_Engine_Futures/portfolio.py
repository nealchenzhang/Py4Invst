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

        self.columns_pos_details = [
            'open_time', 'symbol', 'exchange', 'quantity', 'direction',
            'position_type', 'open_price', 'UPL', 'holdingPnL'
        ]
        self.dict_columns_pos_details = {
            'open_time': np.str, 'symbol': np.str, 'exchange': np.str,
            'quantity': np.int64, 'direction': np.str, 'position_type': np.str,
            'open_price': np.float64, 'UPL': np.float64, 'holdingPnL': np.float64
        }
        self.columns_positions = [
            'datetime', 'symbol', 'direction', 'total_quantity', 'YstPosition',
            'T0Position', 'Position_avlbl', 'Open_price_avg', 'UPL',
            'Holding_price_avg', 'holdingPnL', 'margin_Occupied'
        ]
        self.dict_columns_positions = {
            'datetime': np.datetime64, 'symbol': np.str, 'direction': np.str,
            'total_quantity': np.int64, 'YstPosition': np.int64,
            'T0Position': np.int64, 'Position_avlbl': np.int64,
            'Open_price_avg': np.float64, 'UPL': np.float64,
            'Holding_price_avg': np.float64, 'holdingPnL': np.float64,
            'margin_Occupied': np.float64
        }
        self.columns_holdings = [
            'preBalance', 'Realized_PnL', 'MTM_PnL', 'Holding_PnL', 'Margin',
            'Commission', 'Fund_avlbl', 'Frozen_fund', 'Risk Degree', 'Balance'
        ]
        self.dict_columns_holdings = {
            'preBalance': np.float64, 'Realized_PnL': np.float64, 'MTM_PnL': np.float64,
            'Holding_PnL': np.float64, 'Margin': np.float64, 'Commission': np.float64,
            'Fund_avlbl': np.float64, 'Frozen_Fund': np.float64, 'Risk Degree': np.float64,
            'Balance': np.float64
        }

        self.df_all_positions = self.construct_all_positions()
        self.df_current_positions = self.construct_current_positions()

        self.df_all_holdings = self.construct_all_holdings()
        self.df_current_holdings = self.construct_current_holdings()

    def construct_current_positions(self):
        """
        Constructs the current position DataFrame which holds current
        positions

            open_time:       The fill_time for this position
            symbol:          The symbol contract
            exchange:        The exchange where the symbol trades
            quantity:        The position quantity
            direction:       "BUY", or "SELL"
            position_type:   "Yst_Pos" or "T0_Pos"
            open_price:      The filled price
            UPL:             The unrealized P/L based on open_price
            holdingPnL:      The unrealized P/L based on Ystday settlement price

        """
        columns = self.columns_pos_details
        df_current_positions = pd.DataFrame(columns=columns)
        return df_current_positions

    def construct_all_positions(self):
        """
        Constructs the positions DataFrame using the start_date
        to determine when the time index will begin.

            datetime:            The positions time
            symbol:              The symbol contract
            direction:           "BUY" or "SELL"
            total_quantity:      Total quantity for symbol
            YstPosition:         Yesterday quantity
            T0Position:          Today quantity
            Position_avlbl:      Quantity available to trade
            Open_price_avg:      Average open price
            UPL:                 Unrealized profit and loss based on avg.
                                 open price
            Holding_price_avg:   Average holding price
            holdingPnL:          Unrealized profit and loss based on avg.
                                 holding price
            margin_Occupied:     Margin occupied

        """
        columns = self.columns_positions
        df_all_positions = pd.DataFrame(columns=columns)
        return df_all_positions

    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.

            preBalance:          The previous Balance
            Realized_PnL:        The realized profit and loss
            MTM_PnL:             The mark to market profit and loss
                                 calculated based on previous settlement
                                 price
            Holding_PnL:         The holding positions profit and loss
                                 calculated based on current price and
                                 Holding_price_avg
            Margin:              The total margin of the portfolio
            Commission:          The commission paid for this trading day
            Fund_avlbl:          Total balance except margin occupied and
                                 frozen fund.
            Frozen_fund:         Fund frozen for order, including commission
                                 and margin required.
            Risk Degree:         Calculated by Margin / Balance
            Balance:             The Balance or equity for now, after
                                 M2M settlement, this will be next trading day
                                 preBalance
            datetime:            The holdings for particular datetime

        """
        columns = self.columns_holdings.append('datetime')
        df_all_holdings = pd.DataFrame(columns=columns)
        return df_all_holdings

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.

            preBalance:          The previous Balance
            Realized_PnL:        The realized profit and loss
            MTM_PnL:             The mark to market profit and loss
                                 calculated based on previous settlement
                                 price
            Holding_PnL:         The holding positions profit and loss
                                 calculated based on current price and
                                 Holding_price_avg
            Margin:              The total margin of the portfolio
            Commission:          The commission paid for this trading day
            Fund_avlbl:          Total balance except margin occupied and
                                 frozen fund.
            Frozen_fund:         Fund frozen for order, including commission
                                 and margin required.
            Risk Degree:         Calculated by Margin / Balance
            Balance:             The Balance or equity for now, after
                                 M2M settlement, this will be next trading day
                                 preBalance

        """
        columns = self.columns_holdings
        df_current_holdings = pd.DataFrame(columns=columns)
        return df_current_holdings

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current 
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCVOI).

        Makes use of a MarketEvent from the events queue.
        """
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])

        # # Update positions
        # # ================
        # TODO: 持仓类型要随着时间进行调整
        # dp = dict((k, v) for k, v in [(s, d_pos) for s in self.symbol_list])
        # dp['datetime'] = latest_datetime
        #
        # for s in self.symbol_list:
        #     dp[s] = self.current_positions[s]
        #
        # # Append the current positions
        # self.all_positions.append(dp)
        #
        # # Update holdings
        # # ===============
        # dh = dict((k, v) for k, v in [(s, self.current_holdings[s]) for s in self.symbol_list])
        # dh['datetime'] = latest_datetime
        # dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        # # ToDO# dh['datetime'] = latest_datetime
        # # dh['equity_T-1'] = self.current_holdings['equity_T-1']
        # # dh['long_quantity'] = self.current_holdings['long_quantity']
        # # dh['short_quantity'] = self.current_holdings['short_quantity']
        # # dh['long_margin'] = self.current_holdings['long_margin']
        # # dh['short_margin'] = self.current_holdings['short_margin']
        # # dh['commission'] = self.current_holdings['commission']
        # # dh['equity_T0'] = self.current_holdings['equity_T-1']
        #
        # # TODO: self.d_pos
        # for s in self.symbol_list:
        #     dh['preBalance'] = 0 # TODO:self.current_holdings[s]['price_settlement']
        #     # TODO: check the margin for intraday/interday trading
        #     dh['Total_margin'] += self.current_holdings[s]['margin']
        #     dh['Fund_avail'] += dh[s]['UPL']
        #     dh['Commission'] += self.current_holdings[s]['commission']
        #     dh['Balance'] = dh['Fund_avail'] + dh['Total_margin']
        #
        # # Append the current holdings
        # self.all_holdings.append(dh)

    # ======================
    # FILL/POSITION HANDLING
    # ======================
#########################################################################
    def update_positions_from_fill(self, fill):
        """
        Takes a Fill object and updates the position dataframe to
        reflect the new position.

        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Todo: fundamental information contract size and etc.
        multiplier = 10
        latest_close_price = self.bars.get_latest_bar_value(fill.symbol, "Close")
        # TODO: last settlement price
        last_settlement_price = self.bars.get_latest_bar_value(fill.symbol, "lastSettlement")

        direction = 0
        if fill.direction == 'BUY':
            direction = 1
        if fill.direction == 'SELL':
            direction = -1

        if fill.position_type == 'OPEN':
            fill_array = np.array(
                [fill.timeindex.strftime('%Y%m%d'), fill.symbol, fill.exchange,
                 fill.quantity, fill.direction, 'T0_Pos', fill.fill_price,
                 direction * (latest_close_price - fill.fill_price) * multiplier,
                 direction * (latest_close_price - fill.fill_price) * multiplier
                 ]
            ).reshape(1, len(self.columns_pos_details))
            df = pd.DataFrame(fill_array, columns=self.columns_pos_details)
            df = df.astype(self.dict_columns_pos_details)
            self.df_current_positions = self.df_current_positions.append(df, ignore_index=True)
            self.df_current_positions.reset_index(drop=True, inplace=True)

        if fill.position_type == 'CLOSE_T0':
            quantity_to_fill = fill.quantity
            if fill.direction == 'BUY':
                df = self.df_current_positions.where(
                    self.df_current_positions['position_type'] == 'T0_Pos' &
                    self.df_current_positions['symbol'] == fill.symbol &
                    self.df_current_positions['direction'] == 'SELL'
                ).dropna()
            if fill.direction == 'SELL':
                df = self.df_current_positions.where(
                    self.df_current_positions['position_type'] == 'T0_Pos' &
                    self.df_current_positions['symbol'] == fill.symbol &
                    self.df_current_positions['direction'] == 'BUY'
                ).dropna()
            df_index = df.index.tolist()

            if len(df_index) == 0:
                print("No available T0 position to close. Try Close.")

            for ix in df_index:
                if self.df_current_positions.loc[df_index[ix], 'quantity'] >= quantity_to_fill:
                    self.df_current_positions.loc[df_index[ix], 'quantity'] -= quantity_to_fill
                    # ToDO: 平仓盈亏计入方法确定 待思考
                    self.df_current_holdings['Realized_PnL'] += \
                        direction * (latest_close_price - fill.fill_price) * multiplier
                    quantity_to_fill -= self.df_current_positions.loc[df_index[ix], 'quantity']
                else:
                    quantity_to_fill -= self.df_current_positions.loc[df_index[ix], 'quantity']
                    # ToDO: 平仓盈亏计入方法确定 待思考
                    self.df_current_holdings['Realized_PnL'] += \
                        direction * (latest_close_price - fill.fill_price) * multiplier

                if quantity_to_fill <= 0:
                    break

            for ix in df_index:
                if self.df_current_positions.loc[df_index[ix], 'quantity'] == 0:
                    self.df_current_positions.drop(df_index[ix], inplace=True)

            self.df_current_positions.reset_index(drop=True, inplace=True)

        if fill.position_type == 'CLOSE':
            quantity_to_fill = fill.quantity
            if fill.direction == 'BUY':
                df = self.df_current_positions.where(
                    self.df_current_positions['symbol'] == fill.symbol &
                    self.df_current_positions['direction'] == 'SELL'
                ).dropna()
            if fill.direction == 'SELL':
                df = self.df_current_positions.where(
                    self.df_current_positions['symbol'] == fill.symbol &
                    self.df_current_positions['direction'] == 'BUY'
                ).dropna()
            df_index = df.index.tolist()

            if len(df_index) == 0:
                print("No available position to close.")

            for ix in df_index:
                if self.df_current_positions.loc[df_index[ix], 'quantity'] >= quantity_to_fill:
                    self.df_current_positions.loc[df_index[ix], 'quantity'] -= quantity_to_fill
                    # ToDO: 平仓盈亏计入方法确定 待思考
                    if self.df_current_positions.loc[df_index[ix], 'position_type'] == 'T0_Pos':
                        self.df_current_holdings['Realized_PnL'] += \
                            direction * (
                                    self.df_current_positions.loc[df_index[ix], 'open_price']
                                    - fill.fill_price
                            ) * multiplier
                    else:
                        self.df_current_holdings['Realized_PnL'] += \
                            direction * (last_settlement_price - fill.fill_price) * multiplier
                    quantity_to_fill -= self.df_current_positions.loc[df_index[ix], 'quantity']
                else:
                    quantity_to_fill -= self.df_current_positions.loc[df_index[ix], 'quantity']
                    # ToDO: 平仓盈亏计入方法确定 待思考
                    if self.df_current_positions.loc[df_index[ix], 'position_type'] == 'T0_Pos':
                        self.df_current_holdings['Realized_PnL'] += \
                            direction * (
                                    self.df_current_positions.loc[df_index[ix], 'open_price']
                                    - fill.fill_price
                            ) * multiplier
                    else:
                        self.df_current_holdings['Realized_PnL'] += \
                            direction * (last_settlement_price - fill.fill_price) * multiplier

                if quantity_to_fill <= 0:
                    break

            for ix in df_index:
                if self.df_current_positions.loc[df_index[ix], 'quantity'] == 0:
                    self.df_current_positions.drop(df_index[ix], inplace=True)

            self.df_current_positions.reset_index(drop=True, inplace=True)

    ########################################################################
    def update_holdings_from_fill(self, fill):
        """
        Takes a Fill object and updates the holdings matrix to
        reflect the holdings value.

        Parameters:
        fill - The Fill object to update the holdings with.
        """
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])
        # Check whether the fill is a long or short
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
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

        # df_current_holdings = pd.DataFrame(np.array(['MA805', 'BUY', 1, 1, 0, 1, 2825, 850, 3773, 70]).reshape(1, 10),
                                           # columns=columns_holdings)

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
        # TODO: cur_quantity = self.current_positions[symbol]['quantity']
        # df = self.df_current_positions.set_index(['symbol','position_type'])
        # df.loc[symbol]
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
        curve = self.all_holdings
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['Balance'].pct_change()
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
