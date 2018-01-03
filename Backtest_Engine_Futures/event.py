# -*- coding: utf-8 -*-

# event.py


class Event(object):
    """
    Event is base class providing an interface for all subsequent 
    (inherited) events, that will trigger further events in the 
    trading infrastructure.   
    """
    pass


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with 
    corresponding bars.
    """

    def __init__(self):
        """
        Initialises the MarketEvent.
        """
        self.type = 'MARKET'


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    
    def __init__(self, strategy_id, symbol, datetime,
                 signal_type, position_type, strength):
        """
        Initialises the SignalEvent.

        Parameters:
        strategy_id - The unique ID of the strategy sending the signal.
        symbol - The ticker symbol, e.g. 'rb1801'.
        datetime - The timestamp at which the signal was generated.
        TODO: signal_type - 'BUY' or 'SELL' or 'EXIT' Optimized this process
        strength - An adjustment factor "suggestion" used to scale
            quantity at the portfolio level. Useful for pairs strategies.
        """
        self.strategy_id = strategy_id
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. 'rb1801'), a type (market or limit),
    quantity ,direction, and position_type.
    """

    def __init__(self, symbol, order_type, quantity, direction, position_type):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), has
        a quantity (integral), its direction ('BUY' or
        'SELL'), and its position_type ('OPEN', 'CLOSE', or 'CLOSE_T0')

        Parameters:
        symbol - The instrument to trade.
        order_type - 'MKT' or 'LMT' for Market or Limit.
        quantity - Non-negative integer for quantity.
        direction - 'BUY' or 'SELL'.
        position_type - 'OPEN', 'CLOSE' or 'CLOSE_T0'
        """
        self.type = 'ORDER'                             # Order event
        self.symbol = symbol                            # Symbol
        self.order_type = order_type                    # LMT or MKT
        if quantity >= 0 & int(quantity) == quantity:
            self.quantity = quantity                    # Quantity
        else:
            print('Please enter the right quantity')
        self.direction = direction                      # BUY or SELL
        self.position_type = position_type              # OPEN, CLOSE, or CLOSE_T0

    def print_order(self):
        """
        Outputs the values within the Order.
        """
        print(
            "Order: Symbol=%s, Type=%s, Quantity=%s, Direction=%s, Position_type=%s" %
            (self.symbol, self.order_type, self.quantity, self.direction, self.position_type)
        )


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    
    TODO: Currently does not support filling positions at
    different prices. This will be simulated by averaging
    the cost.

    """

    def __init__(self, timeindex, symbol, exchange, quantity, 
                 direction, position_type, fill_price, commission=None):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, price of fill and an optional
        commission.

        If commission is not provided, the Fill object will
        calculate it based on the trade size.

        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL')
        position_type - The position_type of fill ('OPEN', 'CLOSE' or 'CLOSE_T0')
        fill_price - The fill price in dollars.
        commission - An optional commission calculated from brokers.
        """
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.position_type = position_type
        self.fill_price = fill_price

        # Calculate commission
        if commission is None:
            self.commission = self.calculate_commission()
        else:
            self.commission = commission

    def calculate_commission(self):
        """
        Calculates the fees of trading based on different futures/stocks brokers.
        """
        full_cost = 1.3
        if self.quantity <= 500:
            full_cost = max(1.3, 0.013 * self.quantity)
        else: # Greater than 500
            full_cost = max(1.3, 0.008 * self.quantity)
        return full_cost
