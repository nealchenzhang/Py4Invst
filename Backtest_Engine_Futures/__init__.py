#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Tue Dec 26 08:58:19 2017

# @author: NealChenZhang

# This program is personal trading platform designed when employed in
# Aihui Asset Management as a quantitative analyst.
#
# Contact:
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################

"""
Backtest_Engine_Futures includes:

    backtest module -- backtest class
    data module -- datahandler class
    event module -- Market Event / Signal Event / Order Event / Fill Event
    execution module -- SimulatedExecutionHandler
    performance module -- Performance calculation TODO: import Analyzer module
    portfolio module -- The core module of backtest
    strategy module -- strategy class
"""

from .backtest import *
from .data import *
from .event import *
from .execution import *
from .performance import *
from .portfolio import *
from .strategy import *

__author__ = 'Neal Chen Zhang'