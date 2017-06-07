#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Mon Mar  6 10:22:19 2017

# @author: nealcz @Aian_fund

# This program is personal trading platform desiged when employed in 
# Aihui Asset Management as a quantatitive analyst.
# 
# Contact: 
# Name: Chen Zhang (Neal)
# Mobile: (+86) 139-1706-0712
# E-mail: nealzc1991@gmail.com

###############################################################################

"""
This package Analyzers is the last step of backtesting.
This is the class for all analyzers, including:
    Sharpe_Ratio
    Sortino_Ratio
    Omega_Ratio
    Draw_down

"""

from __future__ import print_function

from .Analyzer import Analyzer
from .Sharpe_Ratio import Sharpe_Ratio
from .Sortino_Ratio import Sortino_Ratio
from .Omega_Ratio import Omega_Ratio
from .Draw_Down import Draw_Down

__author__ = 'Neal Chen Zhang'


__all__ = ['Analyzer', 'Sharpe_Ratio', 'Sortino_Ratio', 'Omega_Ratio', \
           'Draw_Down']