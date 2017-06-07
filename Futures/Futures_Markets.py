#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
#
# Created on Wed Apr 05 10:27:45 2017

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
This module is used as basic information about Chinese Futures Exchange:
    Including:
        Four Major Exchanges and Futures Contracts
        Futures Trading Calendar
"""

class Futures_Contracts(object):
    """
    This class is used as Futures contracts in Four Major Futures Exchange in
    China.
    
    SHFE: Shanghai Futures Exchange
    DCE: Dalian Commodity Exchange
    CZCE: Zhengzhou Commodity Exchange
    CFFEX: China Financial Futures Exchange
    
    """
    SHFE = ['AL','CU','ZN','RU','FU','AU','RB',
            'WR','PB','AG','BU','HC','NI','SN']
    DCE = ['P','L','M','Y','A','C','B','PVC','J',
           'JM','I','JD','FB','BB','PP','CS']
    CZCE = ['TA','SR','WH','PM','CF','OI','RI',
            'MA','FG','RS','RM','ZC','JR','LR',
            'SF','SM']
    CFFEX = ['IF','IH','IC','T','TF']
    
    Contracts = {i:'SHFE' for i in SHFE}
    Contracts.update({i:'DCE' for i in DCE})
    Contracts.update({i:'CZCE' for i in CZCE})
    Contracts.update({i:'CFFEX' for i in CFFEX})
    
    ####未来更新class类属性，找出主力合约与次主力合约

class Futures_Trading_Calendar(object):
    """
    Check two contracts trading dates
    """