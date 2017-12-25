#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Wed Apr 05 10:18:04 2017

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
This class Fund is used as single fund of the funds portfolio.

"""
print(__doc__)

import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Fund(object):
    def __init__(self, name, weight, basis='Weekly', start_date, end_date):
        self.name = name
        self.basis = {'Daily':252, 'Weekly':52, 'Monthly':12}[basis]
        self.weight = weight
        self.start_date = start_date
        self.end_date = end_date
    
    