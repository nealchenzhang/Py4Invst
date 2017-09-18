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

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import warnings
import pylab as pl

pl.mpl.rcParams['font.sans-serif'] = ['SimHei']
pl.mpl.rcParams['axes.unicode_minus'] = False

