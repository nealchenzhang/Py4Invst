#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Thu Apr 06 10:31:48 2017

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

This Module using Markowtiz mean-variance framework to analyze portfolio
    including:
        Mean-Variance Optimization (MVO)
        Black-Letterman Model

"""

from __future__ import division
from __future__ import print_function
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm  # Generally use sm.api
import scipy.stats as scs # Scipy stats

ret = pd.read_csv('D:\\Neal\\Quant\\PythonProject\\Py4Invst\\Data\\ret.csv')

ret = ret.set_index('Unnamed: 0')

((ret.cumsum()+pd.DataFrame(np.ones(ret.shape), columns=ret.columns, index=ret.index))*100).plot(figsize=(8,5))

import tushare as ts
stock_set = ['000413','000063','002007','000001','000002']
df = pd.DataFrame()
for i in stock_set:
    df[i] = ts.get_hist_data(i, start='2015-01-01', end='2015-12-31')['close']

df = df.sort_index(ascending=True)
returns = np.log(df / df.shift(1))
returns.mean()*252
noa = len(stock_set)
weights = np.random.random(noa)
weights /= np.sum(weights)

np.sum(ret.mean()*weights)*252

np.dot(weights.T, np.dot(ret.cov()*252,weights))
np.sqrt(np.dot(weights.T, np.dot(ret.cov()*252,weights)))

port_returns = []

port_variance = []

for p in range(4000):
    weights = np.random.random(noa)
    weights /=np.sum(weights)
    port_returns.append(np.sum(ret.mean()*252*weights))
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(ret.cov()*252, weights))))

port_returns = np.array(port_returns)
port_variance = np.array(port_variance)

    
risk_free = 0.04

plt.figure(figsize = (8,4))
plt.scatter(port_variance, port_returns, c=(port_returns-risk_free)/port_variance, marker='o')
plt.grid(True)
plt.xlabel('excepted volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe ratio')

def statistics(weights):
    weights = np.array(weights)
    port_returns = np.sum(ret.mean()*weights)*252
    port_variance = np.sqrt(np.dot(weights.T, np.dot(ret.cov()*252,weights)))
    return np.array([port_returns, port_variance, port_returns/port_variance])

import scipy.optimize as sco

def min_sharpe(weights):
    return -statistics(weights)[2]

def min_variance(weights):
    return statistics(weights)[1]

cons = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1})
bnds = tuple((0,1) for x in range(noa))
optv = sco.minimize(min_variance, noa*[1./noa], method='SLSQP', bounds=bnds, constraints=cons)
opts = sco.minimize(min_sharpe, noa*[1./noa], method='SLSQP', bounds=bnds, constraints=cons)
opts

bnds = ((0,.35),(0,.3),(0,.3),(0,.4),(0,.5))
cov = np.array([[.01,0,0,0,0],[0,.04,-.05,0,0],[0,-.05,.3,0,0],[0,0,0,.4,.2],[0,0,0,.2,.4]])
cons = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1},
        {'type':'ineq', 'fun':lambda x: x[0] + x[1] - 0.2},
        {'type':'ineq', 'fun':lambda x: -x[0] - x[1] + 0.6},
        {'type':'ineq', 'fun':lambda x: x[2] + x[3] + x[4] - 0.3},
        {'type':'ineq', 'fun':lambda x: -x[2] - x[3] - x[4] + 0.7})

#在不同目标收益率水平（target_returns）循环时，最小化的一个约束条件会变化。
target_returns = np.linspace(0.0,0.5,50)
target_variance = []
for tar in target_returns:
    cons = ({'type':'eq','fun':lambda x:statistics(x)[0]-tar},{'type':'eq','fun':lambda x:np.sum(x)-1})
    res = sco.minimize(min_variance, noa*[1./noa,],method = 'SLSQP', bounds = bnds, constraints = cons)
    target_variance.append(res['fun'])

target_variance = np.array(target_variance)


plt.figure(figsize = (8,4))
#圆圈：蒙特卡洛随机产生的组合分布
plt.scatter(port_variance, port_returns, c = port_returns/port_variance,marker = 'o')
#叉号：有效前沿
plt.scatter(target_variance,target_returns, c = target_returns/target_variance, marker = 'x')
#红星：标记最高sharpe组合
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0], 'r*', markersize = 15.0)
#黄星：标记最小方差组合
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'y*', markersize = 15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe ratio')