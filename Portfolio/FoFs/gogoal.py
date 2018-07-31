#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Created on Thu Apr 06 13:52:15 2017

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
Script for MySQL link to Go-Goal DataBase.

Info for DB:
    地址：106.75.45.237
    端口：15427
    账号：simu_ahzb
    密码：D0DTJtm34bMLMCZI
    数据库：CUS_FUND_DB  (视图表)

    mysql -u simu_ahzb -p D0DTJtm34bMLMCZI -h 106.75.45.237 -P 15427 -D CUS_FUND_DB

"""
__author__ = 'Neal Chen Zhang'

import pymysql

config = {
          'host': '106.75.45.237',
          'port': 15427,
          'user': 'simu_ahzb',
          'password': 'D0DTJtm34bMLMCZI',
          'db': 'CUS_FUND_DB'
          }

conn = pymysql.connect(**config)

cursor = conn.cursor()

date_start = '2016-01-01'
date_end = '2016-06-30'

# 执行sql语句
try:
    with conn.cursor() as cursor:
        # 执行sql语句，进行查询
        sql = 'select a.fund_id,a.fund_type_strategy,a.reg_code,a.fund_name,b.statistic_date ,b.nav,b.added_nav \
               from v_fund_info a LEFT JOIN v_fund_weekly_performance b on a.fund_id=b.fund_id \
               where a.fund_type_strategy="管理期货" order by a.fund_id ,statistic_date desc'
        cursor.execute(sql)
        # 获取查询结果
        result = cursor.fetchall()
        print(result)
    # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
    conn.commit()
 
finally:
    conn.close();
