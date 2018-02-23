import pandas as pd
import numpy as np
import datetime

term = 'long-term'
GDP = 'increase'

dict_indicator = {'GDP': None,
                  'Unemployment_rate': None,
                  'CPI': None,
                  'PPI': None}

def Global_Macro_Indicator(term, dict_indicator):
    if term == 'short-term':
        print()

    elif term == 'long-term':

