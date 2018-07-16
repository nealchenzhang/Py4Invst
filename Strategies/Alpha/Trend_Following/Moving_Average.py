import numpy as np
import pandas as pd


class Moving_Averages(object):

    def simpleMA(tsPrice, k):
        """

        :param tsPrice: past price Series
        :param k: lag number
        :return sma: simple MA price Series
        """
        sma = tsPrice.rolling(window=k).apply(np.mean)
        return sma

    def weightedMA(tsPrice, weight):
        """
        :param tsPrice: past price Series
        :param weight: list of past price weights
        :return wma: weighted MA price Series
        """
        wma = tsPrice.rolling(window=len(weight)).apply(lambda x: np.average(x, weights=weight))
        return wma

    def ewMA(tsprice, period=5, exponential=0.2):
        Ewma = pd.Series(0.0, index=tsprice.index)
        Ewma[period-1] = np.mean(tsprice[:period])
        for i in range(period, len(tsprice)):
            Ewma[i] = exponential*tsprice[i] + (1- exponential)*Ewma[i-1]
        return Ewma
