import numpy as np
import pandas as pd

returns = prices.pct_change()
returns.dropna()

returns.std()

deviations = (returns - returns.mean())**2
squared_deviations = deviations ** 2
variance = squared_deviations.mean()
volatility = np.sqrt(variance)


me_m = pd.read_csv('./Data/Portfolios_Formed_on_ME_monthly_EW.csv',
                     header=0, index_col=0, parse_dates=True, na_values=-99.99)
rets = me_m[['Lo 10', 'Hi 10']]
rets.columns = ['SmallCap', 'LargeCap']
rets = rets / 100
rets.plot.line()

rets.head()
rets.index = pd.to_datetime(rets.index, format='%Y%m')
rets.head()
rets.index = rets.index.to_period('M')
rets['1975']

wealth_index = 1000 * (1+rets['LargeCap']).cumprod()
wealth_index.plot.line()
previous_peaks = wealth_index.cummax()
previous_peaks.plot.line()
drawdown = (wealth_index - previous_peaks) / previous_peaks
drawdown.plot()
drawdown.min()

drawdown['1975':].min()
drawdown['1975':].idxmin()


def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percent drawdowns
    :param return_series:
    :return:
    """
    wealth_index = 1000 * (1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame(
        {
            "Wealth": wealth_index,
            "Peaks": previous_peaks,
            "Drawdown": drawdowns
        }
    )


drawdown(rets['LargeCap']).head()
drawdown(rets['LargeCap'])[['Wealth', 'Peaks']].plot()


import pandas as pd
import EDHEC.edhec_risk_kit as erk


hfi = erk.get_hfi_returns()
hfi.head()


pd.concat([hfi.mean(), hfi.median(), hfi.mean()>hfi.median()], axis='columns')
erk.skewness(hfi).sort_values()

import scipy.stats
scipy.stats.skew(hfi)

import numpy as np
normal_rets = np.random.normal(0, .15, size=(263, 1))
erk.skewness(normal_rets)

erk.kurtosis(normal_rets)
erk.kurtosis(hfi)
scipy.stats.kurtosis(normal_rets)

scipy.stats.jarque_bera(normal_rets)
scipy.stats.jarque_bera(hfi)

erk.is_normal(normal_rets)
hfi.aggregate(erk.is_normal)
ffme = erk.get_ffme_returns()
erk.skewness(ffme)
erk.kurtosis(ffme)


hfi.std(ddof=0)
hfi[hfi<0].std(ddof=0)
erk.semideviation(hfi)


# Historical VaR
# Parametric VaR - Gaussian
# Modified Cornish-Fisher VaR

np.percentile(hfi, q=5, axis=0)

hfi.apply(lambda x: np.percentile(x, q=5, axis=0))

erk.var_historic(hfi)

from scipy.stats import norm
z = norm.ppf(.05)

hfi.mean() + z*hfi.std(ddof=0)
erk.var_gaussian(hfi)

var_list = [erk.var_gaussian(hfi), erk.var_gaussian(hfi, modified=True), erk.var_historic(hfi)]
comparison = pd.concat(var_list, axis=1)
comparison.columns = ['Gaussian', 'Cornish-Fisher', 'Historic']
comparison.plot.bar(title='EDHEC Hedge Fund Indices: VaR Comparison')

erk.cvar_historic(hfi)



