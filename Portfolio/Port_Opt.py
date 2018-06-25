import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import os
os.chdir('/home/nealzc/PycharmProjects/Py4Invst/Portfolio')

class Port_Opt(object):
    def __init__(self, port_list):
        self.columns = [i.lower() for i in port_list]

        for i in port_list:
            setattr(self, i, pd.read_csv(i+'_CLOSE', index_col='Date', parse_dates=True))

        setattr(self, 'df_port', pd.concat([getattr(self, i) for i in port_list], axis=1))
        self.df_port.columns = self.columns

        df_port_daily_log_ret = np.log(self.df_port / self.df_port.shift(1))
        setattr(self, 'log_ret', df_port_daily_log_ret)

    def port_analysis(self):
        df_port = self.df_port
        df_port_normed = df_port / df_port.iloc[0]
        df_port_normed.plot()

        df_port_daily_log_ret = np.log(df_port / df_port.shift(1))
        df_port_daily_log_ret.hist(bins=100, figsize=(12,6));
        plt.tight_layout()

    def MC_simulation(self, num_ports=15000):
        log_ret = self.log_ret
        all_weights = np.zeros((num_ports, len(self.df_port.columns)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)

        for ind in range(num_ports):
            weights = np.array(np.random.random(4))
            weights = weights / np.sum(weights)

            all_weights[ind, :] = weights

            ret_arr[ind] = np.sum((log_ret.mean() * weights) * 252)
            vol_arr[ind] = np.sqrt(weights.T @ (log_ret.cov()*252) @ weights)

            sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

        i_max = sharpe_arr.argmax()
        max_sr_ret = ret_arr[i_max]
        max_sr_vol = vol_arr[i_max]

        plt.figure(figsize=(12, 8))
        plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')
        # ax = plt.gca()
        # ax.text(max_sr_vol-0.02, max_sr_ret+0.05, 'The weight is \n{}'.format(all_weights[i_max]))
        # plt.show()

    def get_ret_vol_sr(self, weights):
        """
        Takes in weights, returns array or return,volatility, sharpe ratio
        """
        log_ret = self.log_ret
        weights = np.array(weights)
        ret = np.sum(log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])


if __name__ == '__main__':
    stocks_list = ['AAPL', 'CISCO', 'IBM', 'AMZN']
    port1 = Port_Opt(stocks_list)
    # port1.MC_simulation()

    ####################################################
    # def neg_sharpe(weights):
    #     return port1.get_ret_vol_sr(weights)[2] * (-1)
    #
    def check_sum(weights):
        return np.sum(weights) - 1
    #
    # obj = neg_sharpe
    cons = ({'type': 'eq', 'fun': check_sum})
    bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
    init_guess = [0.25, 0.25, 0.25, 0.25]
    # ####################################################
    # opt_results = minimize(obj, init_guess, method='SLSQP',
    #                        bounds=bounds, constraints=cons)
    # print(opt_results)

    ####################################################

    frontier_y = np.linspace(0, 0.3, 100)

    def minimize_vol(weights):
        return port1.get_ret_vol_sr(weights)[1]


    frontier_volatility = []

    for possible_return in frontier_y:
        # function for return
        cons = ({'type': 'eq', 'fun': check_sum},
                {'type': 'eq', 'fun': lambda w: port1.get_ret_vol_sr(w)[0] - possible_return})

        result = minimize(minimize_vol, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

        frontier_volatility.append(result['fun'])

    port1.MC_simulation()
    plt.plot(frontier_volatility, frontier_y, 'g--', linewidth=3)
    plt.show()

