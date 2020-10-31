import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class Long_Short_Comb(object):
    def __init__(self, stock, index):
        self.stock = stock
        self.index = index
        self.index_ret = np.log(self.index/self.index.shift(1)).dropna()
        frame = pd.concat([self.index, self.stock], axis=1)
        self.ret_frame = np.log(frame / frame.shift(1)).dropna()
        self.ret_frame.columns = ['index', 'stock']

    def get_short_ret(self)->dict:
        ret = {}
        ret['index'] = float(self.ret_frame['index'].mean() * 252)
        ret['stock'] = float(self.ret_frame['stock'].mean() * 252)
        return ret

    def get_short_vol(self)->dict:
        ret = {}
        ret['index'] = float(self.ret_frame['index'].std() * 252 ** 0.5)
        ret['stock'] = float(self.ret_frame['stock'].std() * 252 ** 0.5)
        return ret

    def get_short_cov(self)->pd.DataFrame:
        return self.ret_frame.cov() * 252

    def get_beta(self)->float:
        cov = self.get_short_cov()
        return float(cov.iloc[0,1] / cov.iloc[0,0])

    def get_beta_hat(self)->float:
        long_index_var = float(self.index_ret.var()) * 252

        short_index_var = float(self.ret_frame['index'].var() * 252)
        short_stock_var = float(self.ret_frame['stock'].var() * 252)

        short_cov = self.get_short_cov()
        beta = self.get_beta()

        long_stock_var = short_stock_var + beta ** 2 * (long_index_var - short_index_var)
        long_cross_cov = float(short_cov.iloc[0,1]) + beta * (long_index_var - short_index_var)
        return long_cross_cov / long_index_var

    def get_long_index_ret(self)->float:
        return self.index_ret.mean() * 252

    def get_long_stock_ret(self)->float:
        short_ret = self.get_short_ret()
        beta = self.get_beta()
        long_index_ret = self.get_long_index_ret()
        long_stock_ret = short_ret['stock'] + beta * (long_index_ret - short_ret['index'])
        return long_stock_ret

    def get_estimate_series(self):
        beta_hat = self.get_beta_hat()
        long_index_ret = self.get_long_index_ret()
        long_stock_ret = self.get_long_stock_ret()
        estimate = (self.index_ret - long_index_ret/252) * beta_hat + long_stock_ret/252
        return estimate


if __name__ == "__main__":
    google = yf.download("GOOG", start='2000-01-01')["Adj Close"]
    nasdaq = yf.download("^IXIC", start='2000-01-01')['Adj Close']
    pricer = Long_Short_Comb(google, nasdaq)
    print(pricer.get_estimate_series())