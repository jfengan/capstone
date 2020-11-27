from datetime import datetime
import numpy as np
import pandas as pd
import yfinance

pd.set_option('mode.chained_assignment', None)

class Data:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = yfinance.download(tickers=self.ticker)
        self.data['return'] = np.log(self.data['Adj Close']) - np.log(self.data["Adj Close"].shift(1))
        self.data = self.data.dropna()


    def get_indicator_MOM(self, window: int):
        self.data['MOM{}'.format(window)] = self.data['High'].rolling(window=window).max() \
                           - self.data["Low"].rolling(window=window).min()

    def get_indicator_TSMOM(self, window: int):
        for i in range(window, len(self.data.index)):
            slice_ret = self.data.iloc[i-window:i, :]
            pos_ret = slice_ret[slice_ret['return'] > 0]['return'].mean()
            neg_ret = slice_ret[slice_ret['return'] < 0]['return'].mean()
            self.data.loc[self.data.index[i], 'TSMOM'.format(window)] = (pos_ret - neg_ret)

    def get_indicator_term(self):
        self.data['logPrice'] = np.log(self.data["Adj Close"])
        self.data['2mo'] = self.data['logPrice'] - self.data['logPrice'].shift(44)
        self.data['4mo'] = self.data['logPrice'] - self.data['logPrice'].shift(88)
        self.data['6mo'] = self.data['logPrice'] - self.data['logPrice'].shift(132)
        self.data['TERM'] = self.data[['2mo', '4mo', '6mo']].mean(axis=1)


if __name__ == "__main__":
    core = Data(ticker="ZC=F")
    #core.get_indicator_MOM(window=1)
    core.get_indicator_TSMOM(window=2)
    print(core.data)