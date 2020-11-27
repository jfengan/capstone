from config import commodities
from Data import Data
import numpy as np
import pandas as pd


def generate_data_set():
    dataset = {}
    for genre, tickers in commodities.items():
        dataset[genre] = {}
        for key, ticker in tickers.items():
            temp = Data(ticker=ticker)
            temp.get_indicator_MOM(window=30)
            temp.get_indicator_TSMOM(window=30)
            dataset[genre][key] = temp


class Commodity_Data:
    def __init__(self, dataset: dict):
        self.ret_frame = dict()
        for key, value in commodities.items():
            ticker = Data(ticker=value)
            self.ret_frame[key] = ticker.data['return']
        self.ret_frame = pd.DataFrame(self.ret_frame)
        # print(self.ret_frame)
        self.ret_frame = self.ret_frame.dropna()

    def get_market_factor(self):
        # del self.ret_frame['market_factor']
        self.ret_frame['market_factor'] = self.ret_frame.mean(axis=1)

    @staticmethod
    def get_factor_HML(x):
        mid = np.median(x)
        high_group = x[x > mid]
        low_group = x[x < mid]
        return high_group.mean() - low_group.mean()

    def get_HML_factor(self):
        self.ret_frame['HML'] = self.ret_frame.apply(lambda x: self.get_factor_HML(x), axis=1)
        # print(self.ret_frame["HML"])


if __name__ == "__main__":
    commodityData = Commodity_Data()
    commodityData.get_HML_factor()
