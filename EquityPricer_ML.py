from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simfin as sf
import talib
import yfinance as yf


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh = logging.FileHandler("train_model.log", mode='a')
fh.setFormatter(formatter)
logger.addHandler(fh)


def get_r_square(df):
    y_bar = df['Price'].mean()
    df['diff'] = df['Price'] - df['predict']
    df["sr"] = df['diff'].apply(lambda x:x**2)
    df['st'] = df['Price'].apply(lambda x: (x-y_bar)**2)
    ssr = df['sr'].sum()
    sst = df['st'].sum()
    return 1-ssr/sst


class EquityPricer(object):
    def __init__(self, ticker, market):
        self.ticker = ticker
        self.market = market
        self.frame = None
        sf.load_api_key('lcZUpsmDGVHd0bioyM44GnzLZzPFIP8b')

    def load_market_index(self):
        GSPC = yf.download('^GSPC')
        IXIC = yf.download("^IXIC")
        DJI = yf.download("^DJI")

        GSPC['ret'] = np.log(GSPC['Adj Close'] / GSPC['Adj Close'].shift(1))
        IXIC['ret'] = np.log(IXIC['Adj Close'] / IXIC['Adj Close'].shift(1))
        DJI['ret'] = np.log(DJI['Adj Close'] / DJI['Adj Close'].shift(1))

        frame = GSPC[['ret']].merge(IXIC[['ret']], left_index=True, right_index=True, how='outer')
        frame = frame.merge(DJI[['ret']], left_index=True, right_index=True, how='outer')
        frame.columns = ['GSPC', 'IXIC', 'DJI']
        frame['GSPC Price'] = GSPC['Adj Close']
        frame['IXIC Price'] = IXIC['Adj Close']
        frame['DJI Price'] = DJI['Adj Close']
        return frame

    def load_equity_indicator(self, window) -> pd.DataFrame:
        data = yf.download(self.ticker)
        data['ret'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        market_frame = self.load_market_index()
        market_frame = pd.merge(market_frame, data[['ret', 'Adj Close', 'Open', "High", "Low", "Close", "Volume"]],
                                        left_index=True, right_index=True, how='outer')
        market_frame = market_frame.dropna()
        frame_corr = market_frame.corr()
        frame_corr = frame_corr.loc['ret'].sort_values(ascending=False)
        for i in range(1, len(frame_corr.index)):
            index = frame_corr.index[i]
            if index in ['GSPC', "IXIC", 'DJI']:
                break
            else:
                continue
        # print(market_frame)
        market_frame = market_frame[['High', 'Open', 'Low', 'Close', 'Volume', 'Adj Close', '{} Price'.format(index)]]
        market_frame.columns = ['High', 'Open', 'Low', 'Close', 'Volume', 'Adj Close', 'Market']

        market_frame['NATR'] = talib.TRANGE(high=market_frame['High'], low=market_frame['Low'], close=market_frame['Close'])
        market_frame['MOM'] = talib.MOM(market_frame['Adj Close'], timeperiod=window)
        market_frame['DCP'] = talib.HT_DCPERIOD(market_frame['Close'])
        market_frame['OBV'] = talib.OBV(market_frame['Close'], market_frame['Volume'])
        df = pd.DataFrame(index=market_frame.index)
        df['Price'] = market_frame["Adj Close"]
        df['TRANGE'] = market_frame['NATR']
        df['DCP'] = market_frame['DCP']
        df['OBV'] = market_frame['OBV']
        df['Market'] = market_frame['Market']
        df = df.dropna()
        return df

    def load_fin_factor(self) -> pd.DataFrame:
        hub = sf.StockHub(market='us', tickers=[self.ticker],
                          offset=pd.DateOffset(days=60), refresh_days=30,
                          refresh_days_shareprices=10)
        sf.set_data_dir('simfin_data/')
        df_fin_signals_daily = hub.fin_signals(variant='daily')
        df_val_signals_daily = hub.val_signals(variant='daily')
        return pd.merge(df_fin_signals_daily, df_val_signals_daily, left_index=True, right_index=True)

    def load_data(self, window: int):
        indicator = self.load_equity_indicator(window=window)
        fin_factor = self.load_fin_factor()
        factor_frame = pd.merge(indicator, fin_factor, right_index=True, left_index=True)
        # print(len(df.index))
        factors = ['Price', 'Market', 'TRANGE', 'DCP', "OBV", 'Return on Assets', 'Return on Equity', 'P/E',
                   'Price to Book Value']
        input_data = factor_frame[factors]
        return input_data

    def train(self, window: int):
        df = self.load_data(window=window)
        df = df.dropna(axis=1)
        no_of_train = int(len(df.index) * 0.95)
        training_set = df.iloc[0:no_of_train, :]
        sc = StandardScaler()
        training_set_scaled = sc.fit_transform(training_set)
        price_ts = df['Price'].values
        price_ts = np.reshape(price_ts, (-1, 1))
        inverse = MinMaxScaler(feature_range=(0, 1))
        training_y = inverse.fit_transform(price_ts)

        x_train = []
        y_train = []
        for i in range(window, len(training_set)):
            x_train.append(training_set_scaled[i - window:i, :])
            y_train.append(training_y[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressor.fit(x_train, y_train, epochs=40, batch_size=16)

        df_test = df.iloc[no_of_train - window:, :]
        inputs = df_test.values
        inputs = sc.transform(inputs)
        x_test = []
        for i in range(window, len(inputs)):
            x_test.append(inputs[i-window:i, :])
        x_test = np.array(x_test)

        predicted_price = regressor.predict(x_test)
        predicted_price = inverse.inverse_transform(predicted_price)

        df_test = df_test.iloc[window:, :]
        df_test['predict'] = predicted_price
        df_test['predict'] = df_test['predict'].astype(float)
        df_test['Price'] = df['Price'].loc[df_test.index[0]:]
        df_test[['Price', 'predict']].plot(figsize=(18,6))
        plt.savefig("results/{}_{}.png".format(self.ticker, window))
        plt.close()
        logger.info("{}-{}: {}".format(self.ticker, window, get_r_square(df_test)))


if __name__ == "__main__":
    for _ticker in ['WMT', 'X', 'XL', 'XOM', 'YHOO']:
        testcase = EquityPricer(ticker=_ticker, market='us')
        testcase.train(window=3)
        testcase.train(window=5)
        testcase.train(window=10)
        testcase.train(window=22)
        testcase.train(window=66)
