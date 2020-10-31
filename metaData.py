import pandas as pd
import yfinance as yf
from datetime import datetime
from Data import Data
import json
import time

meta = pd.read_csv("EOD_metadata.csv")
stocks = list(meta['code'])

data_engie = Data(datetime(2000,1,1), datetime(2020,10,1))

meta = {}
for stock in stocks:
        print(stock)
        try:
            info = data_engie.get_ticker_info(ticker=stock)
            meta[stock] = info['sector']
        except:
            print("Error : {}".format(stock))
        time.sleep(0.01)

meta = json.dumps(meta)
with open("Meta.json", 'w') as f:
    json.dump(meta, f)