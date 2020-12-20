# Capstone Project of HKUST MAFM 6100

- [Capstone Project of HKUST MAFM 6100](#capstone-project-of-hkust-mafm-6100)
  * [Main Idea](#main-idea)
  * [Projects](#projects)
    + [Data.py](#datapy)
    + [Long_Short_Pricer.py](#long-short-pricerpy)
    + [CommodityPricer.ipynb](#commoditypriceripynb)
    + [EquityPricer.py](#equitypricerpy)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## Main Idea

The aim is to come up with a methodology for creating proper factor models for the missing data for the asset classes specified. The factor models could be linear (multifactor beta) factor models or a combination of linear and nonlinear factors. The factors within the models could be tradable assets or indexes (preferred) or derived from Principal Components. The factor models obtained should capture a high R^2 with the dependent variable with in-sample data but the returns of the dependent variable via the fallback factor models should preserve specific correlation with other assets during the historical period of interest.



## Projects

---

> requirements: talib, yfinance, pandas, numpy, keras, sklearn

---



### Data.py

The data project is an encapsulation of yahoo finance API, together with some pre-processing for computing daily returns and other technical risk factors, which are used to build the factor models

---



### Long_Short_Pricer.py

1. **Highlights**: This script is more for simulating the stock return under different market conditions, ***especially for stocks that are not listed yet***. 

2. **Methdology**:

   - Collect data for underlying index(longer-term) and a specific instrument(shorter-term).
     - the underlying index could be either specific instruments or indices 
     - the program would assert an error if the instrument time goes beyond the underlying index

   - Combine the long and short data together: 
     - extend the short-term vol with long-term vol, which means that we use the vol proxy to capture the market condition
     -  then we employ beta factor to simulate the stock return combination under the unobservable period
   - Use the sign of the return as the benchmark to test the efficiency of the model

~~~bash
python3 -index IXIC -symbol GOOG -index_start 2003-09-08 -symbol_start 2005-08-09
~~~



### CommodityPricer.ipynb

1. **Highlights**: This model follows a auto-model-selection method to let the model select risk factors available 
2. **Methodology:** 
   - Load data, compute the technical factors of commodity instruments
   - compute sector factors and market factors
   - use AIC as the indicator and forward selection to keep the top4 effective factors to form a multi-factor model



### EquityPricer.py

1. **Highlights:** We employ the Long Short-Term Memory model to predict the price move of a specific stock
2. **Methodology:** 
   - load data:
     - select the most relative index as the market factor (Dow Jones, Nasdaq, Russel or S&P 500)
     - compute technical factors
       - Momentum
       - Open Balance Volume
       - True Range
       - Dominant Cycle Period (Hilbert Transform)
     - load financial indicators:
       - P/E ratios
       - P/B ratios
       - Return on Assets
       - Return on Equity
   - model training:
     - Notice: shuffle the time-series data input to get better result
   - Evaluation: 
     - use R square as the evaluation method



