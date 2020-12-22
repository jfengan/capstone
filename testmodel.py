from EquityPricer import *
from scipy import stats

startdate='2013-01-01'

#dowjones_symlist = ['V', 'WMT', 'MMM', 'IBM', 'MRK', 'MSFT', 'DIS', 'MCD', 'WBA', 'JPM', 'PG', 'JNJ', 'TRV', 'CRM', 'BA', 'VZ', 'KO', 'HON', 'GS', 'UNH', 'CAT', 'AMGN', 'AXP', 'HD', 'CVX', 'CSCO', 'AAPL', 'NKE', 'DOW', 'INTC']
#dowjones_adjclose=yf.download(tickers=dowjones_symlist, start=startdate, group_by='ticker', threads=True)
#dowjones_adjclose=dowjones_adjclose.loc[:, (slice(None), 'Adj Close')]
#dowjones_adjclose.columns = [a for a, b in dowjones_adjclose.columns]
#dowjones_adjclose=dowjones_adjclose.dropna(how='all')
#dowjones_adjclose.to_csv('C:/Users/Administrator/Desktop/dowjones_adjclose_since2013.csv')
dowjones_adjclose=pd.read_csv('C:/Users/Administrator/Desktop/dowjones_adjclose_since2013.csv')
dowjones_adjclose=dowjones_adjclose.set_index(dowjones_adjclose.Date)
dowjones_adjclose=dowjones_adjclose.drop('Date', axis=1)
dowjones_adjclose_valid=dowjones_adjclose.dropna(axis=1)
dowjones_adjclose_valid.index=pd.to_datetime(dowjones_adjclose_valid.index)
dowjones_index = yf.download('^DJI', start=startdate)['Adj Close']
dowjones_adjclose_test, dowjones_adjclose_train = np.split(dowjones_adjclose_valid, [int(.25*len(dowjones_adjclose_valid))])
result=dowjones_adjclose_valid.iloc[1:].copy(True)
result.iloc[:,:]=0
for j in range(0,len(result.iloc[0])):
    pricer = Long_Short_Comb(dowjones_adjclose_train.iloc[:,j], dowjones_index)
    result.iloc[:,j]=pricer.get_estimate_series()

#test set block start
dowjones_ret_test=np.log(dowjones_adjclose_test/dowjones_adjclose_test.shift(1)).dropna()
dowjones_ret_test_mean=dowjones_ret_test.mean()
result_test=result.iloc[0:len(dowjones_ret_test)]
result_test_mean=result_test.mean()
ret_equalsign=(result_test_mean*dowjones_ret_test_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#0.9310344827586207
dowjones_ret_test_std=dowjones_ret_test.std()
result_test_std=result_test.std()
print((result_test_std/dowjones_ret_test_std).mean())
#0.5535023262022343
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_test.iloc[:,j], dowjones_ret_test.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.068966
corr_test=[0]*len(result.iloc[0])
for j in range(0,len(corr_test)):
    corr_test[j]=dowjones_ret_test.iloc[:,j].corr(result_test.iloc[:,j])
corr_test=pd.DataFrame(corr_test)
print(corr_test.mean())
#0.568961
print(corr_test.std())
#0.126818
#test set block end

#training set block start
dowjones_ret_train=np.log(dowjones_adjclose_train/dowjones_adjclose_train.shift(1)).dropna()
dowjones_ret_train_mean=dowjones_ret_train.mean()
result_train=result.iloc[len(result)-len(dowjones_ret_train):]
result_train_mean=result_train.mean()
ret_equalsign=(result_train_mean*dowjones_ret_train_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#1.0
dowjones_ret_train_std=dowjones_ret_train.std()
result_train_std=result_train.std()
print((result_train_std/dowjones_ret_train_std).mean())
#0.6975872471532581
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_train.iloc[:,j], dowjones_ret_train.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.0
corr_train=[0]*len(result.iloc[0])
for j in range(0,len(corr_train)):
    corr_train[j]=dowjones_ret_train.iloc[:,j].corr(result_train.iloc[:,j])
corr_train=pd.DataFrame(corr_train)
print(corr_train.mean())
#0.697587
print(corr_train.std())
#0.089875
#training set block end

#nasdaq_symlist = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX', 'CMCSA', 'INTC', 'PEP', 'COST', 'QCOM', 'CSCO', 'AVGO', 'TMUS', 'TXN', 'AMGN', 'CHTR', 'SBUX', 'AMD', 'INTU', 'ZM', 'ISRG', 'JD', 'MDLZ', 'GILD', 'BKNG', 'MELI', 'ADP', 'FISV', 'CSX', 'AMAT', 'ATVI', 'REGN', 'MU', 'LRCX', 'VRTX', 'ADSK', 'BIIB', 'ADI', 'ILMN', 'MNST', 'DOCU', 'LULU', 'EXC', 'KDP', 'BIDU', 'NXPI', 'CTSH', 'WDAY', 'IDXX', 'ALGN', 'KHC', 'XEL', 'EA', 'SNPS', 'KLAC', 'CTAS', 'CDNS', 'EBAY', 'DXCM', 'ROST', 'ORLY', 'SPLK', 'NTES', 'MAR', 'PDD', 'WBA', 'XLNX', 'SGEN', 'VRSK', 'PCAR', 'PAYX', 'ASML', 'MCHP', 'CPRT', 'MRNA', 'ANSS', 'ALXN', 'FAST', 'SIRI', 'SWKS', 'VRSN', 'DLTR', 'CERN', 'MXIM', 'CDW', 'TTWO', 'INCY', 'CHKP', 'TCOM', 'CTXS', 'EXPE', 'BMRN', 'ULTA', 'FOXA', 'LBTYK', 'FOX', 'LBTYA']
#nasdaq_adjclose=yf.download(tickers=nasdaq_symlist, start=startdate, group_by='ticker', threads=True)
#nasdaq_adjclose=nasdaq_adjclose.loc[:, (slice(None), 'Adj Close')]
#nasdaq_adjclose.columns = [a for a, b in nasdaq_adjclose.columns]
#nasdaq_adjclose=nasdaq_adjclose.dropna(how='all')
#nasdaq_adjclose.to_csv('C:/Users/Administrator/Desktop/nasdaq_adjclose_since2013.csv')
nasdaq_adjclose=pd.read_csv('C:/Users/Administrator/Desktop/nasdaq_adjclose_since2013.csv')
nasdaq_adjclose=nasdaq_adjclose.set_index(nasdaq_adjclose.Date)
nasdaq_adjclose=nasdaq_adjclose.drop('Date', axis=1)
nasdaq_adjclose_valid=nasdaq_adjclose.dropna(axis=1)
nasdaq_adjclose_valid.index=pd.to_datetime(nasdaq_adjclose_valid.index)
nasdaq_index = yf.download('^IXIC', start=startdate)['Adj Close']
nasdaq_adjclose_test, nasdaq_adjclose_train = np.split(nasdaq_adjclose_valid, [int(.25*len(nasdaq_adjclose_valid))])
result=nasdaq_adjclose_valid.iloc[1:].copy(True)
result.iloc[:,:]=0
for j in range(0,len(result.iloc[0])):
    pricer = Long_Short_Comb(nasdaq_adjclose_train.iloc[:,j], nasdaq_index)
    result.iloc[:,j]=pricer.get_estimate_series()

#test set block start
nasdaq_ret_test=np.log(nasdaq_adjclose_test/nasdaq_adjclose_test.shift(1)).dropna()
nasdaq_ret_test_mean=nasdaq_ret_test.mean()
result_test=result.iloc[0:len(nasdaq_ret_test)]
result_test_mean=result_test.mean()
ret_equalsign=(result_test_mean*nasdaq_ret_test_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#0.9354838709677419
nasdaq_ret_test_std=nasdaq_ret_test.std()
result_test_std=result_test.std()
print((result_test_std/nasdaq_ret_test_std).mean())
#0.5043586416625926
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_test.iloc[:,j], nasdaq_ret_test.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.096774
corr_test=[0]*len(result.iloc[0])
for j in range(0,len(corr_test)):
    corr_test[j]=nasdaq_ret_test.iloc[:,j].corr(result_test.iloc[:,j])
corr_test=pd.DataFrame(corr_test)
print(corr_test.mean())
#0.506971
print(corr_test.std())
#0.110469
#test set block end

#training set block start
nasdaq_ret_train=np.log(nasdaq_adjclose_train/nasdaq_adjclose_train.shift(1)).dropna()
nasdaq_ret_train_mean=nasdaq_ret_train.mean()
result_train=result.iloc[len(result)-len(nasdaq_ret_train):]
result_train_mean=result_train.mean()
ret_equalsign=(result_train_mean*nasdaq_ret_train_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#1.0
nasdaq_ret_train_std=nasdaq_ret_train.std()
result_train_std=result_train.std()
print((result_train_std/nasdaq_ret_train_std).mean())
#0.6176624651521365
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_train.iloc[:,j], nasdaq_ret_train.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.0
corr_train=[0]*len(result.iloc[0])
for j in range(0,len(corr_train)):
    corr_train[j]=nasdaq_ret_train.iloc[:,j].corr(result_train.iloc[:,j])
corr_train=pd.DataFrame(corr_train)
print(corr_train.mean())
#0.617662
print(corr_train.std())
#0.110614
#training set block end

#sp500_symlist = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK-B', 'JNJ', 'NVDA', 'PG', 'V', 'UNH', 'JPM', 'HD', 'MA', 'VZ', 'PYPL', 'ADBE', 'CRM', 'DIS', 'NFLX', 'TMO', 'MRK', 'CMCSA', 'ABT', 'WMT', 'PFE', 'T', 'INTC', 'PEP', 'KO', 'BAC', 'COST', 'ABBV', 'QCOM', 'MCD', 'NKE', 'CSCO', 'DHR', 'AVGO', 'ACN', 'NEE', 'TXN', 'BMY', 'MDT', 'XOM', 'CVX', 'AMGN', 'LIN', 'HON', 'UNP', 'LOW', 'UPS', 'LLY', 'ORCL', 'PM', 'AMT', 'SBUX', 'NOW', 'IBM', 'AMD', 'CHTR', 'MMM', 'BLK', 'INTU', 'WFC', 'RTX', 'C', 'LMT', 'CAT', 'ISRG', 'SPGI', 'CVS', 'BA', 'FIS', 'TGT', 'ZTS', 'MDLZ', 'ANTM', 'DE', 'PLD', 'GILD', 'CI', 'MS', 'BKNG', 'TMUS', 'ADP', 'CL', 'GE', 'EQIX', 'D', 'SYK', 'MO', 'GS', 'DUK', 'BDX', 'CCI', 'APD', 'FDX', 'CSX', 'TJX', 'AMAT', 'SO', 'AXP', 'CB', 'MU', 'ATVI', 'SCHW', 'REGN', 'SHW', 'ITW', 'LRCX', 'TFC', 'FISV', 'VRTX', 'PGR', 'HUM', 'NSC', 'ADSK', 'ICE', 'MMC', 'DG', 'NEM', 'CME', 'EL', 'USB', 'BIIB', 'GPN', 'BSX', 'ECL', 'ADI', 'PNC', 'EW', 'GM', 'MCO', 'NOC', 'KMB', 'WM', 'ILMN', 'AEP', 'AON', 'ETN', 'DD', 'EMR', 'EXC', 'IDXX', 'ROP', 'CTSH', 'BAX', 'DLR', 'CNC', 'LHX', 'XEL', 'CMG', 'GIS', 'SNPS', 'HCA', 'SRE', 'KLAC', 'DOW', 'APH', 'COF', 'MSCI', 'PSA', 'CDNS', 'A', 'TEL', 'DXCM', 'TT', 'EA', 'SBAC', 'ALGN', 'TWTR', 'CMI', 'EBAY', 'ROST', 'ORLY', 'INFO', 'IQV', 'XLNX', 'PPG', 'JCI', 'GD', 'TRV', 'VRSK', 'WEC', 'BLL', 'CARR', 'MCHP', 'COP', 'ES', 'MET', 'STZ', 'SYY', 'PCAR', 'MNST', 'RMD', 'PH', 'TROW', 'F', 'YUM', 'CTAS', 'PEG', 'AWK', 'ANSS', 'ROK', 'SWK', 'AIG', 'ALL', 'ZBH', 'BK', 'MTD', 'APTV', 'TDG', 'BBY', 'PAYX', 'MSI', 'FCX', 'MAR', 'MCK', 'ALXN', 'CLX', 'AZO', 'WBA', 'ADM', 'FAST', 'HPQ', 'GLW', 'ED', 'KR', 'HLT', 'OTIS', 'CPRT', 'CTVA', 'PRU', 'AME', 'WLTW', 'SWKS', 'AFL', 'DTE', 'LUV', 'DHI', 'WELL', 'FTV', 'MKC', 'DLTR', 'VFC', 'CERN', 'KMI', 'EIX', 'HSY', 'CHD', 'STT', 'WMB', 'WST', 'FRC', 'MKTX', 'MXIM', 'PPL', 'KEYS', 'SLB', 'AJG', 'WY', 'VRSN', 'ETR', 'DFS', 'LEN', 'LH', 'AVB', 'MPC', 'PSX', 'KHC', 'AMP', 'RSG', 'EOG', 'DAL', 'O', 'AEE', 'ODFL', 'CDW', 'TTWO', 'FLT', 'HOLX', 'PAYC', 'SPG', 'LYB', 'ZBRA', 'ARE', 'IP', 'AMCR', 'CMS', 'EFX', 'VMC', 'EQR', 'GWW', 'ETSY', 'CBRE', 'CAG', 'LVS', 'KSU', 'NTRS', 'CTLT', 'DOV', 'TSN', 'DGX', 'QRVO', 'BR', 'TER', 'GRMN', 'TYL', 'VIAC', 'FITB', 'AKAM', 'COO', 'TSCO', 'XYL', 'SIVB', 'K', 'MLM', 'FTNT', 'VAR', 'FE', 'VLO', 'PKI', 'TFX', 'INCY', 'STE', 'DPZ', 'CAH', 'POOL', 'NDAQ', 'ABC', 'KMX', 'MAS', 'PEAK', 'DRE', 'IR', 'ESS', 'EXPD', 'CTXS', 'NUE', 'VTR', 'EXR', 'NVR', 'SYF', 'ANET', 'HIG', 'CE', 'FMC', 'TIF', 'EXPE', 'MAA', 'HRL', 'URI', 'WAT', 'BIO', 'GPC', 'BF-B', 'IEX', 'AES', 'DRI', 'SJM', 'IT', 'LNT', 'PXD', 'J', 'MTB', 'KEY', 'WHR', 'RF', 'EVRG', 'CNP', 'TDY', 'AVY', 'CHRW', 'ABMD', 'FBHS', 'ULTA', 'NLOK', 'LDOS', 'JKHY', 'ALB', 'OKE', 'STX', 'HPE', 'PHM', 'CFG', 'WDC', 'EMN', 'PKG', 'IFF', 'ATO', 'CINF', 'WAB', 'AAP', 'HAL', 'HAS', 'PFG', 'OMC', 'RCL', 'NTAP', 'BXP', 'JBHT', 'HBAN', 'BKR', 'UAL', 'XRAY', 'WRK', 'HES', 'UDR', 'CPB', 'LW', 'ALLE', 'RJF', 'LKQ', 'PNW', 'CBOE', 'PNR', 'UHS', 'MGM', 'PWR', 'L', 'TXT', 'FOXA', 'SNA', 'BWA', 'ROL', 'HSIC', 'WRB', 'LUMN', 'FFIV', 'NI', 'WU', 'CXO', 'RE', 'OXY', 'GL', 'LYV', 'WYNN', 'AIZ', 'IRM', 'NRG', 'MYL', 'LB', 'DVA', 'HST', 'AOS', 'IPG', 'HWM', 'MHK', 'NWL', 'TAP', 'CCL', 'IPGP', 'JNPR', 'DISH', 'TPR', 'SEE', 'COG', 'CMA', 'HII', 'LNC', 'PRGO', 'RHI', 'MOS', 'CF', 'DISCK', 'AAL', 'NWSA', 'REG', 'LEG', 'ZION', 'BEN', 'IVZ', 'NLSN', 'VNO', 'FRT', 'ALK', 'FLIR', 'DXC', 'HBI', 'PBCT', 'GPS', 'NCLH', 'KIM', 'PVH', 'FOX', 'FANG', 'VNT', 'AIV', 'DVN', 'UNM', 'FLS', 'XRX', 'NOV', 'RL', 'MRO', 'DISCA', 'SLG', 'APA', 'UAA', 'HFC', 'UA', 'FTI', 'NWS']
#sp500_adjclose=yf.download(tickers=sp500_symlist, start=startdate, group_by='ticker', threads=True)
#sp500_adjclose=sp500_adjclose.loc[:, (slice(None), 'Adj Close')]
#sp500_adjclose.columns = [a for a, b in sp500_adjclose.columns]
#sp500_adjclose=sp500_adjclose.dropna(how='all')
#sp500_adjclose.to_csv('C:/Users/Administrator/Desktop/sp500_adjclose_since2013.csv')
sp500_adjclose=pd.read_csv('C:/Users/Administrator/Desktop/sp500_adjclose_since2013.csv')
sp500_adjclose=sp500_adjclose.set_index(sp500_adjclose.Date)
sp500_adjclose=sp500_adjclose.drop('Date', axis=1)
sp500_adjclose_valid=sp500_adjclose.dropna(axis=1)
sp500_adjclose_valid.index=pd.to_datetime(sp500_adjclose_valid.index)
sp500_index = yf.download('^GSPC', start=startdate)['Adj Close']
sp500_adjclose_test, sp500_adjclose_train = np.split(sp500_adjclose_valid, [int(.25*len(sp500_adjclose_valid))])
result=sp500_adjclose_valid.iloc[1:].copy(True)
result.iloc[:,:]=0
for j in range(0,len(result.iloc[0])):
    pricer = Long_Short_Comb(sp500_adjclose_train.iloc[:,j], sp500_index)
    result.iloc[:,j]=pricer.get_estimate_series()

#test set block start
sp500_ret_test=np.log(sp500_adjclose_test/sp500_adjclose_test.shift(1)).dropna()
sp500_ret_test_mean=sp500_ret_test.mean()
result_test=result.iloc[0:len(sp500_ret_test)]
result_test_mean=result_test.mean()
ret_equalsign=(result_test_mean*sp500_ret_test_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#0.9529914529914529
sp500_ret_test_std=sp500_ret_test.std()
result_test_std=result_test.std()
print((result_test_std/sp500_ret_test_std).mean())
#0.5520235325326173
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_test.iloc[:,j], sp500_ret_test.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.100427
corr_test=[0]*len(result.iloc[0])
for j in range(0,len(corr_test)):
    corr_test[j]=sp500_ret_test.iloc[:,j].corr(result_test.iloc[:,j])
corr_test=pd.DataFrame(corr_test)
print(corr_test.mean())
#0.549264
print(corr_test.std())
#0.123373
#test set block end

#training set block start
sp500_ret_train=np.log(sp500_adjclose_train/sp500_adjclose_train.shift(1)).dropna()
sp500_ret_train_mean=sp500_ret_train.mean()
result_train=result.iloc[len(result)-len(sp500_ret_train):]
result_train_mean=result_train.mean()
ret_equalsign=(result_train_mean*sp500_ret_train_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#1.0
sp500_ret_train_std=sp500_ret_train.std()
result_train_std=result_train.std()
print((result_train_std/sp500_ret_train_std).mean())
#0.6149652931139756
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_train.iloc[:,j], sp500_ret_train.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.0
corr_train=[0]*len(result.iloc[0])
for j in range(0,len(corr_train)):
    corr_train[j]=sp500_ret_train.iloc[:,j].corr(result_train.iloc[:,j])
corr_train=pd.DataFrame(corr_train)
print(corr_train.mean())
#0.614965
print(corr_train.std())
#0.105157
#training set block end

#russell2000_symlist = ['AAN', 'AAOI', 'AAON', 'AAT', 'AAWW', 'AAXN', 'ABCB', 'ABEO', 'ABG', 'ABM', 'ABTX', 'AC', 'ACA', 'ACAD', 'ACBI', 'ACCO', 'ACER', 'ACHN', 'ACIA', 'ACIW', 'ACLS', 'ACNB', 'ACOR', 'ACRE', 'ACRS', 'ACRX', 'ACTG', 'ADC', 'ADES', 'ADMA', 'ADMS', 'ADNT', 'ADRO', 'ADSW', 'ADTN', 'ADUS', 'ADVM', 'AEGN', 'AEIS', 'AEL', 'AEO', 'AERI', 'AFI', 'AFIN', 'AFMD', 'AGE', 'AGEN', 'AGLE', 'AGM', 'AGS', 'AGX', 'AGYS', 'AHH', 'AHT', 'AI', 'AIMC', 'AIMT', 'AIN', 'AIR', 'AIRG', 'AIT', 'AJRD', 'AJX', 'AKBA', 'AKCA', 'AKR', 'AKRO', 'AKRX', 'AKS', 'AKTS', 'ALBO', 'ALCO', 'ALDR', 'ALDX', 'ALE', 'ALEC', 'ALEX', 'ALG', 'ALGT', 'ALLK', 'ALLO', 'ALOT', 'ALRM', 'ALTM', 'ALTR', 'ALX', 'AMAG', 'AMAL', 'AMBA', 'AMBC', 'AMC', 'AMED', 'AMEH', 'AMK', 'AMKR', 'AMN', 'AMNB', 'AMOT', 'AMPH', 'AMRC', 'AMRS', 'AMRX', 'AMSC', 'AMSF', 'AMSWA', 'AMTB', 'AMWD', 'ANAB', 'ANDE', 'ANF', 'ANGO', 'ANH', 'ANIK', 'ANIP', 'AOBC', 'AOSL', 'APAM', 'APEI', 'APLS', 'APOG', 'APPF', 'APPN', 'APPS', 'APTS', 'APYX', 'AQUA', 'ARA', 'ARAY', 'ARCB', 'ARCH', 'ARDX', 'ARES', 'ARGO', 'ARI', 'ARL', 'ARLO', 'ARNA', 'AROC', 'AROW', 'ARQL', 'ARR', 'ARTNA', 'ARVN', 'ARWR', 'ASC', 'ASGN', 'ASIX', 'ASMB', 'ASNA', 'ASPS', 'ASRT', 'ASTE', 'AT', 'ATEC', 'ATEN', 'ATEX', 'ATGE', 'ATHX', 'ATI', 'ATKR', 'ATLO', 'ATNI', 'ATNX', 'ATRA', 'ATRC', 'ATRI', 'ATRO', 'ATRS', 'ATSG', 'AUB', 'AVA', 'AVAV', 'AVCO', 'AVD', 'AVDR', 'AVID', 'AVNS', 'AVRO', 'AVX', 'AVXL', 'AVYA', 'AWR', 'AX', 'AXAS', 'AXDX', 'AXE', 'AXGN', 'AXL', 'AXLA', 'AXNX', 'AXSM', 'AXTI', 'AYR', 'AZZ', 'B', 'BANC', 'BAND', 'BANF', 'BANR', 'BATRA', 'BATRK', 'BBBY', 'BBCP', 'BBIO', 'BBSI', 'BBX', 'BCBP', 'BCC', 'BCEI', 'BCEL', 'BCML', 'BCO', 'BCOR', 'BCOV', 'BCPC', 'BCRX', 'BDC', 'BDGE', 'BDSI', 'BE', 'BEAT', 'BECN', 'BELFB', 'BFC', 'BFIN', 'BFS', 'BFST', 'BGG', 'BGS', 'BGSF', 'BH', 'BHB', 'BHE', 'BHLB', 'BHR', 'BHVN', 'BIG', 'BIOS', 'BJ', 'BJRI', 'BKD', 'BKE', 'BKH', 'BL', 'BLBD', 'BLD', 'BLDR', 'BLFS', 'BLKB', 'BLMN', 'BLX', 'BMCH', 'BMI', 'BMRC', 'BMTC', 'BNED', 'BNFT', 'BOCH', 'BOLD', 'BOMN', 'BOOM', 'BOOT', 'BOX', 'BPFH', 'BPMC', 'BPRN', 'BRC', 'BREW', 'BRG', 'BRID', 'BRKL', 'BRKS', 'BRT', 'BRY', 'BSET', 'BSGM', 'BSIG', 'BSRR', 'BSTC', 'BSVN', 'BTAI', 'BTU', 'BUSE', 'BV', 'BWB', 'BWFG', 'BXC', 'BXG', 'BXMT', 'BXS', 'BY', 'BYD', 'BYSI', 'BZH', 'CAC', 'CADE', 'CAI', 'CAKE', 'CAL', 'CALA', 'CALM', 'CALX', 'CAMP', 'CAR', 'CARA', 'CARB', 'CARE', 'CARG', 'CARO', 'CARS', 'CASA', 'CASH', 'CASI', 'CASS', 'CATC', 'CATM', 'CATO', 'CATS', 'CATY', 'CBAN', 'CBAY', 'CBB', 'CBL', 'CBLK', 'CBM', 'CBMG', 'CBNK', 'CBPX', 'CBRL', 'CBTX', 'CBU', 'CBZ', 'CCB', 'CCBG', 'CCF', 'CCMP', 'CCNE', 'CCO', 'CCOI', 'CCRN', 'CCS', 'CCXI', 'CDE', 'CDLX', 'CDMO', 'CDNA', 'CDR', 'CDXC', 'CDXS', 'CDZI', 'CECE', 'CECO', 'CEIX', 'CELC', 'CELH', 'CENT', 'CENTA', 'CENX', 'CERC', 'CERS', 'CETV', 'CEVA', 'CFB', 'CFFI', 'CFFN', 'CFMS', 'CHAP', 'CHCO', 'CHCT', 'CHDN', 'CHEF', 'CHGG', 'CHMA', 'CHMG', 'CHMI', 'CHRA', 'CHRS', 'CHS', 'CHUY', 'CIA', 'CIO', 'CIR', 'CISN', 'CIVB', 'CIX', 'CJ', 'CKH', 'CKPT', 'CLAR', 'CLBK', 'CLCT', 'CLDR', 'CLDT', 'CLF', 'CLFD', 'CLI', 'CLNC', 'CLNE', 'CLPR', 'CLVS', 'CLW', 'CLXT', 'CMBM', 'CMC', 'CMCO', 'CMCT', 'CMLS', 'CMO', 'CMP', 'CMPR', 'CMRE', 'CMRX', 'CMTL', 'CNBKA', 'CNCE', 'CNDT', 'CNMD', 'CNNE', 'CNO', 'CNOB', 'CNR', 'CNS', 'CNSL', 'CNST', 'CNTY', 'CNX', 'CNXN', 'CODA', 'COHU', 'COKE', 'COLB', 'COLL', 'CONN', 'COOP', 'CORE', 'CORR', 'CORT', 'COWN', 'CPE', 'CPF', 'CPK', 'CPLG', 'CPRX', 'CPS', 'CPSI', 'CRAI', 'CRBP', 'CRC', 'CRCM', 'CRD-A', 'CRK', 'CRMD', 'CRMT', 'CRNX', 'CROX', 'CRS', 'CRTX', 'CRUS', 'CRVL', 'CRY', 'CRZO', 'CSFL', 'CSGS', 'CSII', 'CSLT', 'CSOD', 'CSTE', 'CSTL', 'CSTR', 'CSV', 'CSWI', 'CTB', 'CTBI', 'CTMX', 'CTO', 'CTRA', 'CTRC', 'CTRE', 'CTRN', 'CTS', 'CTSO', 'CTT', 'CTWS', 'CUB', 'CUBI', 'CUE', 'CULP', 'CURO', 'CUTR', 'CVA', 'CVBF', 'CVCO', 'CVCY', 'CVGI', 'CVGW', 'CVI', 'CVIA', 'CVLT', 'CVLY', 'CVM', 'CVRS', 'CVTI', 'CWCO', 'CWEN', 'CWEN-A', 'CWH', 'CWK', 'CWST', 'CWT', 'CXW', 'CYCN', 'CYH', 'CYRX', 'CYTK', 'CZNC', 'DAKT', 'DAN', 'DAR', 'DBD', 'DBI', 'DCO', 'DCOM', 'DCPH', 'DDD', 'DDS', 'DEA', 'DECK', 'DENN', 'DERM', 'DF', 'DFIN', 'DGICA', 'DGII', 'DHIL', 'DHT', 'DHX', 'DIN', 'DIOD', 'DJCO', 'DK', 'DLA', 'DLTH', 'DLX', 'DMRC', 'DNBF', 'DNLI', 'DNOW', 'DNR', 'DO', 'DOC', 'DOMO', 'DOOR', 'DORM', 'DOVA', 'DPLO', 'DRH', 'DRNA', 'DRQ', 'DS', 'DSKE', 'DSPG', 'DSSI', 'DTIL', 'DVAX', 'DX', 'DXPE', 'DY', 'DZSI', 'EAT', 'EB', 'EBF', 'EBIX', 'EBS', 'EBSB', 'EBTC', 'ECHO', 'ECOL', 'ECOM', 'ECOR', 'ECPG', 'EDIT', 'EE', 'EEX', 'EFC', 'EFSC', 'EGAN', 'EGBN', 'EGHT', 'EGLE', 'EGOV', 'EGP', 'EGRX', 'EHTH', 'EIDX', 'EIG', 'EIGI', 'EIGR', 'ELF', 'ELOX', 'ELVT', 'ELY', 'EME', 'EML', 'ENDP', 'ENFC', 'ENOB', 'ENPH', 'ENS', 'ENSG', 'ENTA', 'ENV', 'ENVA', 'ENZ', 'EOLS', 'EPAY', 'EPC', 'EPM', 'EPRT', 'EPZM', 'EQBK', 'ERA', 'ERI', 'ERII', 'EROS', 'ESCA', 'ESE', 'ESGR', 'ESNT', 'ESPR', 'ESQ', 'ESSA', 'ESTE', 'ESXB', 'ETH', 'ETM', 'EVBG', 'EVBN', 'EVC', 'EVER', 'EVFM', 'EVH', 'EVI', 'EVLO', 'EVOP', 'EVRI', 'EVTC', 'EXLS', 'EXPI', 'EXPO', 'EXPR', 'EXTN', 'EXTR', 'EYE', 'EYPT', 'EZPW', 'FARM', 'FARO', 'FATE', 'FBC', 'FBIZ', 'FBK', 'FBM', 'FBMS', 'FBNC', 'FBP', 'FC', 'FCAP', 'FCBC', 'FCBP', 'FCCY', 'FCF', 'FCFS', 'FCN', 'FCPT', 'FDBC', 'FDEF', 'FDP', 'FELE', 'FET', 'FF', 'FFBC', 'FFG', 'FFIC', 'FFIN', 'FFNW', 'FFWM', 'FG', 'FGBI', 'FGEN', 'FI', 'FIBK', 'FII', 'FISI', 'FIT', 'FIVN', 'FIX', 'FIXX', 'FIZZ', 'FLDM', 'FLIC', 'FLMN', 'FLNT', 'FLOW', 'FLWS', 'FLXN', 'FLXS', 'FMAO', 'FMBH', 'FMBI', 'FMNB', 'FN', 'FNCB', 'FNHC', 'FNKO', 'FNLC', 'FNWB', 'FOCS', 'FOE', 'FOLD', 'FOR', 'FORM', 'FORR', 'FOSL', 'FOXF', 'FPI', 'FPRX', 'FR', 'FRAC', 'FRAF', 'FRBA', 'FRBK', 'FRGI', 'FRME', 'FRPH', 'FRPT', 'FRTA', 'FSB', 'FSBW', 'FSCT', 'FSP', 'FSS', 'FSTR', 'FTK', 'FTR', 'FTSI', 'FTSV', 'FUL', 'FULC', 'FULT', 'FVCB', 'FWRD', 'GABC', 'GAIA', 'GALT', 'GATX', 'GBCI', 'GBL', 'GBLI', 'GBT', 'GBX', 'GCAP', 'GCBC', 'GCI', 'GCO', 'GCP', 'GDEN', 'GDOT', 'GDP', 'GEF', 'GEF-B', 'GEN', 'GENC', 'GEO', 'GEOS', 'GERN', 'GES', 'GFF', 'GFN', 'GHDX', 'GHL', 'GHM', 'GIII', 'GKOS', 'GLDD', 'GLNG', 'GLOG', 'GLRE', 'GLT', 'GLUU', 'GLYC', 'GME', 'GMED', 'GMRE', 'GMS', 'GNC', 'GNE', 'GNK', 'GNL', 'GNLN', 'GNMK', 'GNRC', 'GNTY', 'GNW', 'GOGO', 'GOLF', 'GOOD', 'GORO', 'GOSS', 'GPI', 'GPMT', 'GPOR', 'GPRE', 'GPRO', 'GPX', 'GRBK', 'GRC', 'GRIF', 'GRPN', 'GRTS', 'GSBC', 'GSHD', 'GSIT', 'GTHX', 'GTLS', 'GTN', 'GTS', 'GTT', 'GTY', 'GTYH', 'GVA', 'GWB', 'GWGH', 'GWRS', 'HA', 'HABT', 'HAE', 'HAFC', 'HALL', 'HALO', 'HARP', 'HASI', 'HAYN', 'HBB', 'HBCP', 'HBMD', 'HBNC', 'HCAT', 'HCC', 'HCCI', 'HCI', 'HCKT', 'HCSG', 'HEES', 'HELE', 'HFFG', 'HFWA', 'HI', 'HIBB', 'HIFS', 'HIIQ', 'HL', 'HLI', 'HLIO', 'HLIT', 'HLNE', 'HLX', 'HMHC', 'HMN', 'HMST', 'HMSY', 'HMTV', 'HNGR', 'HNI', 'HNRG', 'HOFT', 'HOMB', 'HOME', 'HONE', 'HOOK', 'HOPE', 'HPR', 'HQY', 'HR', 'HRI', 'HRTG', 'HRTX', 'HSC', 'HSII', 'HSKA', 'HSTM', 'HT', 'HTBI', 'HTBK', 'HTH', 'HTLD', 'HTLF', 'HTZ', 'HUBG', 'HUD', 'HURC', 'HURN', 'HVT', 'HWBK', 'HWC', 'HWKN', 'HY', 'HZO', 'I', 'IBCP', 'IBKC', 'IBOC', 'IBP', 'IBTX', 'ICD', 'ICFI', 'ICHR', 'ICPT', 'IDCC', 'IDEX', 'IDT', 'IESC', 'IHC', 'III', 'IIIN', 'IIIV', 'IIN', 'IIPR', 'IIVI', 'ILPT', 'IMAX', 'IMGN', 'IMKTA', 'IMMR', 'IMMU', 'IMXI', 'INBK', 'INDB', 'INFN', 'INGN', 'INN', 'INO', 'INOV', 'INS', 'INSE', 'INSG', 'INSM', 'INSP', 'INST', 'INSW', 'INT', 'INTL', 'INVA', 'INWK', 'IOSP', 'IOTS', 'IOVA', 'IPAR', 'IPHI', 'IPHS', 'IPI', 'IRBT', 'IRDM', 'IRET', 'IRMD', 'IRT', 'IRTC', 'IRWD', 'ISBC', 'ISCA', 'ISRL', 'ISTR', 'ITCI', 'ITGR', 'ITI', 'ITIC', 'ITRI', 'IVC', 'IVR', 'JACK', 'JAG', 'JAX', 'JBSS', 'JBT', 'JCAP', 'JCOM', 'JCP', 'JELD', 'JILL', 'JJSF', 'JNCE', 'JOE', 'JOUT', 'JRVR', 'JYNT', 'KAI', 'KALA', 'KALU', 'KALV', 'KAMN', 'KBAL', 'KBH', 'KBR', 'KDMN', 'KE', 'KELYA', 'KEM', 'KFRC', 'KFY', 'KIDS', 'KIN', 'KLDO', 'KLXE', 'KMT', 'KN', 'KNL', 'KNSA', 'KNSL', 'KOD', 'KOP', 'KPTI', 'KRA', 'KREF', 'KRG', 'KRNY', 'KRO', 'KRTX', 'KRUS', 'KRYS', 'KTB', 'KTOS', 'KURA', 'KVHI', 'KW', 'KWR', 'KZR', 'LAD', 'LADR', 'LANC', 'LAND', 'LASR', 'LAUR', 'LAWS', 'LBAI', 'LBC', 'LBRT', 'LC', 'LCI', 'LCII', 'LCNB', 'LCTX', 'LCUT', 'LDL', 'LE', 'LEAF', 'LEE', 'LEGH', 'LEVL', 'LFVN', 'LGIH', 'LGND', 'LHCG', 'LILA', 'LILAK', 'LIND', 'LITE', 'LIVN', 'LIVX', 'LJPC', 'LKFN', 'LKSD', 'LL', 'LLNW', 'LMAT', 'LMNR', 'LMNX', 'LNDC', 'LNN', 'LNTH', 'LOB', 'LOCO', 'LOGC', 'LORL', 'LOVE', 'LPG', 'LPI', 'LPSN', 'LPX', 'LQDA', 'LQDT', 'LRN', 'LSCC', 'LTC', 'LTHM', 'LTRPA', 'LTS', 'LTXB', 'LVGO', 'LXFR', 'LXP', 'LXRX', 'LXU', 'LZB', 'MANT', 'MATW', 'MATX', 'MAXR', 'MBI', 'MBII', 'MBIN', 'MBIO', 'MBUU', 'MBWM', 'MC', 'MCB', 'MCBC', 'MCFT', 'MCHX', 'MCRB', 'MCRI', 'MCRN', 'MCS', 'MDC', 'MDCA', 'MDCO', 'MDGL', 'MDP', 'MDR', 'MDRX', 'MEC', 'MED', 'MEDP', 'MEET', 'MEI', 'MEIP', 'MESA', 'METC', 'MFIN', 'MFNC', 'MFSF', 'MG', 'MGEE', 'MGLN', 'MGNX', 'MGPI', 'MGRC', 'MGTA', 'MGTX', 'MGY', 'MHO', 'MIK', 'MINI', 'MIRM', 'MITK', 'MITT', 'MJCO', 'MLAB', 'MLHR', 'MLI', 'MLND', 'MLP', 'MLR', 'MLVF', 'MMAC', 'MMI', 'MMS', 'MMSI', 'MNK', 'MNKD', 'MNLO', 'MNOV', 'MNR', 'MNRL', 'MNRO', 'MNSB', 'MNTA', 'MOBL', 'MOD', 'MODN', 'MOFG', 'MOG-A', 'MORF', 'MOV', 'MPAA', 'MPB', 'MPX', 'MR', 'MRC', 'MRCY', 'MRKR', 'MRLN', 'MRNS', 'MRSN', 'MRTN', 'MRTX', 'MSA', 'MSBI', 'MSEX', 'MSGN', 'MSON', 'MSTR', 'MTDR', 'MTEM', 'MTH', 'MTOR', 'MTRN', 'MTRX', 'MTSC', 'MTSI', 'MTW', 'MTX', 'MTZ', 'MUSA', 'MVBF', 'MWA', 'MXL', 'MYE', 'MYGN', 'MYOK', 'MYRG', 'NANO', 'NAT', 'NATH', 'NATR', 'NAV', 'NBEV', 'NBHC', 'NBN', 'NBR', 'NBTB', 'NC', 'NCBS', 'NCI', 'NCMI', 'NCSM', 'NDLS', 'NE', 'NEO', 'NEOG', 'NERV', 'NESR', 'NEWM', 'NEXT', 'NFBK', 'NG', 'NGHC', 'NGM', 'NGS', 'NGVC', 'NGVT', 'NHC', 'NHI', 'NINE', 'NJR', 'NKSH', 'NL', 'NMIH', 'NMRK', 'NNBR', 'NNI', 'NODK', 'NOG', 'NOVA', 'NOVT', 'NP', 'NPK', 'NPO', 'NPTN', 'NR', 'NRC', 'NRCG', 'NRIM', 'NSA', 'NSIT', 'NSP', 'NSSC', 'NSTG', 'NTB', 'NTCT', 'NTGN', 'NTGR', 'NTLA', 'NTRA', 'NTUS', 'NUVA', 'NVAX', 'NVCR', 'NVEC', 'NVEE', 'NVRO', 'NVTA', 'NWBI', 'NWE', 'NWFL', 'NWLI', 'NWN', 'NWPX', 'NX', 'NXGN', 'NXRT', 'NXTC', 'NYMT', 'NYNY', 'OAS', 'OBNK', 'OCFC', 'OCN', 'OCUL', 'OCX', 'ODC', 'ODP', 'ODT', 'OEC', 'OFG', 'OFIX', 'OFLX', 'OGS', 'OII', 'OIS', 'OLBK', 'OLP', 'OMCL', 'OMER', 'OMI', 'OMN', 'ONB', 'ONCE', 'ONDK', 'OOMA', 'OPB', 'OPBK', 'OPI', 'OPK', 'OPRX', 'OPTN', 'OPY', 'ORA', 'ORBC', 'ORC', 'ORGO', 'ORIT', 'ORRF', 'OSBC', 'OSG', 'OSIS', 'OSMT', 'OSPN', 'OSTK', 'OSUR', 'OSW', 'OTTR', 'OVBC', 'OVLY', 'OXM', 'PACB', 'PACD', 'PAHC', 'PAR', 'PARR', 'PATK', 'PAYS', 'PBFS', 'PBH', 'PBI', 'PBIP', 'PBPB', 'PBYI', 'PCB', 'PCH', 'PCRX', 'PCSB', 'PCYO', 'PDCE', 'PDCO', 'PDFS', 'PDLB', 'PDLI', 'PDM', 'PEB', 'PEBK', 'PEBO', 'PEGI', 'PEI', 'PENN', 'PETQ', 'PETS', 'PFBC', 'PFBI', 'PFGC', 'PFIS', 'PFNX', 'PFS', 'PFSI', 'PGC', 'PGNX', 'PGTI', 'PHAS', 'PHR', 'PHUN', 'PHX', 'PI', 'PICO', 'PIRS', 'PJC', 'PJT', 'PKBK', 'PKD', 'PKE', 'PKOH', 'PLAB', 'PLAY', 'PLCE', 'PLMR', 'PLOW', 'PLPC', 'PLSE', 'PLT', 'PLUG', 'PLUS', 'PLXS', 'PMBC', 'PMT', 'PNM', 'PNRG', 'PNTG', 'POL', 'POR', 'POWI', 'POWL', 'PPBI', 'PQG', 'PRA', 'PRAA', 'PRFT', 'PRGS', 'PRGX', 'PRIM', 'PRK', 'PRLB', 'PRMW', 'PRNB', 'PRO', 'PROS', 'PROV', 'PRPL', 'PRSC', 'PRSP', 'PRTA', 'PRTH', 'PRTK', 'PRTY', 'PRVL', 'PSB', 'PSDO', 'PSMT', 'PSN', 'PSNL', 'PTCT', 'PTE', 'PTGX', 'PTLA', 'PTN', 'PTSI', 'PTVCB', 'PUB', 'PUMP', 'PVAC', 'PVBC', 'PWOD', 'PYX', 'PZN', 'PZZA', 'QADA', 'QCRH', 'QDEL', 'QEP', 'QLYS', 'QNST', 'QTRX', 'QTS', 'QTWO', 'QUAD', 'QUOT', 'RAD', 'RAMP', 'RARE', 'RARX', 'RAVN', 'RBB', 'RBBN', 'RBCAA', 'RBNC', 'RC', 'RCII', 'RCKT', 'RCKY', 'RCM', 'RCUS', 'RDFN', 'RDI', 'RDN', 'RDNT', 'RDUS', 'REAL', 'RECN', 'REGI', 'REI', 'REPH', 'REPL', 'RES', 'RESI', 'RETA', 'REV', 'REVG', 'REX', 'REXR', 'RFL', 'RGCO', 'RGEN', 'RGNX', 'RGR', 'RGS', 'RH', 'RHP', 'RICK', 'RIGL', 'RILY', 'RLGT', 'RLGY', 'RLH', 'RLI', 'RLJ', 'RM', 'RMAX', 'RMBI', 'RMBS', 'RMNI', 'RMR', 'RMTI', 'RNET', 'RNST', 'ROAD', 'ROAN', 'ROCK', 'ROG', 'ROIC', 'ROLL', 'ROSE', 'RPD', 'RPT', 'RRBI', 'RRD', 'RRGB', 'RRR', 'RRTS', 'RST', 'RTEC', 'RTIX', 'RTRX', 'RTW', 'RUBI', 'RUBY', 'RUN', 'RUSHA', 'RUSHB', 'RUTH', 'RVI', 'RVNC', 'RVSB', 'RWT', 'RXN', 'RYAM', 'RYI', 'RYTM', 'SAFE', 'SAFM', 'SAFT', 'SAH', 'SAIA', 'SAIC', 'SAIL', 'SALT', 'SAM', 'SAMG', 'SANM', 'SASR', 'SAVE', 'SB', 'SBBP', 'SBBX', 'SBCF', 'SBH', 'SBOW', 'SBRA', 'SBSI', 'SBT', 'SCHL', 'SCHN', 'SCL', 'SCOR', 'SCS', 'SCSC', 'SCU', 'SCVL', 'SCWX', 'SD', 'SDRL', 'SEAS', 'SEM', 'SEMG', 'SENEA', 'SENS', 'SF', 'SFBS', 'SFE', 'SFIX', 'SFL', 'SFNC', 'SFST', 'SGA', 'SGC', 'SGH', 'SGMO', 'SGMS', 'SGRY', 'SHAK', 'SHBI', 'SHEN', 'SHO', 'SHOO', 'SHSP', 'SIBN', 'SIC', 'SIEB', 'SIEN', 'SIG', 'SIGA', 'SIGI', 'SILK', 'SITE', 'SJI', 'SJW', 'SKT', 'SKY', 'SKYW', 'SLAB', 'SLCA', 'SLCT', 'SLDB', 'SLP', 'SM', 'SMBC', 'SMBK', 'SMHI', 'SMMF', 'SMP', 'SMPL', 'SMTC', 'SNBR', 'SNCR', 'SND', 'SNDX', 'SNH', 'SNR', 'SOI', 'SOLY', 'SONA', 'SONM', 'SONO', 'SP', 'SPAR', 'SPFI', 'SPKE', 'SPNE', 'SPOK', 'SPPI', 'SPRO', 'SPSC', 'SPTN', 'SPWH', 'SPWR', 'SPXC', 'SR', 'SRCE', 'SRCI', 'SRDX', 'SRG', 'SRI', 'SRNE', 'SRRK', 'SRT', 'SSB', 'SSD', 'SSP', 'SSTI', 'SSTK', 'SSYS', 'STAA', 'STAG', 'STAR', 'STBA', 'STC', 'STFC', 'STIM', 'STML', 'STMP', 'STNG', 'STOK', 'STRA', 'STRL', 'STRO', 'STRS', 'STXB', 'SUM', 'SUPN', 'SVMK', 'SVRA', 'SWAV', 'SWM', 'SWN', 'SWX', 'SXC', 'SXI', 'SXT', 'SYBT', 'SYBX', 'SYKE', 'SYNA', 'SYNH', 'SYNL', 'SYRS', 'SYX', 'TACO', 'TALO', 'TAST', 'TBBK', 'TBI', 'TBIO', 'TBK', 'TBNK', 'TBPH', 'TCBK', 'TCDA', 'TCFC', 'TCI', 'TCMD', 'TCRR', 'TCS', 'TCX', 'TDOC', 'TDW', 'TECD', 'TELL', 'TEN', 'TENB', 'TERP', 'TESS', 'TEUM', 'TEX', 'TG', 'TGH', 'TGI', 'TGNA', 'TGTX', 'TH', 'THC', 'THFF', 'THOR', 'THR', 'THRM', 'TILE', 'TIPT', 'TISI', 'TITN', 'TIVO', 'TK', 'TLRA', 'TLRD', 'TLYS', 'TMDX', 'TMHC', 'TMP', 'TMST', 'TNAV', 'TNC', 'TNDM', 'TNET', 'TNK', 'TOCA', 'TORC', 'TOWN', 'TPB', 'TPC', 'TPCO', 'TPH', 'TPIC', 'TPRE', 'TPTX', 'TR', 'TRC', 'TREC', 'TREX', 'TRHC', 'TRMK', 'TRNO', 'TRNS', 'TROX', 'TRS', 'TRST', 'TRTN', 'TRTX', 'TRUE', 'TRUP', 'TRWH', 'TRXC', 'TSBK', 'TSC', 'TSE', 'TTEC', 'TTEK', 'TTGT', 'TTI', 'TTMI', 'TTS', 'TUP', 'TUSK', 'TVTY', 'TWI', 'TWIN', 'TWNK', 'TWST', 'TXMD', 'TXRH', 'TYME', 'TYPE', 'TZOO', 'UBA', 'UBFO', 'UBNK', 'UBSI', 'UBX', 'UCBI', 'UCFC', 'UCTT', 'UE', 'UEC', 'UEIC', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UHT', 'UIHC', 'UIS', 'ULH', 'UMBF', 'UMH', 'UNB', 'UNF', 'UNFI', 'UNIT', 'UNT', 'UNTY', 'UPLD', 'UPWK', 'URGN', 'USCR', 'USLM', 'USNA', 'USPH', 'USWS', 'USX', 'UTL', 'UTMD', 'UUUU', 'UVE', 'UVSP', 'UVV', 'VAC', 'VALU', 'VAPO', 'VBIV', 'VBTX', 'VC', 'VCEL', 'VCRA', 'VCYT', 'VEC', 'VECO', 'VG', 'VGR', 'VHC', 'VHI', 'VIAV', 'VICR', 'VIVO', 'VKTX', 'VLGEA', 'VLY', 'VNCE', 'VNDA', 'VPG', 'VRA', 'VRAY', 'VRCA', 'VREX', 'VRNS', 'VRNT', 'VRRM', 'VRS', 'VRTS', 'VRTU', 'VRTV', 'VSEC', 'VSH', 'VSLR', 'VSTO', 'VVI', 'VYGR', 'WAAS', 'WABC', 'WAFD', 'WAIR', 'WASH', 'WATT', 'WBT', 'WD', 'WDFC', 'WDR', 'WERN', 'WETF', 'WEYS', 'WGO', 'WHD', 'WHG', 'WIFI', 'WINA', 'WING', 'WIRE', 'WK', 'WLDN', 'WLFC', 'WLH', 'WLL', 'WMC', 'WMGI', 'WMK', 'WMS', 'WNC', 'WNEB', 'WOR', 'WOW', 'WPG', 'WRE', 'WRLD', 'WRTC', 'WSBC', 'WSBF', 'WSC', 'WSFS', 'WSR', 'WTBA', 'WTI', 'WTRE', 'WTRH', 'WTS', 'WTTR', 'WVE', 'WW', 'WWW', 'XAN', 'XBIT', 'XELA', 'XENT', 'XERS', 'XFOR', 'XHR', 'XLRN', 'XNCR', 'XOG', 'XON', 'XPER', 'XXII', 'YCBD', 'YELP', 'YETI', 'YEXT', 'YGYI', 'YMAB', 'YORW', 'YRCW', 'ZAGG', 'ZEUS', 'ZGNX', 'ZIOP', 'ZIXI', 'ZUMZ', 'ZUO', 'ZYNE', 'ZYXI']
#russell2000_adjclose=yf.download(tickers=russell2000_symlist, start=startdate, group_by='ticker', threads=True)
#russell2000_adjclose=russell2000_adjclose.loc[:, (slice(None), 'Adj Close')]
#russell2000_adjclose.columns = [a for a, b in russell2000_adjclose.columns]
#russell2000_adjclose=russell2000_adjclose.dropna(how='all')
#russell2000_adjclose.to_csv('C:/Users/Administrator/Desktop/russell2000_adjclose_since2013.csv')
russell2000_adjclose=pd.read_csv('C:/Users/Administrator/Desktop/russell2000_adjclose_since2013.csv')
russell2000_adjclose=russell2000_adjclose.set_index(russell2000_adjclose.Date)
russell2000_adjclose=russell2000_adjclose.drop('Date', axis=1)
russell2000_adjclose_valid=russell2000_adjclose.dropna(axis=1)
russell2000_adjclose_valid.index=pd.to_datetime(russell2000_adjclose_valid.index)
russell2000_index = yf.download('^RUT', start=startdate)['Adj Close']
russell2000_adjclose_test, russell2000_adjclose_train = np.split(russell2000_adjclose_valid, [int(.25*len(russell2000_adjclose_valid))])
result=russell2000_adjclose_valid.iloc[1:].copy(True)
result.iloc[:,:]=0
for j in range(0,len(result.iloc[0])):
    pricer = Long_Short_Comb(russell2000_adjclose_train.iloc[:,j], russell2000_index)
    result.iloc[:,j]=pricer.get_estimate_series()

#test set block start
russell2000_ret_test=np.log(russell2000_adjclose_test/russell2000_adjclose_test.shift(1)).dropna()
russell2000_ret_test_mean=russell2000_ret_test.mean()
result_test=result.iloc[0:len(russell2000_ret_test)]
result_test_mean=result_test.mean()
ret_equalsign=(result_test_mean*russell2000_ret_test_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#0.7554655870445344
russell2000_ret_test_std=russell2000_ret_test.std()
result_test_std=result_test.std()
print((result_test_std/russell2000_ret_test_std).mean())
#0.5256578142378369
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_test.iloc[:,j], russell2000_ret_test.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.161943
corr_test=[0]*len(result.iloc[0])
for j in range(0,len(corr_test)):
    corr_test[j]=russell2000_ret_test.iloc[:,j].corr(result_test.iloc[:,j])
corr_test=pd.DataFrame(corr_test)
print(corr_test.mean())
#0.43337
print(corr_test.std())
#0.179558
#test set block end

#training set block start
russell2000_ret_train=np.log(russell2000_adjclose_train/russell2000_adjclose_train.shift(1)).dropna()
russell2000_ret_train_mean=russell2000_ret_train.mean()
result_train=result.iloc[len(result)-len(russell2000_ret_train):]
result_train_mean=result_train.mean()
ret_equalsign=(result_train_mean*russell2000_ret_train_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#0.9862348178137652
russell2000_ret_train_std=russell2000_ret_train.std()
result_train_std=result_train.std()
print((result_train_std/russell2000_ret_train_std).mean())
#0.5238269618788547
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_train.iloc[:,j], russell2000_ret_train.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.025911
corr_train=[0]*len(result.iloc[0])
for j in range(0,len(corr_train)):
    corr_train[j]=russell2000_ret_train.iloc[:,j].corr(result_train.iloc[:,j])
corr_train=pd.DataFrame(corr_train)
print(corr_train.mean())
#0.523301
print(corr_train.std())
#0.139877
#training set block end

#gsci_symlist = ['KE=F', 'ZC=F', 'ZS=F', 'KC=F', 'SB=F', 'CC=F', 'CT=F', 'HE=F', 'LE=F', 'GF=F', 'CL=F', 'HO=F', 'RB=F', 'BZ=F', 'NG=F', 'HG=F', 'GC=F', 'SI=F']
#gsci_adjclose = yf.download(tickers=gsci_symlist, start=startdate, group_by='ticker', threads=True)
#gsci_adjclose = gsci_adjclose.loc[:, (slice(None), 'Adj Close')]
#gsci_adjclose.columns = [a for a, b in gsci_adjclose.columns]
#gsci_adjclose = gsci_adjclose.dropna(how='all')
#gsci_adjclose.to_csv('C:/Users/Administrator/Desktop/gsci_adjclose_since2013.csv')
gsci_adjclose = pd.read_csv('C:/Users/Administrator/Desktop/gsci_adjclose_since2013.csv')
gsci_adjclose = gsci_adjclose.set_index(gsci_adjclose.Date)
gsci_adjclose = gsci_adjclose.drop('Date', axis=1)
gsci_adjclose.index = pd.to_datetime(gsci_adjclose.index).date
gsci_index = pd.read_csv('C:/Users/Administrator/Desktop/gsci_index_since2013.csv')
gsci_index = gsci_index.set_index(gsci_index.Date)
gsci_index = gsci_index.drop('Date', axis=1)
gsci_index.index=pd.to_datetime(gsci_index.index).date
gsci_adjclose_valid =gsci_adjclose.reindex(index = gsci_index.index)
gsci_adjclose_valid =gsci_adjclose_valid.dropna(thresh=len(gsci_adjclose_valid) - 100, axis=1)
gsci_adjclose_valid = gsci_adjclose_valid.dropna()
gsci_index = gsci_index.reindex(index = gsci_adjclose_valid.index)
gsci_adjclose_test, gsci_adjclose_train = np.split(gsci_adjclose_valid, [int(.25*len(gsci_adjclose_valid))])
result = gsci_adjclose_valid.iloc[1:].copy(True)
result.iloc[:,:] = 0
for j in range(0,len(result.iloc[0])):
    pricer = Long_Short_Comb(gsci_adjclose_train.iloc[:,j], gsci_index)
    result.iloc[:,j] = pricer.get_estimate_series()

#test set block start
gsci_ret_test=np.log(gsci_adjclose_test/gsci_adjclose_test.shift(1)).dropna()
gsci_ret_test_mean=gsci_ret_test.mean()
result_test=result.iloc[0:len(gsci_ret_test)]
result_test_mean=result_test.mean()
ret_equalsign=(result_test_mean*gsci_ret_test_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#0.6470588235294118
gsci_ret_test_std=gsci_ret_test.std()
result_test_std=result_test.std()
print((result_test_std/gsci_ret_test_std).mean())
#0.2684598157289675
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_test.iloc[:,j], gsci_ret_test.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.176471
corr_test=[0]*len(result.iloc[0])
for j in range(0,len(corr_test)):
    corr_test[j]=gsci_ret_test.iloc[:,j].corr(result_test.iloc[:,j])
corr_test=pd.DataFrame(corr_test)
print(corr_test.mean())
#0.281362
print(corr_test.std())
#0.269915
#test set block end

#training set block start
gsci_ret_train=np.log(gsci_adjclose_train/gsci_adjclose_train.shift(1)).dropna()
gsci_ret_train_mean=gsci_ret_train.mean()
result_train=result.iloc[len(result)-len(gsci_ret_train):]
result_train_mean=result_train.mean()
ret_equalsign=(result_train_mean*gsci_ret_train_mean)>=0
ret_equalsign_percentage=ret_equalsign.sum()/len(ret_equalsign)
print(ret_equalsign_percentage)
#0.9411764705882353
gsci_ret_train_std=gsci_ret_train.std()
result_train_std=result_train.std()
print((result_train_std/gsci_ret_train_std).mean())
#0.30932647250462414
ks_pvalue=[0]*len(result.iloc[0])
for j in range(0,len(ks_pvalue)):
    ks_pvalue[j]=stats.ks_2samp(result_train.iloc[:,j], gsci_ret_train.iloc[:,j]).pvalue
ks_pvalue_5pct=pd.DataFrame(ks_pvalue)>=0.05
print(ks_pvalue_5pct.sum()/len(ks_pvalue))
#0.117647
corr_train=[0]*len(result.iloc[0])
for j in range(0,len(corr_train)):
    corr_train[j]=gsci_ret_train.iloc[:,j].corr(result_train.iloc[:,j])
corr_train=pd.DataFrame(corr_train)
print(corr_train.mean())
#0.305068
print(corr_train.std())
#0.256469
#training set block end