from alpha_vantage.timeseries import TimeSeries
# Your key here
key = 'YKWDWRABSEHDCCMK'
ts = TimeSeries(key)
aapl, meta = ts.get_daily(symbol='AAPL')
print(aapl['2019-09-12'])
print(aapl)

