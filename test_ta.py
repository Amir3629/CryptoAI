from ta.volatility import AverageTrueRangeIndicator

# Sample data
high = [10, 11, 12, 13, 14]
low = [8, 9, 9, 10, 11]
close = [9, 10, 11, 12, 13]

atr_indicator = AverageTrueRangeIndicator(high=high, low=low, close=close, window=3)
atr_values = atr_indicator.average_true_range()
print("ATR Values:", atr_values)
