# test_ta.py

from ta.volatility import TrueRangeIndicator
import pandas as pd

# Sample data
data = {
    'high': [10, 12, 11, 13],
    'low': [8, 9, 10, 11],
    'close': [9, 11, 10, 12]
}
df = pd.DataFrame(data)

# Initialize TrueRangeIndicator
tr_indicator = TrueRangeIndicator(high=df['high'], low=df['low'], close=df['close'])
tr = tr_indicator.true_range()

print(tr)
