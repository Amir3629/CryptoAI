import yaml

sample_yaml = """
name: CryptoBot
version: 1.0
dependencies:
  - ccxt
  - ta
"""

data = yaml.safe_load(sample_yaml)
print(data)
