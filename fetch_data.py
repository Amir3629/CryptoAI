from dotenv import load_dotenv
import os
import requests

# Load environment variables from the .env file
load_dotenv()

# Retrieve API keys from environment variables
coingecko_key = os.getenv("COINGECKO_API_KEY")
cryptocompare_key = os.getenv("CRYPTOCOMPARE_API_KEY")
alphavantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
ayrshare_key = os.getenv("AYRSHARE_API_KEY")
financial_modeling_prep_key = os.getenv("FINANCIAL_MODELING_PREP_API_KEY")

# Test by printing the keys (you should not print keys in production)
print("CoinGecko API Key:", coingecko_key)
print("CryptoCompare API Key:", cryptocompare_key)

# Example: Fetch data from CoinGecko API
def fetch_market_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 10,
        "page": 1,
        "sparkline": False,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching data:", response.status_code, response.text)

if __name__ == "__main__":
    # Fetch and print the top 10 cryptocurrencies data
    data = fetch_market_data()
    print(data)