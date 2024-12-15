# crypto_bot.py

import os
import pandas as pd
import numpy as np
import yaml
import asyncio
import aiohttp
import time
import logging
import traceback
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import ta
import ccxt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------
# Load Environment
# ----------------------------------------
load_dotenv()

def load_config():
    # Load configurations from config.yaml if present
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

def get_config_value(key_path, default=None):
    keys = key_path.split('.')
    current = config
    for k in keys:
        if k in current:
            current = current[k]
        else:
            env_val = os.getenv(key_path.upper().replace('.', '_'))
            if env_val is not None:
                try:
                    return type(default)(env_val) if default is not None else env_val
                except ValueError:
                    return env_val
            return default
    return current

# Load Environment Variables
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

MEXC_API_KEY = os.getenv("MEXC_API_KEY")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET")

# Configurable Parameters
COINS = get_config_value("trading.coins", ["BTC", "ETH", "BNB"])
USE_DYNAMIC_COINS = get_config_value("trading.dynamic_coins", True)
BASE_SYMBOL = get_config_value("trading.base_symbol", "USDT")
BUY_THRESHOLD = get_config_value("trading.buy_threshold", 1.005)
SELL_THRESHOLD = get_config_value("trading.sell_threshold", 0.995)
TRADE_AMOUNT = get_config_value("trading.trade_amount", 0.001)
MAX_DAILY_TRADES = get_config_value("trading.max_daily_trades", 10)

TEST_SIZE = get_config_value("model.test_size", 0.2)
PARAM_GRID = get_config_value("model.param_grid", {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
})

LOOP_INTERVAL = get_config_value("execution.loop_interval", 60)

PRIMARY_DB = get_config_value("database.primary", "postgresql")
SECONDARY_DB = get_config_value("database.secondary", "mysql")

# Logging Setup
logging.basicConfig(filename='logs\crypto_ai.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Crypto AI Program started.")

# Setup CCXT Client
exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
})

async def fetch_market_data(coin):
    url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={coin}&tsym={BASE_SYMBOL}&limit=200"
    headers = {'Authorization': f'Apikey {CRYPTOCOMPARE_API_KEY}'}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Data' in data and 'Data' in data['Data']:
                        return data['Data']['Data']
                    else:
                        logging.error(f"Unexpected data format for {coin}: {data}")
                        return None
                else:
                    logging.error(f"Failed to fetch data for {coin}: {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Exception fetching data for {coin}: {e}")
            return None

async def fetch_top_coins():
    if not USE_DYNAMIC_COINS or not COINGECKO_API_KEY:
        return COINS
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=3&page=1&sparkline=false"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [coin['symbol'].upper() for coin in data][:3]
                else:
                    logging.warning("Failed to fetch dynamic coins from CoinGecko, using default coins.")
                    return COINS
        except Exception as e:
            logging.error(f"Error fetching dynamic coins: {e}")
            return COINS

def add_technical_indicators(df):
    try:
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
        
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=5).rsi()
        
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd().fillna(0)
        df['MACD_signal'] = macd.macd_signal().fillna(0)
        
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        
        stoch = ta.momentum.StochRSIIndicator(df['close'], window=14, smooth1=3, smooth2=3)
        df['StochRSI'] = stoch.stochrsi()
        
        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"Failed to add indicators: {e}")
        return None

def connect_database(primary=True):
    if primary:
        if PRIMARY_DB.lower() == 'postgresql':
            db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
        else:
            db_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    else:
        if SECONDARY_DB.lower() == 'mysql':
            db_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        else:
            db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
    
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine

def save_to_database(data, table_name):
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        try:
            engine = connect_database(primary=True)
            data.to_sql(table_name, engine, if_exists='replace', index=False)
            logging.info(f"Data saved to {PRIMARY_DB.upper()} table: {table_name}")
            return
        except OperationalError as e:
            logging.error(f"Primary DB error: {e}, retrying...")
            attempts += 1
            time.sleep(3)
    
    # If primary fails, try secondary
    logging.warning("Primary DB failed, trying secondary database.")
    try:
        engine = connect_database(primary=False)
        data.to_sql(table_name, engine, if_exists='replace', index=False)
        logging.info(f"Data saved to {SECONDARY_DB.upper()} table: {table_name}")
    except Exception as e:
        logging.error(f"Error saving to secondary DB: {e}")

def train_and_evaluate(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
        
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=PARAM_GRID, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Model MSE: {mse:.2f}, R2: {r2:.2f}")
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        return best_model
    except Exception as e:
        logging.error(f"Model training failed: {e}\n{traceback.format_exc()}")
        return None

def place_trade(action, coin, amount):
    max_retries = 3
    retries = 0
    symbol = f"{coin}/{BASE_SYMBOL}"
    while retries < max_retries:
        try:
            if action == "buy":
                order = exchange.create_market_buy_order(symbol, amount)
            elif action == "sell":
                order = exchange.create_market_sell_order(symbol, amount)
            else:
                logging.error(f"Invalid trade action: {action}")
                return None
            logging.info(f"Trade executed: {order}")
            return order
        except ccxt.BaseError as e:
            logging.error(f"Failed to execute trade: {e}, retrying...")
            time.sleep(2)
            retries += 1
    logging.error("Trade execution failed after retries.")
    return None

def risk_management_check(current_trades):
    if current_trades >= MAX_DAILY_TRADES:
        logging.warning("Max daily trades limit reached. No trades will be made.")
        return False
    return True

def generate_features_and_target(df):
    features = ['SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'StochRSI']
    X = df[features]
    y = df['close'].shift(-1).dropna()
    X = X.iloc[:-1]
    return X, y

def decide_trade(future_price, current_price):
    if future_price > current_price * BUY_THRESHOLD:
        return "buy"
    elif future_price < current_price * SELL_THRESHOLD:
        return "sell"
    return "hold"

async def main():
    logging.info("Starting the Crypto Bot Execution.")
    
    actual_coins = COINS
    if USE_DYNAMIC_COINS:
        actual_coins = await fetch_top_coins()
        logging.info(f"Dynamic coin list: {actual_coins}")

    daily_trades = 0
    for coin in actual_coins:
        raw_data = await fetch_market_data(coin)
        if raw_data is None:
            logging.error(f"No data for {coin}, skipping.")
            continue
        
        df = pd.DataFrame(raw_data)
        required_cols = {'close', 'high', 'low', 'open', 'volumefrom', 'volumeto'}
        if not required_cols.issubset(df.columns):
            logging.error(f"Data missing required columns for {coin}, skipping.")
            continue
        
        df = add_technical_indicators(df)
        if df is None or df.empty:
            logging.error(f"No valid data after indicators for {coin}, skipping.")
            continue
        
        X, y = generate_features_and_target(df)
        if X.empty or y.empty:
            logging.error(f"No training data for {coin}, skipping.")
            continue

        model = train_and_evaluate(X, y)
        if model:
            save_to_database(df, f"{coin}_data")

            current_price = df['close'].iloc[-1]
            future_price = model.predict(X.iloc[-1].values.reshape(1, -1))[0]
            logging.info(f"Predicted Future Price for {coin}: {future_price}, Current Price: {current_price}")

            if not risk_management_check(daily_trades):
                logging.info("Risk management prevented trading.")
                continue

            action = decide_trade(future_price, current_price)
            if action in ["buy", "sell"]:
                order = place_trade(action, coin, amount=TRADE_AMOUNT)
                if order:
                    daily_trades += 1
            else:
                logging.info(f"No trade signal for {coin}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    while True:
        try:
            asyncio.run(main())
            logging.info("Crypto Bot executed successfully.")
        except Exception as e:
            logging.error(f"Error during execution: {e}\n{traceback.format_exc()}")
        finally:
            time.sleep(LOOP_INTERVAL)
