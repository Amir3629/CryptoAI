# crypto_bot.py

import os
import sys
import time
import asyncio
import aiohttp
import logging
import traceback
import yaml
import joblib
import pandas as pd
import numpy as np
import requests
import warnings
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
import ta
import ccxt.async_support as ccxt  # Asynchronous version of CCXT
from logging.handlers import RotatingFileHandler

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------
# Load Environment Variables
# ----------------------------------------
load_dotenv()

# ----------------------------------------
# Configuration Loading
# ----------------------------------------
def load_config():
    """
    Load configuration from config.yaml and config.local.yaml (if enabled).
    """
    config = {}
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

    # If local config is enabled and exists, override main config
    if config.get('trading', {}).get('use_local_config', False):
        if os.path.exists('config.local.yaml'):
            with open('config.local.yaml', 'r') as f:
                local_config = yaml.safe_load(f)
            # Deep update of the main config with local config
            for key, value in local_config.items():
                if isinstance(value, dict):
                    if key in config and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value
                else:
                    config[key] = value
        else:
            logging.warning("Local config enabled but 'config.local.yaml' not found.")

    return config

config = load_config()

def get_config_value(key_path, default=None):
    """
    Retrieve configuration value using dot notation.
    """
    keys = key_path.split('.')
    current = config
    for k in keys:
        if k in current:
            current = current[k]
        else:
            env_val = os.getenv(key_path.upper().replace('.', '_'))
            if env_val is not None:
                try:
                    # Attempt to cast to the type of default if possible
                    if isinstance(default, bool):
                        return env_val.lower() in ('true', '1', 'yes')
                    return type(default)(env_val)
                except (ValueError, TypeError):
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
BUY_THRESHOLD = get_config_value("trading.buy_threshold", 1.002)
SELL_THRESHOLD = get_config_value("trading.sell_threshold", 0.998)
TRADE_AMOUNT = get_config_value("trading.trade_amount", 1000)  # USD
MAX_DAILY_TRADES = get_config_value("trading.max_daily_trades", 10)
TIMEFRAME = get_config_value("trading.timeframe", "1H")
LOOKBACK = get_config_value("trading.lookback", 500)

TEST_SIZE = get_config_value("model.test_size", 0.2)
PARAM_GRID = get_config_value("model.param_grid", {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
})
RETRAIN_MODEL = get_config_value("model.retrain", True)  # Set to True to train the model
MODEL_PATH = get_config_value("model.model_path", "models/trading_model.pkl")

LOOP_INTERVAL = get_config_value("execution.loop_interval", 60)

PRIMARY_DB = get_config_value("database.primary", "postgresql")
SECONDARY_DB = get_config_value("database.secondary", "mysql")

CRAWLING_ENABLED = get_config_value("crawling.enabled", False)
CRAWLING_WEBSITES = config.get('crawling', {}).get('websites', [])

# Logging Setup
LOG_FILE = get_config_value("logging.file", "logs/crypto_ai.log")
LOG_LEVEL = get_config_value("logging.level", "INFO").upper()

# Ensure log directory exists
LOG_DIR = os.path.dirname(LOG_FILE)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Define logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create logger
logger = logging.getLogger()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# File handler with log rotation
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)  # 10MB per file, keep 5 backups
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Crypto AI Program started.")

# ----------------------------------------
# Web Crawling Functions
# ----------------------------------------

def crawl_website(url, parse_function):
    """
    Crawl a website and parse its content using the provided parse_function.
    """
    headers = {
        'User-Agent': 'CryptoAI Bot/1.0 (+https://yourdomain.com/cryptoai)'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return parse_function(response.text)
        else:
            logging.error(f"Failed to fetch {url}: Status code {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Exception while crawling {url}: {e}", exc_info=True)
        return None

def parse_crypto_news(html_content):
    """
    Parse cryptocurrency news from Coindesk.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    news_items = []
    # Adjust the parsing logic based on Coindesk's actual HTML structure
    for item in soup.find_all('div', class_='article'):
        title_tag = item.find('h3')
        link_tag = item.find('a', href=True)
        if title_tag and link_tag:
            title = title_tag.text.strip()
            link = link_tag['href']
            # Ensure the link is absolute
            if not link.startswith('http'):
                link = f"https://www.coindesk.com{link}"
            news_items.append({'title': title, 'link': link})
    return news_items

def parse_another_crypto_site(html_content):
    """
    Parse cryptocurrency data from CoinMarketCap.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    data_items = []
    # Adjust the parsing logic based on CoinMarketCap's actual HTML structure
    for item in soup.find_all('li', class_='data-point'):
        name_tag = item.find('span', class_='name')
        value_tag = item.find('span', class_='value')
        if name_tag and value_tag:
            name = name_tag.text.strip()
            value = value_tag.text.strip()
            data_items.append({'name': name, 'value': value})
    return data_items

async def fetch_additional_data():
    """
    Fetch additional data from multiple websites as configured.
    """
    all_news = []
    all_data = []

    for site in CRAWLING_WEBSITES:
        name = site.get('name')
        url = site.get('url')
        parse_function_name = site.get('parse_function')

        # Dynamically get the parse function from globals
        parse_function = globals().get(parse_function_name)
        if not parse_function:
            logging.error(f"Parse function '{parse_function_name}' not found for website '{name}'.")
            continue

        logging.info(f"Crawling website: {name}")
        crawled_data = crawl_website(url, parse_function)
        if crawled_data:
            if 'news' in name.lower():
                all_news.extend(crawled_data)
            else:
                all_data.extend(crawled_data)

    # Process or store the crawled data as needed
    if all_news:
        logging.info(f"Fetched {len(all_news)} news items.")
        # Example: Save to a CSV file
        os.makedirs('data', exist_ok=True)
        news_df = pd.DataFrame(all_news)
        news_file = os.path.join('data', 'crypto_news.csv')
        news_df.to_csv(news_file, index=False)
        logging.info(f"Saved news data to {news_file}")

    if all_data:
        logging.info(f"Fetched {len(all_data)} data items.")
        # Example: Save to a CSV file
        os.makedirs('data', exist_ok=True)
        data_df = pd.DataFrame(all_data)
        data_file = os.path.join('data', 'crypto_additional_data.csv')
        data_df.to_csv(data_file, index=False)
        logging.info(f"Saved additional data to {data_file}")

# ----------------------------------------
# Risk Management
# ----------------------------------------

class RiskManager:
    def __init__(self, max_investment, stop_loss_pct, take_profit_pct):
        self.max_investment = max_investment
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def calculate_order_size(self, balance, price, min_amount, is_sell=False, position_quantity=0):
        """
        Calculate the order size based on max investment and min_amount.
        For BUY orders, use balance. For SELL orders, use position_quantity.
        Ensures that the order size meets the exchange's minimum requirements.
        """
        if is_sell:
            investment = position_quantity
        else:
            investment = min(self.max_investment, balance)
        
        order_size = investment / price
        
        # For SELL orders, ensure we're not selling more than we have
        if is_sell and order_size > position_quantity:
            order_size = position_quantity

        # Ensure order_size meets the minimum amount
        if order_size < min_amount:
            logging.warning(f"Calculated order size {order_size} is below the minimum amount {min_amount}. Adjusting to minimum.")
            return min_amount
        return order_size

    def get_stop_loss_price(self, entry_price):
        """
        Calculate stop-loss price.
        """
        return entry_price * (1 - self.stop_loss_pct / 100)

    def get_take_profit_price(self, entry_price):
        """
        Calculate take-profit price.
        """
        return entry_price * (1 + self.take_profit_pct / 100)

# ----------------------------------------
# Machine Learning Model
# ----------------------------------------

class TradingModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_fit = False  # Indicator whether the model is fitted
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.model_fit = True
            logging.info("Loaded existing ML model.")
        else:
            self.model = RandomForestRegressor()
            logging.info("Initialized new ML model.")

    def train(self, X, y):
        """
        Train the ML model.
        """
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        self.model_fit = True
        logging.info("Trained and saved ML model.")

    def predict(self, X):
        """
        Make predictions using the ML model.
        """
        if not self.model_fit:
            raise NotFittedError("This RandomForestRegressor instance is not fitted yet.")
        return self.model.predict(X)

# ----------------------------------------
# Data Source Interaction
# ----------------------------------------

class DataSource:
    def __init__(self, data_source, symbol, timeframe, lookback):
        self.data_source = data_source
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback

    async def get_data(self):
        """
        Fetch data based on the data source.
        """
        if self.data_source == 'simulated':
            return self.get_simulated_data()
        elif self.data_source == 'cryptocompare':
            return await self.get_cryptocompare_data()
        elif self.data_source == 'coingecko':
            return await self.get_coingecko_data()
        else:
            logging.error(f"Unsupported data source: {self.data_source}")
            return None

    def get_simulated_data(self):
        """
        Generate simulated market data.
        """
        logging.info("Generating simulated market data.")
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=self.lookback, freq=self.timeframe)
        prices = np.random.lognormal(mean=0, sigma=0.01, size=self.lookback).cumprod() * 100  # Simulated prices
        df = pd.DataFrame({
            'close': prices
        }, index=dates)
        df['open'] = df['close'].shift(1)
        df['open'].fillna(df['close'], inplace=True)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, size=self.lookback))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, size=self.lookback))
        df['volume'] = np.random.randint(100, 1000, size=self.lookback)
        return df

    async def get_cryptocompare_data(self):
        """
        Fetch market data from CryptoCompare.
        """
        # Extract base and quote symbols
        try:
            fsym, tsym = self.symbol.split('/')
        except ValueError:
            logging.error(f"Symbol format incorrect: {self.symbol}. Expected format 'BASE/QUOTE'.")
            return None

        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym={tsym}&limit={self.lookback}"
        headers = {'Authorization': f'Apikey {CRYPTOCOMPARE_API_KEY}'}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'Data' in data and 'Data' in data['Data']:
                            df = pd.DataFrame(data['Data']['Data'])
                            if not df.empty:
                                df.set_index(pd.to_datetime(df['time'], unit='s'), inplace=True)
                                df.rename(columns={
                                    'close': 'close',
                                    'high': 'high',
                                    'low': 'low',
                                    'open': 'open',
                                    'volumefrom': 'volume_from',
                                    'volumeto': 'volume_to'
                                }, inplace=True)
                                return df
                            else:
                                logging.error(f"No data returned from CryptoCompare for {self.symbol}.")
                                return None
                        else:
                            logging.error(f"Unexpected data format from CryptoCompare: {data}")
                            return None
                    else:
                        error_content = await response.text()
                        logging.error(f"Failed to fetch data from CryptoCompare: Status code {response.status}, Response: {error_content}")
                        return None
            except Exception as e:
                logging.error(f"Exception fetching data from CryptoCompare: {e}", exc_info=True)
                return None

    async def get_coingecko_data(self):
        """
        Fetch market data from CoinGecko.
        """
        # Extract base and quote symbols
        try:
            fsym, tsym = self.symbol.split('/')
        except ValueError:
            logging.error(f"Symbol format incorrect: {self.symbol}. Expected format 'BASE/QUOTE'.")
            return None

        fsym_lower = fsym.lower()
        tsym_lower = tsym.lower()
        url = f"https://api.coingecko.com/api/v3/coins/{fsym_lower}/market_chart"
        params = {
            'vs_currency': 'usd',  # Corrected to 'usd'
            'days': '5',
            'interval': 'hourly'
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        prices = data.get('prices', [])
                        if not prices:
                            logging.error(f"No price data returned from CoinGecko for {self.symbol}.")
                            return None
                        df = pd.DataFrame(prices, columns=['time', 'close'])
                        df['time'] = pd.to_datetime(df['time'], unit='ms')
                        df.set_index('time', inplace=True)
                        df['open'] = df['close'].shift(1)
                        df['open'].fillna(df['close'], inplace=True)
                        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, size=len(df)))
                        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, size=len(df)))
                        df['volume'] = np.random.randint(100, 1000, size=len(df))
                        return df
                    else:
                        error_content = await response.text()
                        logging.error(f"Failed to fetch data from CoinGecko: Status code {response.status}, Response: {error_content}")
                        return None
        except Exception as e:
            logging.error(f"Exception fetching data from CoinGecko: {e}", exc_info=True)
            return None

# ----------------------------------------
# Risk Management
# ----------------------------------------

# (Already defined above)

# ----------------------------------------
# Machine Learning Model
# ----------------------------------------

# (Already defined above)

# ----------------------------------------
# Data Source Interaction
# ----------------------------------------

# (Already defined above)

# ----------------------------------------
# Trading Bot
# ----------------------------------------

class TradingBot:
    def __init__(self, config):
        self.symbol = config['trading']['base_symbol']
        self.timeframe = config['trading'].get('timeframe', '1H')
        self.lookback = config['trading'].get('lookback', 200)
        self.risk_manager = RiskManager(
            max_investment=config['trading']['trade_amount'],
            stop_loss_pct=5,  # Example: 5%
            take_profit_pct=10  # Example: 10%
        )
        self.model = TradingModel(config['model'].get('model_path', 'models/trading_model.pkl'))
        self.position = None  # To track current position
        self.balance = config['trading']['trade_amount']  # Starting balance based on trade_amount

        self.data_source = DataSource(
            data_source=config['api']['data_source'],
            symbol=self.symbol,
            timeframe=self.timeframe,
            lookback=self.lookback
        )
        
        # Initialize exchange within the TradingBot
        self.exchange = ccxt.mexc({
            'apiKey': MEXC_API_KEY,
            'secret': MEXC_API_SECRET,
            'enableRateLimit': True,
        })

    async def close_exchange(self):
        """
        Close the exchange connection.
        """
        await self.exchange.close()

    def feature_engineering(self, df):
        """
        Create features for the ML model.
        """
        df = add_technical_indicators(df)
        return df

    def train_model_if_needed(self, df):
        """
        Train the ML model if retrain is set to True.
        """
        if RETRAIN_MODEL:
            logging.info("Retraining the ML model as per configuration.")
            X, y = generate_features_and_target(df)
            if X.empty or y.empty:
                logging.error("Insufficient data for training.")
                return
            # Ensure all features are numeric
            if not all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
                logging.error("Non-numeric features detected. Please check feature engineering.")
                return
            # Ensure target is numeric
            if not np.issubdtype(y.dtype, np.number):
                logging.error("Non-numeric target detected. Please check data preparation.")
                return
            self.model.train(X, y)

    def make_decision(self, df):
        """
        Use the ML model to make a trading decision.
        """
        latest_data = df.iloc[-1:]
        try:
            features = ['SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'StochRSI']
            predicted_price = self.model.predict(latest_data[features])[0]
            logging.debug(f"Predicted Price: {predicted_price}")
        except Exception as e:
            logging.error(f"Error during prediction: {e}", exc_info=True)
            return 1  # Neutral signal
        current_price = df['close'].iloc[-1]
        predicted_change = predicted_price / current_price
        logging.debug(f"Predicted Change: {predicted_change}, Current Price: {current_price}")
        return predicted_change

    async def run(self):
        """
        Run the trading bot.
        """
        logging.info(f"Running Trading Bot for {self.symbol}...")
        
        try:
            # Fetch and prepare data
            df = await self.data_source.get_data()
            if df is None or df.empty:
                logging.error("No data fetched, skipping this iteration.")
                return

            df = validate_data(df)
            df = self.feature_engineering(df)
            if df is None or df.empty:
                logging.error("Data after feature engineering is empty, skipping this iteration.")
                return

            # Log feature data types
            logging.info(
                f"Features data types:\n{df[['SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'StochRSI']].dtypes}"
            )

            # Train model if needed
            self.train_model_if_needed(df)

            # Ensure the model is fitted before making predictions
            if not self.model.model_fit:
                logging.error("ML model is not fitted yet. Skipping trading decision.")
                return

            # Make trading decision
            signal = self.make_decision(df)
            current_price = df['close'].iloc[-1]
            logging.info(f"Signal (Predicted Change): {signal}, Current Price: {current_price}")

            # Execute trade based on signal
            if signal > BUY_THRESHOLD:
                logging.info(f"BUY signal detected for {self.symbol}. Placing BUY order for {self.symbol}.")
                # Fetch market details to get minimum order size
                try:
                    market = self.exchange.market(self.symbol)
                except Exception as e:
                    logging.error(f"Error fetching market details for {self.symbol}: {e}")
                    return

                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
                order_size = self.risk_manager.calculate_order_size(
                    balance=self.balance,
                    price=current_price,
                    min_amount=min_amount,
                    is_sell=False
                )
                await self.place_order('buy', self.symbol, order_size, min_amount)
            elif signal < SELL_THRESHOLD:
                if self.position:
                    logging.info(f"SELL signal detected for {self.symbol}. Placing SELL order for {self.symbol}.")
                    try:
                        market = self.exchange.market(self.symbol)
                    except Exception as e:
                        logging.error(f"Error fetching market details for {self.symbol}: {e}")
                        return

                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
                    order_size = self.risk_manager.calculate_order_size(
                        balance=self.balance,
                        price=current_price,
                        min_amount=min_amount,
                        is_sell=True,
                        position_quantity=self.position['quantity']
                    )
                    # Ensure that order_size does not exceed the position quantity
                    order_size = min(order_size, self.position['quantity'])
                    await self.place_order('sell', self.symbol, order_size, min_amount)
                else:
                    logging.info(f"No open position to SELL for {self.symbol}.")
            else:
                logging.info(f"No actionable signal for {self.symbol}.")

            # Check for stop-loss or take-profit
            self.check_stop_loss_take_profit(current_price)
        except Exception as e:
            logging.error(f"Exception in run method for {self.symbol}: {e}", exc_info=True)
        finally:
            await self.close_exchange()

    async def place_order(self, side, symbol, amount, min_amount):
        """
        Place an order on the exchange with retry logic.
        """
        max_attempts = 3
        delay = 2  # Initial delay in seconds
        for attempt in range(1, max_attempts + 1):
            try:
                logging.info(f"Attempt {attempt}: Placing {side.upper()} order for {symbol} with quantity {amount}...")
                if side.lower() == 'buy':
                    order = await self.exchange.create_market_buy_order(symbol, amount)
                elif side.lower() == 'sell':
                    order = await self.exchange.create_market_sell_order(symbol, amount)
                else:
                    logging.error(f"Invalid order side: {side}")
                    return
                logging.info(f"Order placed successfully: {order}")
                # Update balance and position
                average_price = order.get('average', None)
                if average_price is None:
                    # If average price not available, use last price
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last'] if 'last' in ticker else 0  # Define a default or handle appropriately
                    average_price = current_price

                if side.lower() == 'buy':
                    self.balance -= amount * average_price
                    self.position = {
                        'quantity': amount,
                        'entry_price': average_price,
                        'stop_loss': self.risk_manager.get_stop_loss_price(average_price),
                        'take_profit': self.risk_manager.get_take_profit_price(average_price)
                    }
                    logging.info(f"Bought {amount:.6f} {symbol} at {average_price:.2f}")
                elif side.lower() == 'sell':
                    self.balance += amount * average_price
                    logging.info(f"Sold {amount:.6f} {symbol} at {average_price:.2f}")
                    self.position = None
                return
            except ccxt.InsufficientFunds as e:
                logging.error(f"Attempt {attempt} failed: Insufficient funds - {e}")
                return
            except ccxt.InvalidOrder as e:
                logging.error(f"Attempt {attempt} failed: Invalid order - {e}")
                return
            except ccxt.BaseError as e:
                logging.error(f"Attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logging.error(f"Failed to place order for {symbol} after {max_attempts} attempts.")

    def check_stop_loss_take_profit(self, current_price):
        """
        Check if current price hits stop-loss or take-profit.
        """
        if self.position:
            if current_price <= self.position['stop_loss']:
                logging.info("Stop-loss triggered.")
                self.execute_trade('sell', self.symbol, self.position['quantity'], current_price)
            elif current_price >= self.position['take_profit']:
                logging.info("Take-profit triggered.")
                self.execute_trade('sell', self.symbol, self.position['quantity'], current_price)

    def execute_trade(self, side, symbol, quantity, price):
        """
        Execute trade based on the signal.
        """
        if side.lower() == 'sell' and self.position:
            self.balance += quantity * price
            logging.info(f"Sold {quantity:.6f} {symbol} at {price:.2f}")
            self.position = None

# ----------------------------------------
# Utility Functions
# ----------------------------------------

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    """
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
        
        # Additional indicators
        df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        df['CCI'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        
        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"Failed to add indicators: {e}", exc_info=True)
        return None

def generate_features_and_target(df):
    """
    Generate features and target for model training.
    """
    features = ['SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'StochRSI', 'ADX', 'CCI']
    X = df[features]
    y = df['close'].shift(-1).dropna()
    X = X.iloc[:-1]
    
    # Log data types
    logging.info(f"Features data types:\n{X.dtypes}")
    logging.info(f"Target data type: {y.dtype}")
    
    return X, y

def validate_data(df):
    """
    Validate and clean the fetched data.
    """
    if df.isnull().values.any():
        logging.warning("Data contains null values. Filling missing data.")
        df = df.fillna(method='ffill').dropna()
    return df

# ----------------------------------------
# Database Connection and Saving
# ----------------------------------------

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
        logging.error(f"Error saving to secondary DB: {e}", exc_info=True)

# ----------------------------------------
# Main Execution
# ----------------------------------------

async def main_execution():
    """
    Main execution loop for the trading bot.
    """
    logging.info("Starting the Crypto Bot Execution.")
    
    while True:
        try:
            # Fetch additional data via web crawling
            if CRAWLING_ENABLED:
                await fetch_additional_data()
            
            # Determine actual coins to trade
            actual_coins = COINS
            if USE_DYNAMIC_COINS:
                actual_coins = await fetch_top_coins()
                logging.info(f"Dynamic coin list: {actual_coins}")

        except Exception as e:
            logging.error(f"Error fetching additional data: {e}", exc_info=True)
            actual_coins = COINS  # Fallback to predefined coins
        
        daily_trades = 0
        for symbol in actual_coins:
            try:
                # Initialize Trading Bot for each symbol
                bot_config = config.copy()
                bot_config['trading']['base_symbol'] = symbol
                bot = TradingBot(bot_config)
                
                await bot.run()
                
                if bot.position:
                    daily_trades += 1
                    if daily_trades >= MAX_DAILY_TRADES:
                        logging.warning("Max daily trades limit reached. No more trades will be made today.")
                        break
            except Exception as e:
                logging.error(f"Error running TradingBot for {symbol}: {e}", exc_info=True)
        
        logging.info("Crypto Bot execution completed for this cycle.")
        
        # Wait for the next cycle
        logging.info(f"Sleeping for {LOOP_INTERVAL} seconds before next cycle.")
        await asyncio.sleep(LOOP_INTERVAL)

async def fetch_top_coins(limit=10):
    """
    Fetch top coins by market capitalization using CoinGecko API.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',  # Corrected to 'usd'
        'order': 'market_cap_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': 'false'
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    coins = [coin['symbol'].upper() for coin in data]
                    # Initialize a temporary exchange to get supported symbols
                    temp_exchange = ccxt.mexc({
                        'apiKey': MEXC_API_KEY,
                        'secret': MEXC_API_SECRET,
                        'enableRateLimit': True,
                    })
                    supported_symbols = await get_supported_symbols(temp_exchange)
                    await temp_exchange.close()
                    exchange_symbols = []
                    for symbol in coins:
                        formatted_symbol = f"{symbol}/USDT"  # Ensures correct formatting
                        if formatted_symbol in supported_symbols:
                            exchange_symbols.append(formatted_symbol)
                        else:
                            logging.warning(f"Symbol {formatted_symbol} not supported by exchange. Skipping.")
                    return exchange_symbols
                else:
                    error_content = await response.text()
                    logging.error(f"Failed to fetch top coins: Status code {response.status}, Response: {error_content}")
                    return COINS  # Fallback to predefined coins
    except Exception as e:
        logging.error(f"Exception fetching top coins: {e}", exc_info=True)
        return COINS  # Fallback to predefined coins

async def get_supported_symbols(exchange):
    """
    Fetch and return a list of supported symbols from the exchange.
    """
    try:
        await exchange.load_markets()
        markets = exchange.markets
        supported_symbols = list(markets.keys())
        logging.info(f"Fetched {len(supported_symbols)} supported symbols.")
        logging.info(f"Sample supported symbols: {supported_symbols[:10]}...")
        return supported_symbols
    except Exception as e:
        logging.error(f"Error fetching supported symbols: {e}", exc_info=True)
        return []

# ----------------------------------------
# Entry Point
# ----------------------------------------

if __name__ == "__main__":
    # Set event loop policy for Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Start the main execution loop
    try:
        asyncio.run(main_execution())
    except KeyboardInterrupt:
        logging.info("Crypto Bot stopped manually.")
    except Exception as e:
        logging.error(f"Error during execution: {e}\n{traceback.format_exc()}", exc_info=True)
