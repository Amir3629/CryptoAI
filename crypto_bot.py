# crypto_bot.py

import os
import sys
import asyncio
import logging
import traceback
import yaml
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
import ccxt.async_support as ccxt
from logging.handlers import RotatingFileHandler
import json
import warnings
import aiohttp
from functools import wraps
import time
import websocket  # For real-time data streaming
import threading
import nltk  # For sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf  # For LSTM models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import dash  # For User Interface
from dash import dcc, html
from dash.dependencies import Input, Output
import requests  # For API rate limiting
from sqlalchemy.exc import OperationalError
import pandas_datareader as pdr  # For backtesting
import smtplib  # For email notifications
from email.mime.text import MIMEText

# ----------------------------------------
# Suppress Specific Warnings
# ----------------------------------------
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="DataFrame.fillna with 'method' is deprecated")
warnings.filterwarnings("ignore", message="'T' is deprecated")

# ----------------------------------------
# Load Environment Variables
# ----------------------------------------
load_dotenv()

# KuCoin API credentials
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

# MEXC Exchange credentials
MEXC_API_KEY = os.getenv("MEXC_API_KEY")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET")

# Telegram Bot Credentials
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Email Configuration
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# Database Configuration (PostgreSQL)
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")
DATABASE_URL = os.getenv("DATABASE_URL")

# Database Configuration (MySQL)
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Elastic Cloud Configuration
ELASTIC_CLOUD_URL = os.getenv("ELASTIC_CLOUD_URL")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")

# ----------------------------------------
# Configuration Loading
# ----------------------------------------
def load_config():
    """
    Load configuration from config.yaml and config.local.yaml (if exists).
    """
    config = {}
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f) or {}

    # Load config.local.yaml if exists to override config.yaml
    if os.path.exists('config.local.yaml'):
        with open('config.local.yaml', 'r') as f:
            local_config = yaml.safe_load(f) or {}
        for key, value in local_config.items():
            if isinstance(value, dict):
                config.setdefault(key, {}).update(value)
            else:
                config[key] = value
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
                if isinstance(default, bool):
                    return env_val.lower() in ('true', '1', 'yes')
                try:
                    return type(default)(env_val)
                except:
                    return env_val
            return default
    return current

# ----------------------------------------
# Configuration Parameters
# ----------------------------------------
COINS = get_config_value("trading.coins", ["BTC", "ETH", "SOL"])  # Only base symbols without "/USDT"
USE_DYNAMIC_COINS = get_config_value("trading.dynamic_coins", False)
BASE_SYMBOL = get_config_value("trading.base_symbol", "USDT")
BUY_THRESHOLD = get_config_value("trading.buy_threshold", 1.005)  # 0.5% predicted increase
SELL_THRESHOLD = get_config_value("trading.sell_threshold", 0.995)  # 0.5% predicted decrease
TRADE_AMOUNT = get_config_value("trading.trade_amount", 500)
MAX_DAILY_TRADES = get_config_value("trading.max_daily_trades", 10)
MAX_DAILY_LOSS = get_config_value("trading.max_daily_loss", 500)  # Maximum daily loss limit
TIMEFRAME = get_config_value("trading.timeframe", "1m")  # 1-minute candlesticks
LOOKBACK = get_config_value("trading.lookback", 200)
RETRAIN_MODEL = get_config_value("model.retrain", True)
MODEL_PATH = get_config_value("model.model_path", "models/trading_model.pkl")
LSTM_MODEL_PATH = get_config_value("model.lstm_model_path", "models/lstm_trading_model.h5")
LOOP_INTERVAL = get_config_value("execution.loop_interval", 30)  # seconds
DATA_SOURCE = get_config_value("trading.data_source", "real")  # "real" or "simulated"
DRY_RUN = get_config_value("trading.dry_run", True)

LOG_FILE = get_config_value("logging.file", "logs/crypto_ai.log")
LOG_LEVEL = get_config_value("logging.level", "DEBUG").upper()

PRIMARY_DB = get_config_value("database.primary", "postgresql")
SECONDARY_DB = get_config_value("database.secondary", "mysql")

# ----------------------------------------
# Logging Setup
# ----------------------------------------
# JSON Formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'funcName': record.funcName,
            'lineNo': record.lineno,
            'process': record.process,
            'thread': record.thread,
        }
        return json.dumps(log_record)

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))

# Rotating File Handler with JSON Formatter
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8', delay=True)
json_formatter = JsonFormatter()
file_handler.setFormatter(json_formatter)
logger.addHandler(file_handler)

# Console Handler with JSON Formatter (optional)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(JsonFormatter())
logger.addHandler(console_handler)

logger.info("Crypto AI Program started.")

# ----------------------------------------
# Retry Decorator with Exponential Backoff
# ----------------------------------------
def retry(max_attempts=5, initial_delay=1, backoff_factor=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logging.error(f"Function {func.__name__} failed after {max_attempts} attempts.")
                        raise
                    else:
                        logging.warning(f"Function {func.__name__} failed on attempt {attempt}/{max_attempts}. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
        return wrapper
    return decorator

# ----------------------------------------
# Indicator Functions (Custom Implementation)
# ----------------------------------------
def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd = fast_ema - slow_ema
    macd_signal = ema(macd, signal)
    return macd, macd_signal

def calc_bollinger(series, window=20, std_dev=2):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    bb_high = mean + std_dev * std
    bb_low = mean - std_dev * std
    return bb_high, bb_low

# ----------------------------------------
# Sentiment Analysis Setup
# ----------------------------------------
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']  # Returns a score between -1 (negative) and +1 (positive)

# ----------------------------------------
# Risk Manager
# ----------------------------------------
class RiskManager:
    def __init__(self, max_investment, stop_loss_pct=5, take_profit_pct=10, max_daily_loss=500):
        self.max_investment = max_investment
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss = max_daily_loss
        self.daily_loss = 0
        self.trade_count = 0

    def calculate_order_size(self, balance, price, min_amount, is_sell=False, position_quantity=0):
        if is_sell:
            investment = position_quantity
        else:
            investment = min(self.max_investment, balance)

        order_size = investment / price
        if is_sell and order_size > position_quantity:
            order_size = position_quantity
        if order_size < min_amount:
            logging.warning(f"Order size {order_size} < min amount {min_amount}, using min_amount.")
            return min_amount
        return order_size

    def get_stop_loss_price(self, entry_price):
        return entry_price * (1 - self.stop_loss_pct / 100)

    def get_take_profit_price(self, entry_price):
        return entry_price * (1 + self.take_profit_pct / 100)

    def update_daily_loss(self, loss):
        self.daily_loss += loss

    def reset_daily_loss(self):
        self.daily_loss = 0
        self.trade_count = 0

# ----------------------------------------
# Trading Model
# ----------------------------------------
class TradingModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_fit = False
        self.model = None
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.model_fit = True
                logging.info(f"Loaded existing ML model from {self.model_path}.")
            except Exception as e:
                logging.error(f"Failed to load model from {self.model_path}: {e}", exc_info=True)
                self.model = GradientBoostingRegressor()
                logging.info("Initialized new ML model.")
        else:
            self.model = GradientBoostingRegressor()
            logging.info("Initialized new ML model.")

    def train(self, X, y):
        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        self.model_fit = True
        logging.info(f"Trained and saved ML model with parameters: {grid_search.best_params_}")

    def predict(self, X):
        if not self.model_fit:
            raise NotFittedError("Model not fitted.")
        return self.model.predict(X)

# ----------------------------------------
# LSTM Trading Model (Advanced ML)
# ----------------------------------------
class LSTMModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_fit = False
        self.model = None
        self.scaler = StandardScaler()
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(f"{self.model_path}_scaler.pkl")
                self.model_fit = True
                logging.info(f"Loaded existing LSTM model from {self.model_path}.")
            except Exception as e:
                logging.error(f"Failed to load LSTM model from {self.model_path}: {e}", exc_info=True)
                self.model = self.build_model()
                logging.info("Initialized new LSTM model.")
        else:
            self.model = self.build_model()
            logging.info("Initialized new LSTM model.")

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X, y):
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1,1))

        # Prepare data for LSTM
        X_train = []
        y_train = []
        for i in range(60, len(X_scaled)):
            X_train.append(X_scaled[i-60:i, 0])
            y_train.append(y_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Train model
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Save model and scaler
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, f"{self.model_path}_scaler.pkl")
        self.model_fit = True
        logging.info(f"Trained and saved LSTM model to {self.model_path}.")

    def predict(self, X):
        if not self.model_fit:
            raise NotFittedError("LSTM Model not fitted.")
        X_scaled = self.scaler.transform(X)
        X_input = []
        for i in range(60, len(X_scaled)):
            X_input.append(X_scaled[i-60:i, 0])
        X_input = np.array(X_input)
        X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
        predictions = self.model.predict(X_input)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions.flatten()

# ----------------------------------------
# Sentiment Analysis Integration
# ----------------------------------------
async def fetch_sentiment(symbol):
    """
    Fetches sentiment data from Twitter or Reddit for the given symbol.
    This is a placeholder function. Integration with actual sentiment data sources is required.
    """
    # Example using NLTK's VADER on fetched tweets (requires Twitter API integration)
    # Placeholder implementation:
    try:
        # Replace this with actual data fetching from Twitter/Reddit
        sample_texts = [
            "I love the growth of BTC!",
            "ETH is looking weak today.",
            "SOL has great potential.",
            "Market is bearish for cryptocurrencies."
        ]
        sentiment_scores = [analyze_sentiment(text) for text in sample_texts]
        average_sentiment = np.mean(sentiment_scores)
        return average_sentiment
    except Exception as e:
        logging.error(f"Exception during sentiment fetching for {symbol}: {e}", exc_info=True)
        return 0

# ----------------------------------------
# Data Source with Real-Time Streaming
# ----------------------------------------
class DataSource:
    def __init__(self, symbol, timeframe, lookback, exchange, data_source):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.exchange = exchange
        self.data_source = data_source.lower()
        self.websocket = None
        self.real_time_data = []

    @retry(max_attempts=5, initial_delay=1, backoff_factor=2, exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    async def get_real_data(self):
        logging.info(f"Fetching real market data for {self.symbol} with timeframe: {self.timeframe}")
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=self.lookback)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def get_simulated_data(self):
        logging.info(f"Generating simulated market data for {self.symbol}.")
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=self.lookback, freq='1T')  # '1T' for 1 minute
        # Simulated price data with vectorized operations
        price_changes = np.random.normal(loc=0, scale=0.001, size=self.lookback)
        prices = 100 * np.exp(np.cumsum(price_changes))  # Cumulative sum for trend
        df = pd.DataFrame({
            'close': prices,
            'open': np.roll(prices, 1),
            'high': prices * (1 + np.random.uniform(0, 0.01, self.lookback)),
            'low': prices * (1 - np.random.uniform(0, 0.01, self.lookback)),
            'volume': np.random.randint(100, 1000, self.lookback)
        }, index=dates)
        df.iloc[0, df.columns.get_loc('open')] = df.iloc[0]['close']  # Replace NaN with close price
        return df

    async def get_data(self):
        if self.data_source == "real":
            try:
                return await self.get_real_data()
            except Exception as e:
                logging.error(f"Error fetching real data for {self.symbol}: {e}", exc_info=True)
                return None
        elif self.data_source == "simulated":
            return self.get_simulated_data()
        else:
            logging.error(f"Unknown data source: {self.data_source}. Defaulting to simulated.")
            return self.get_simulated_data()

    def start_websocket(self):
        """
        Initiates a WebSocket connection for real-time data streaming.
        Replace the WebSocket URL with the actual exchange's WebSocket API.
        """
        # Example using Binance WebSocket (remove or replace if not using Binance)
        # stream = f"wss://stream.binance.com:9443/ws/{self.symbol.lower().replace('/', '')}@kline_{self.timeframe}"
        # self.websocket = websocket.WebSocketApp(stream,
        #                                        on_message=self.on_message,
        #                                        on_error=self.on_error,
        #                                        on_close=self.on_close)
        # threading.Thread(target=self.websocket.run_forever, daemon=True).start()
        pass  # Remove Binance WebSocket if not using

    def on_message(self, ws, message):
        data = json.loads(message)
        kline = data['k']
        close_price = float(kline['c'])
        timestamp = pd.to_datetime(kline['T'], unit='ms')
        self.real_time_data.append({'timestamp': timestamp, 'close': close_price})
        if len(self.real_time_data) > self.lookback:
            self.real_time_data.pop(0)

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.warning("WebSocket connection closed. Attempting to reconnect...")
        # Reconnect after a short delay
        time.sleep(5)
        self.start_websocket()

    def get_real_time_dataframe(self):
        if not self.real_time_data:
            return pd.DataFrame()
        df = pd.DataFrame(self.real_time_data)
        df.set_index('timestamp', inplace=True)
        return df

# ----------------------------------------
# Feature Engineering
# ----------------------------------------
def feature_engineering(df):
    try:
        logging.debug(f"Initial DataFrame shape: {df.shape}")
        # Forward fill and drop remaining NaNs
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        logging.debug(f"DataFrame shape after initial fillna and dropna: {df.shape}")

        # Compute indicators using vectorized operations
        df['SMA_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['EMA_5'] = ema(df['close'], 5)
        df['RSI'] = calc_rsi(df['close'], 14)
        macd, macd_signal = calc_macd(df['close'])
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        bb_high, bb_low = calc_bollinger(df['close'], window=20, std_dev=2)
        df['BB_high'] = bb_high
        df['BB_low'] = bb_low

        # Drop rows with any NaN values in the indicators
        required_features = ['SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low']
        df = df.dropna(subset=required_features)
        logging.debug(f"DataFrame shape after dropping NaNs in indicators: {df.shape}")

        if df.empty:
            logging.error("DataFrame is empty after feature engineering.")
        return df
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}", exc_info=True)
        return pd.DataFrame()

def generate_features_and_target(df):
    try:
        features = ['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low']
        X = df[features]
        y = df['close'].shift(-1).dropna()
        X = X.iloc[:-1]  # Align X with y
        return X, y
    except Exception as e:
        logging.error(f"Error generating features and target: {e}", exc_info=True)
        return pd.DataFrame(), pd.Series()

# ----------------------------------------
# Trading Bot Class
# ----------------------------------------
class TradingBot:
    def __init__(self, config, symbol, exchange):
        self.symbol = symbol  # e.g., "BTC/USDT"
        self.timeframe = config['trading'].get('timeframe', '1m')
        self.lookback = config['trading'].get('lookback', 200)
        self.risk_manager = RiskManager(
            max_investment=config['trading']['trade_amount'],
            max_daily_loss=config['trading']['max_daily_loss']
        )
        self.model = TradingModel(config['model'].get('model_path', 'models/trading_model.pkl'))
        self.lstm_model = LSTMModel(config['model'].get('lstm_model_path', 'models/lstm_trading_model.h5'))
        self.position = None
        self.balance = config['trading']['trade_amount']
        self.sentiment = 0
        self.daily_loss = 0
        self.trade_count = 0

        # Initialize exchange
        self.exchange = exchange

        # Initialize data source
        self.data_source = DataSource(
            symbol=self.symbol,
            timeframe=self.timeframe,
            lookback=self.lookback,
            exchange=self.exchange,
            data_source=DATA_SOURCE
        )

        # Unique model and position file paths per symbol
        symbol_safe = self.symbol.replace('/', '_')
        self.model_path = f"models/{symbol_safe}_trading_model.pkl"
        self.lstm_model_path = f"models/{symbol_safe}_lstm_trading_model.h5"
        self.model.model_path = self.model_path  # Update model path
        self.lstm_model.model_path = self.lstm_model_path
        self.position_file = f"positions/{symbol_safe}_position.json"
        self.load_position()

        # Initialize sentiment
        self.sentiment = 0

    def load_position(self):
        if os.path.exists(self.position_file):
            with open(self.position_file, 'r') as f:
                self.position = json.load(f)
            logging.info(f"Loaded existing position for {self.symbol}: {self.position}")
        else:
            self.position = None

    def save_position(self):
        os.makedirs('positions', exist_ok=True)
        with open(self.position_file, 'w') as f:
            json.dump(self.position, f)

    async def close_exchange(self):
        await self.exchange.close()

    async def send_notification(self, message):
        """
        Sends a notification via Telegram and Email.
        """
        # Telegram Notification
        try:
            telegram_url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendMessage"
            payload = {
                'chat_id': CHAT_ID,
                'text': message
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(telegram_url, data=payload) as response:
                    if response.status != 200:
                        logging.error(f"Failed to send Telegram message: HTTP {response.status}")
        except Exception as e:
            logging.error(f"Failed to send Telegram notification: {e}", exc_info=True)

        # Email Notification
        try:
            msg = MIMEText(message)
            msg['Subject'] = 'CryptoAI Alert'
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECIPIENT_EMAIL

            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()
            logging.info("Email notification sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send Email notification: {e}", exc_info=True)

    def check_stop_loss_take_profit(self, current_price):
        if self.position:
            if current_price <= self.position['stop_loss']:
                logging.info("Stop-loss triggered.")
                asyncio.create_task(self.handle_sell(current_price))
                asyncio.create_task(self.send_notification(f"Stop-loss triggered for {self.symbol} at price {current_price}"))
                # Update daily loss
                loss = (self.position['entry_price'] - current_price) * self.position['quantity']
                self.risk_manager.update_daily_loss(loss)
            elif current_price >= self.position['take_profit']:
                logging.info("Take-profit triggered.")
                asyncio.create_task(self.handle_sell(current_price))
                asyncio.create_task(self.send_notification(f"Take-profit triggered for {self.symbol} at price {current_price}"))
                # Update daily profit
                profit = (current_price - self.position['entry_price']) * self.position['quantity']
                self.balance += profit
                self.risk_manager.trade_count += 1

    def save_for_backtesting(self, df):
        """
        Saves the latest data to a CSV file for backtesting purposes.
        """
        os.makedirs('data', exist_ok=True)
        df.to_csv(f"data/{self.symbol.replace('/', '_')}_data.csv", mode='a', header=False)

    @retry(max_attempts=5, initial_delay=1, backoff_factor=2, exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ccxt.InvalidOrder))
    async def handle_buy(self, current_price):
        try:
            market = self.exchange.market(self.symbol)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
            order_size = self.risk_manager.calculate_order_size(
                balance=self.balance,
                price=current_price,
                min_amount=min_amount,
                is_sell=False
            )
            await self.place_order('buy', self.symbol, order_size, min_amount, current_price)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logging.error(f"Network or Exchange error during BUY: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error during BUY: {e}", exc_info=True)

    @retry(max_attempts=5, initial_delay=1, backoff_factor=2, exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ccxt.InvalidOrder))
    async def handle_sell(self, current_price):
        try:
            if self.position:
                market = self.exchange.market(self.symbol)
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
                order_size = self.risk_manager.calculate_order_size(
                    balance=self.balance,
                    price=current_price,
                    min_amount=min_amount,
                    is_sell=True,
                    position_quantity=self.position['quantity']
                )
                order_size = min(order_size, self.position['quantity'])
                await self.place_order('sell', self.symbol, order_size, min_amount, current_price)
            else:
                logging.info(f"No open position to SELL for {self.symbol}.")
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logging.error(f"Network or Exchange error during SELL: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error during SELL: {e}", exc_info=True)

    @retry(max_attempts=5, initial_delay=1, backoff_factor=2, exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ccxt.InvalidOrder))
    async def place_order(self, side, symbol, amount, min_amount, price):
        try:
            logging.info(f"Placing {side.upper()} order for {symbol} with amount {amount:.6f}...")
            if side.lower() == 'buy':
                if not DRY_RUN:
                    # Place real buy order
                    order = await self.exchange.create_market_buy_order(symbol, amount)
                    logging.info(f"Executed BUY order: {order}")
                    await self.send_notification(f"Executed BUY order for {symbol}: {amount} at {price}")
                    # Update balance and position
                    self.balance -= amount * price
                    self.position = {
                        'quantity': amount,
                        'entry_price': price,
                        'stop_loss': self.risk_manager.get_stop_loss_price(price),
                        'take_profit': self.risk_manager.get_take_profit_price(price)
                    }
                    self.save_position()
                    self.trade_count += 1
                else:
                    logging.info(f"[DRY RUN] Simulated BUY order for {symbol}: {amount} at {price}")
            elif side.lower() == 'sell':
                if self.position:
                    if not DRY_RUN:
                        # Place real sell order
                        order = await self.exchange.create_market_sell_order(symbol, amount)
                        logging.info(f"Executed SELL order: {order}")
                        await self.send_notification(f"Executed SELL order for {symbol}: {amount} at {price}")
                        # Update balance and position
                        self.balance += amount * price
                        self.position = None
                        self.save_position()
                        self.trade_count += 1
                    else:
                        logging.info(f"[DRY RUN] Simulated SELL order for {symbol}: {amount} at {price}")
                else:
                    logging.info(f"No open position to SELL for {symbol}.")
            else:
                logging.error(f"Invalid order side: {side}")
        except ccxt.InsufficientFunds as e:
            logging.error(f"Insufficient funds for {side.upper()} order: {e}")
            await self.send_notification(f"Insufficient funds for {side.upper()} order: {e}")
        except ccxt.InvalidOrder as e:
            logging.error(f"Invalid {side.upper()} order: {e}")
            await self.send_notification(f"Invalid {side.upper()} order: {e}")
            raise
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logging.error(f"Network or Exchange error during {side.upper()} order: {e}", exc_info=True)
            await self.send_notification(f"Network or Exchange error during {side.upper()} order: {e}")
            raise
        except Exception as e:
            logging.error(f"Failed to place {side.upper()} order for {symbol}: {e}", exc_info=True)
            await self.send_notification(f"Failed to place {side.upper()} order for {symbol}: {e}")

    async def run(self):
        logging.info(f"Running Trading Bot for {self.symbol}...")
        try:
            # Start WebSocket for real-time data
            self.data_source.start_websocket()
            logging.debug("WebSocket started for real-time data.")

            # Fetch initial Data
            df = await self.data_source.get_data()
            if df is None or df.empty:
                logging.error("No data fetched, skipping.")
                return
            logging.debug(f"Fetched data for {self.symbol}: {df.shape[0]} records.")
            logging.debug(df.head())

            # Feature Engineering
            df = feature_engineering(df)
            if df is None or df.empty:
                logging.error("Data after feature engineering is empty, skipping.")
                return
            logging.debug("Completed feature engineering.")
            logging.debug(df.head())

            # Fetch Sentiment
            self.sentiment = await fetch_sentiment(self.symbol)
            logging.info(f"Sentiment Score for {self.symbol}: {self.sentiment}")

            # Retrain Models if Needed
            if RETRAIN_MODEL:
                logging.info("Retraining the ML models as per configuration.")
                X, y = generate_features_and_target(df)
                if not X.empty and not y.empty:
                    self.model.train(X, y)
                    self.lstm_model.train(X, y)
                    logging.debug("Models retrained successfully.")
                else:
                    logging.error("Insufficient data to train models.")
                    return

            if not self.model.model_fit or not self.lstm_model.model_fit:
                logging.error("One or both ML models are not fitted yet. Skipping trading decision.")
                return

            # Make Decision with Combined Models and Sentiment
            features = ['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low']
            latest_data = df[features].tail(1)
            predicted_price_gb = self.model.predict(latest_data)[0]
            predicted_price_lstm = self.lstm_model.predict(latest_data)[-1]
            current_price = df['close'].iloc[-1]

            # Combine predictions with sentiment
            combined_prediction = (predicted_price_gb + predicted_price_lstm) / 2
            sentiment_factor = (self.sentiment + 1) / 2  # Normalize sentiment to [0,1]
            adjusted_prediction = combined_prediction * sentiment_factor
            predicted_change = adjusted_prediction / current_price

            logging.info(f"Combined Prediction Change: {predicted_change:.6f}, Current Price: {current_price:.6f}")

            # Check if daily loss limit is reached
            if self.risk_manager.daily_loss >= self.risk_manager.max_daily_loss:
                logging.warning(f"Daily loss limit of {self.risk_manager.max_daily_loss} reached. Halting trades.")
                await self.send_notification(f"Daily loss limit of {self.risk_manager.max_daily_loss} reached for {self.symbol}. Halting trades.")
                return

            # Ensure Market is Loaded
            await self.exchange.load_markets()
            if self.symbol not in self.exchange.symbols:
                logging.error(f"{self.exchange.id} does not have market symbol {self.symbol}. Skipping.")
                return

            # Trading Logic
            if predicted_change > BUY_THRESHOLD:
                logging.info(f"BUY signal for {self.symbol}.")
                await self.handle_buy(current_price)
            elif predicted_change < SELL_THRESHOLD:
                logging.info(f"SELL signal for {self.symbol}.")
                await self.handle_sell(current_price)
            else:
                logging.info(f"No actionable signal for {self.symbol}.")

            # Check Stop-Loss and Take-Profit
            self.check_stop_loss_take_profit(current_price)

            # Save Data for Backtesting
            self.save_for_backtesting(df)
        except Exception as e:
            logging.error(f"Exception in run method for {self.symbol}: {e}", exc_info=True)
        finally:
            await self.close_exchange()
            logging.debug(f"Exchange connection closed for {self.symbol}.")

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
            data.to_sql(table_name, engine, if_exists='append', index=False)
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
        data.to_sql(table_name, engine, if_exists='append', index=False)
        logging.info(f"Data saved to {SECONDARY_DB.upper()} table: {table_name}")
    except Exception as e:
        logging.error(f"Error saving to secondary DB: {e}", exc_info=True)

# ----------------------------------------
# Backtesting Framework
# ----------------------------------------
def backtest_strategy(symbol, start_date, end_date):
    """
    Backtests the trading strategy using historical data.
    """
    try:
        df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
        df = feature_engineering(df)
        X, y = generate_features_and_target(df)
        model = TradingModel(f"models/{symbol}_trading_model.pkl")
        lstm_model = LSTMModel(f"models/{symbol}_lstm_trading_model.h5")
        if model.model_fit and lstm_model.model_fit:
            predictions_gb = model.predict(X)
            predictions_lstm = lstm_model.predict(X)
            combined_predictions = (predictions_gb + predictions_lstm) / 2
            sentiment_factor = 0.5  # Placeholder for sentiment
            adjusted_predictions = combined_predictions * sentiment_factor
            df = df.iloc[:-1]
            df['Predicted_Close'] = adjusted_predictions
            df['Actual_Close'] = y
            df['Signal'] = df['Predicted_Close'] > df['Actual_Close']
            # Incorporate sentiment if available
            df['Sentiment'] = 0  # Placeholder
            # Simple strategy: Buy when Signal is True
            df['Position'] = df['Signal'].astype(int).diff()
            df['Strategy_Return'] = df['Position'].shift(1) * df['Actual_Close'].pct_change()
            df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
            logging.info(f"Backtesting completed for {symbol}.")
            return df
        else:
            logging.error("One or both models are not fitted. Cannot perform backtesting.")
            return None
    except Exception as e:
        logging.error(f"Error during backtesting for {symbol}: {e}", exc_info=True)
        return None

# ----------------------------------------
# User Interface with Dash
# ----------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("CryptoAI Trading Bot Dashboard"),
    dcc.Dropdown(
        id='symbol-dropdown',
        options=[{'label': f"{symbol}", 'value': f"{symbol}"} for symbol in COINS],
        value=COINS[0] if COINS else 'BTC'
    ),
    dcc.Graph(id='price-chart'),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('price-chart', 'figure'),
    [Input('symbol-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(selected_symbol, n):
    try:
        df = pd.read_csv(f"data/{selected_symbol.replace('/', '_')}_data.csv", parse_dates=['timestamp'], index_col='timestamp')
        df = feature_engineering(df)
        if df.empty:
            logging.error(f"No data available for {selected_symbol}.")
            return {}
        X, y = generate_features_and_target(df)
        model = TradingModel(f"models/{selected_symbol}_trading_model.pkl")
        lstm_model = LSTMModel(f"models/{selected_symbol}_lstm_trading_model.h5")
        if model.model_fit and lstm_model.model_fit:
            predictions_gb = model.predict(X)
            predictions_lstm = lstm_model.predict(X)
            combined_predictions = (predictions_gb + predictions_lstm) / 2
            sentiment_factor = 0.5  # Placeholder for sentiment
            adjusted_prediction = combined_predictions * sentiment_factor
            df = df.iloc[:-1]
            df['Predicted_Close'] = adjusted_prediction
            df['Actual_Close'] = y
            fig = {
                'data': [
                    {'x': df.index, 'y': df['close'], 'type': 'line', 'name': 'Close Price'},
                    {'x': df.index, 'y': df['Predicted_Close'], 'type': 'line', 'name': 'Predicted Close'}
                ],
                'layout': {
                    'title': f"{selected_symbol} Price Chart"
                }
            }
            return fig
        else:
            logging.error(f"Models not fitted for {selected_symbol}.")
            return {}
    except FileNotFoundError:
        logging.error(f"Data file for {selected_symbol} not found.")
        return {}
    except Exception as e:
        logging.error(f"Error updating graph for {selected_symbol}: {e}", exc_info=True)
        return {}

# ----------------------------------------
# Main Execution
# ----------------------------------------
async def main_execution():
    logging.info("Starting the Crypto Bot Execution.")

    # Initialize exchanges
    exchanges = {}

    def initialize_exchanges():
        # KuCoin
        if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE:
            exchanges['kucoin'] = ccxt.kucoin({
                'apiKey': KUCOIN_API_KEY,
                'secret': KUCOIN_API_SECRET,
                'password': KUCOIN_API_PASSPHRASE,
                'enableRateLimit': True,
            })
            logging.info("Initialized KuCoin exchange.")
        else:
            logging.warning("KuCoin API credentials not found. Skipping KuCoin exchange.")

        # MEXC
        if MEXC_API_KEY and MEXC_API_SECRET:
            exchanges['mexc'] = ccxt.mexc({
                'apiKey': MEXC_API_KEY,
                'secret': MEXC_API_SECRET,
                'enableRateLimit': True,
            })
            logging.info("Initialized MEXC exchange.")
        else:
            logging.warning("MEXC API credentials not found. Skipping MEXC exchange.")

    initialize_exchanges()

    if not exchanges:
        logging.error("No exchanges initialized. Please check your API credentials.")
        return

    # Initialize Dash app in background
    asyncio.create_task(run_dash_app())

    # Define a semaphore to limit concurrent tasks
    MAX_CONCURRENT_TASKS = 5  # Adjust based on your system's capability
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def run_with_semaphore(coro, semaphore):
        async with semaphore:
            await coro

    while True:
        try:
            if USE_DYNAMIC_COINS:
                actual_coins = await fetch_top_coins(limit=10)  # Adjust the limit as needed
            else:
                actual_coins = [f"{c}/{BASE_SYMBOL}" for c in COINS if c.upper() != 'USDT']  # Ensure correct format

            tasks = []
            trade_count = 0
            daily_loss = 0
            for symbol in actual_coins:
                if trade_count >= MAX_DAILY_TRADES:
                    logging.warning("Max daily trades limit reached. No more trades will be made this cycle.")
                    break

                # Determine which exchange supports the symbol
                supported_exchanges = []
                for exchange_name, exchange_instance in exchanges.items():
                    try:
                        await exchange_instance.load_markets()
                        if symbol in exchange_instance.symbols:
                            supported_exchanges.append(exchange_instance)
                    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                        logging.error(f"Error loading markets for {exchange_name}: {e}", exc_info=True)
                        continue

                if not supported_exchanges:
                    logging.warning(f"No supported exchanges found for symbol {symbol}. Skipping.")
                    continue

                # For simplicity, choose the first supported exchange
                selected_exchange = supported_exchanges[0]

                # Initialize and run TradingBot
                bot = TradingBot(config, symbol, selected_exchange)
                task = asyncio.create_task(run_with_semaphore(bot.run(), semaphore))
                tasks.append(task)
                trade_count += 1

            # Run all bot tasks concurrently
            if tasks:
                await asyncio.gather(*tasks)
            else:
                logging.info("No valid trading symbols to process.")

            logging.info("Crypto Bot execution completed for this cycle.")
            logging.info(f"Sleeping for {LOOP_INTERVAL} seconds before next cycle.")
            await asyncio.sleep(LOOP_INTERVAL)
        except Exception as e:
            logging.error(f"Error in main_execution loop: {e}", exc_info=True)
            logging.info(f"Sleeping for {LOOP_INTERVAL} seconds before retrying.")
            await asyncio.sleep(LOOP_INTERVAL)

async def run_dash_app():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, app.run_server, '0.0.0.0', 8050)

# ----------------------------------------
# Fetch Top Coins Function
# ----------------------------------------
async def fetch_top_coins(limit=10):
    """
    Fetches the top 'limit' coins by market capitalization from CoinGecko.

    Parameters:
        limit (int): The number of top coins to fetch.

    Returns:
        List[str]: A list of coin symbols in the format "SYMBOL/USDT" (e.g., "BTC/USDT").
    """
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': 'false'
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logging.error(f"Failed to fetch top coins: HTTP {response.status}")
                    return []
                data = await response.json()
                top_coins = [f"{coin['symbol'].upper()}/{BASE_SYMBOL}" for coin in data]
                logging.info(f"Fetched top {limit} coins: {top_coins}")
                return top_coins
        except Exception as e:
            logging.error(f"Exception while fetching top coins: {e}", exc_info=True)
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
