# crypto_bot.py

import os
import sys
import json
import yaml
import time
import logging
import traceback
import asyncio
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from ta.volatility import TrueRangeIndicator, BollingerBands
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator

# ----------------------------------------
# Load Environment Variables
# ----------------------------------------

load_dotenv()

# ----------------------------------------
# Configuration
# ----------------------------------------

def load_config(local=False):
    config_file = 'config.local.yaml' if local else 'config.yaml'
    if not os.path.exists(config_file):
        logging.error(f"Configuration file {config_file} not found.")
        sys.exit(1)
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# ----------------------------------------
# Logging Setup
# ----------------------------------------

LOG_LEVEL = config.get("logging", {}).get("level", "INFO").upper()
LOG_FILE = config.get("logging", {}).get("file", "logs/crypto_ai.log")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# ----------------------------------------
# Helper Functions
# ----------------------------------------

def get_config_value(key, default=None):
    """
    Retrieve a configuration value using dot notation.
    """
    keys = key.split('.')
    value = config
    try:
        for k in keys:
            value = value[k]
        return value
    except KeyError:
        return default

# ----------------------------------------
# Data Fetching
# ----------------------------------------

class DataSource:
    def __init__(self, data_source, symbol, timeframe, lookback):
        self.data_source = data_source
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback

    async def get_data(self):
        if self.data_source.lower() == 'cryptocompare':
            return await self.fetch_from_cryptocompare()
        elif self.data_source.lower() == 'coingecko':
            return await self.fetch_from_coingecko()
        elif self.data_source.lower() == 'simulated':
            return self.generate_fake_data()
        else:
            logging.error(f"Unsupported data source: {self.data_source}")
            return None

    async def fetch_from_cryptocompare(self):
        logging.info("Fetching data from CryptoCompare...")
        api_key = os.getenv('CRYPTOCOMPARE_API_KEY')
        url = f"https://min-api.cryptocompare.com/data/v2/histohour"
        params = {
            'fsym': self.symbol.split('/')[0],
            'tsym': self.symbol.split('/')[1],
            'limit': self.lookback - 1,
            'api_key': api_key
        }
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['Response'] == 'Success':
                            df = pd.DataFrame(data['Data']['Data'])
                            df['time'] = pd.to_datetime(df['time'], unit='s')
                            df.rename(columns={
                                'time': 'timestamp',
                                'open': 'open',
                                'high': 'high',
                                'low': 'low',
                                'close': 'close',
                                'volumefrom': 'volume'
                            }, inplace=True)
                            df.set_index('timestamp', inplace=True)
                            logging.info(f"Fetched {len(df)} data points from CryptoCompare.")
                            return df
                        else:
                            logging.error(f"CryptoCompare API error: {data['Message']}")
                            return None
                    else:
                        error_content = await response.text()
                        logging.error(f"Failed to fetch data from CryptoCompare: Status code {response.status}, Response: {error_content}")
                        return None
        except Exception as e:
            logging.error(f"Exception fetching data from CryptoCompare: {e}", exc_info=True)
            return None

    async def fetch_from_coingecko(self):
        logging.info("Fetching data from CoinGecko...")
        # Implement actual data fetching from CoinGecko
        # Placeholder implementation
        return pd.DataFrame()  # Return empty DataFrame for placeholder

    def generate_fake_data(self):
        logging.info("Generating simulated data...")
        date_range = pd.date_range(end=pd.Timestamp.now(), periods=self.lookback, freq=self.timeframe)
        data = {
            'timestamp': date_range,
            'open': np.random.uniform(100, 200, size=self.lookback),
            'high': np.random.uniform(200, 300, size=self.lookback),
            'low': np.random.uniform(50, 100, size=self.lookback),
            'close': np.random.uniform(100, 200, size=self.lookback),
            'volume': np.random.uniform(1000, 5000, size=self.lookback)
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

# ----------------------------------------
# Risk Manager
# ----------------------------------------

class RiskManager:
    def __init__(self, max_investment, stop_loss_pct, take_profit_pct):
        self.max_investment = max_investment
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100

    def calculate_order_size(self, balance, price, min_amount, is_sell=False, position_quantity=0):
        if is_sell:
            return min(balance, position_quantity)
        size = self.max_investment / price
        return max(size, min_amount)

    def get_stop_loss_price(self, entry_price):
        return entry_price * (1 - self.stop_loss_pct)

    def get_take_profit_price(self, entry_price):
        return entry_price * (1 + self.take_profit_pct)

# ----------------------------------------
# Trading Model
# ----------------------------------------

class TradingModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.model_fit = False
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                self.model_fit = True
                logging.info("Loaded existing ML model.")
            except Exception as e:
                logging.error(f"Failed to load ML model: {e}")
                self.model = None
                self.model_fit = False
        else:
            logging.warning("ML model not found. It will be trained.")
            self.model_fit = False

    def train(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV

        if X.empty or y.empty:
            logging.error("Cannot train ML model with empty data.")
            return

        try:
            rf = RandomForestRegressor(random_state=42)
            grid = GridSearchCV(estimator=rf, param_grid=config['model'].get('param_grid', {}), cv=5, n_jobs=-1)
            grid.fit(X, y)
            self.model = grid.best_estimator_
            self.model_fit = True
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                import joblib
                joblib.dump(self.model, f)
            logging.info("Trained and saved ML model.")
        except Exception as e:
            logging.error(f"Failed to train ML model: {e}", exc_info=True)

    def predict(self, X):
        if self.model and self.model_fit:
            return self.model.predict(X)
        else:
            logging.error("ML model is not trained yet.")
            return np.array([1])  # Neutral signal

# ----------------------------------------
# Trading Bot
# ----------------------------------------

class TradingBot:
    def __init__(self, config, symbol):
        self.symbol = symbol
        self.timeframe = config['trading'].get('timeframe', '1H')
        self.lookback = config['trading'].get('lookback', 200)
        self.buy_threshold = config['trading'].get('buy_threshold', 1.002)
        self.sell_threshold = config['trading'].get('sell_threshold', 0.998)
        self.max_daily_trades = config['trading'].get('max_daily_trades', 10)

        self.risk_manager = RiskManager(
            max_investment=config['trading'].get('trade_amount', 1000),
            stop_loss_pct=5,  # Example: 5%
            take_profit_pct=10  # Example: 10%
        )
        self.model = TradingModel(config['model'].get('model_path', 'models/trading_model.pkl'))
        self.position = None  # To track current position
        self.balance = config['trading'].get('trade_amount', 1000)  # Starting balance based on trade_amount

        self.data_source = DataSource(
            data_source=config['api'].get('data_source', 'simulated'),
            symbol=self.symbol,
            timeframe=self.timeframe,
            lookback=self.lookback
        )

        # Initialize exchange within the TradingBot
        self.exchange = ccxt.mexc({
            'apiKey': os.getenv('MEXC_API_KEY'),
            'secret': os.getenv('MEXC_API_SECRET'),
            'enableRateLimit': True,
        })

        # Load existing position if exists
        self.position_file = f"positions/{self.symbol.replace('/', '_')}_position.json"
        self.load_position()

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
        Train the ML model if retrain is set to True and interval met.
        """
        if config['model'].get('retrain', True):
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
            features = ['SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'StochRSI', 'ADX', 'CCI']
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
            if df is None or df.empty:
                logging.error("Data after validation is empty, skipping this iteration.")
                return

            df = self.feature_engineering(df)
            if df is None or df.empty:
                logging.error("Data after feature engineering is empty, skipping this iteration.")
                return

            # Log feature data types
            logging.debug(
                f"Features data types:\n{df[['SMA_10', 'EMA_5', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'StochRSI', 'ADX', 'CCI']].dtypes}"
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
            if signal > self.buy_threshold:
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
            elif signal < self.sell_threshold:
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
                    self.save_position()
                elif side.lower() == 'sell':
                    self.balance += amount * average_price
                    logging.info(f"Sold {amount:.6f} {symbol} at {average_price:.2f}")
                    self.position = None
                    self.save_position()
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
            self.save_position()

# ----------------------------------------
# Utility Functions
# ----------------------------------------

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe using class-based indicators.
    Includes detailed debugging logs.
    """
    try:
        logging.debug(f"Initial dataframe shape: {df.shape}")
        logging.debug(f"Initial dataframe head:\n{df.head()}")

        # Calculate True Range using TrueRangeIndicator
        tr_indicator = TrueRangeIndicator(high=df['high'], low=df['low'], close=df['close'])
        tr = tr_indicator.true_range()
        df = df.assign(TR=tr)
        logging.debug(f"True Range calculated. Non-zero TR count: {(df['TR'] != 0).sum()} out of {len(df)}")
        logging.debug(f"Dataframe head after TR:\n{df.head()}")

        # Remove rows where TR is zero
        df = df[df['TR'] != 0]
        logging.debug(f"Dataframe shape after TR filtering: {df.shape}")
        logging.debug(f"Dataframe head after TR filtering:\n{df.head()}")

        # Add Simple Moving Average
        sma_indicator = SMAIndicator(close=df['close'], window=10)
        df['SMA_10'] = sma_indicator.sma_indicator()
        logging.debug("SMA_10 calculated.")
        logging.debug(f"Dataframe head after SMA_10:\n{df.head()}")

        # Add Exponential Moving Average
        ema_indicator = EMAIndicator(close=df['close'], window=5)
        df['EMA_5'] = ema_indicator.ema_indicator()
        logging.debug("EMA_5 calculated.")
        logging.debug(f"Dataframe head after EMA_5:\n{df.head()}")

        # Add Relative Strength Index
        rsi_indicator = RSIIndicator(close=df['close'], window=5)
        df['RSI'] = rsi_indicator.rsi()
        logging.debug("RSI calculated.")
        logging.debug(f"Dataframe head after RSI:\n{df.head()}")

        # Add MACD
        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        logging.debug("MACD and MACD_signal calculated.")
        logging.debug(f"Dataframe head after MACD:\n{df.head()}")

        # Add Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        logging.debug("Bollinger Bands calculated.")
        logging.debug(f"Dataframe head after Bollinger Bands:\n{df.head()}")

        # Add Stochastic RSI
        stoch = StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3)
        df['StochRSI'] = stoch.stochrsi()
        logging.debug("StochRSI calculated.")
        logging.debug(f"Dataframe head after StochRSI:\n{df.head()}")

        # Add ADX
        adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['ADX'] = adx_indicator.adx()
        logging.debug("ADX calculated.")
        logging.debug(f"Dataframe head after ADX:\n{df.head()}")

        # Add CCI
        cci_indicator = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20)
        df['CCI'] = cci_indicator.cci()
        logging.debug("CCI calculated.")
        logging.debug(f"Dataframe head after CCI:\n{df.head()}")

        # Handle NaN values
        df = df.fillna(0)  # Alternatively, use df.dropna()
        logging.debug(f"Dataframe shape after filling NaNs: {df.shape}")
        logging.debug(f"Final dataframe head:\n{df.head()}")

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
    logging.debug(f"Features data types:\n{X.dtypes}")
    logging.debug(f"Target data type: {y.dtype}")

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
        if config['database']['primary'].lower() == 'postgresql':
            db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DATABASE')}"
        else:
            db_url = f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"
    else:
        if config['database']['secondary'].lower() == 'mysql':
            db_url = f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"
        else:
            db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DATABASE')}"
    
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine

def save_to_database(data, table_name):
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        try:
            engine = connect_database(primary=True)
            data.to_sql(table_name, engine, if_exists='replace', index=False)
            logging.info(f"Data saved to {config['database']['primary'].upper()} table: {table_name}")
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
        logging.info(f"Data saved to {config['database']['secondary'].upper()} table: {table_name}")
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
            if config['crawling']['enabled']:
                await fetch_additional_data()

            # Determine actual coins to trade
            actual_coins = config['trading']['coins']
            if config['trading'].get('dynamic_coins', False):
                actual_coins = await fetch_top_coins()
                logging.info(f"Dynamic coin list: {actual_coins}")

        except Exception as e:
            logging.error(f"Error fetching additional data: {e}", exc_info=True)
            actual_coins = config['trading']['coins']  # Fallback to predefined coins

        daily_trades = 0
        for symbol in actual_coins:
            try:
                # Initialize Trading Bot for each symbol
                bot = TradingBot(config, symbol)

                await bot.run()

                if bot.position:
                    daily_trades += 1
                    if daily_trades >= config['trading']['max_daily_trades']:
                        logging.warning("Max daily trades limit reached. No more trades will be made today.")
                        break
            except Exception as e:
                logging.error(f"Error running TradingBot for {symbol}: {e}", exc_info=True)

        logging.info("Crypto Bot execution completed for this cycle.")

        # Wait for the next cycle
        loop_interval = get_config_value("execution.loop_interval", 60)
        logging.info(f"Sleeping for {loop_interval} seconds before next cycle.")
        await asyncio.sleep(loop_interval)

async def fetch_additional_data():
    """
    Placeholder for web crawling functions.
    Implement actual web crawling as needed.
    """
    logging.info("Fetching additional data via web crawling...")
    # Implement your web crawling logic here
    await asyncio.sleep(1)  # Placeholder

async def fetch_top_coins(limit=10):
    """
    Fetch top coins by market capitalization using CoinGecko API.
    """
    import aiohttp

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
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
                        'apiKey': os.getenv('MEXC_API_KEY'),
                        'secret': os.getenv('MEXC_API_SECRET'),
                        'enableRateLimit': True,
                    })
                    await temp_exchange.load_markets()
                    supported_symbols = list(temp_exchange.markets.keys())
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
                    return config['trading']['coins']  # Fallback to predefined coins
    except Exception as e:
        logging.error(f"Exception fetching top coins: {e}", exc_info=True)
        return config['trading']['coins']  # Fallback to predefined coins

# ----------------------------------------
# Entry Point
# ----------------------------------------

if __name__ == "__main__":
    # Ensure the 'positions' and 'logs' directories exist
    os.makedirs('positions', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Verify 'ta' library functionality
    try:
        from ta.volatility import TrueRangeIndicator
        tr_indicator = TrueRangeIndicator(high=[10, 12, 11, 13], low=[8, 9, 10, 11], close=[9, 11, 10, 12])
        tr = tr_indicator.true_range()
        logging.info("ta.volatility.TrueRangeIndicator is accessible and functioning.")
    except ImportError as e:
        logging.error(f"ta.volatility.TrueRangeIndicator not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error while testing ta library: {e}", exc_info=True)
        sys.exit(1)

    # Start the main execution loop
    try:
        asyncio.run(main_execution())
    except KeyboardInterrupt:
        logging.info("Crypto Bot stopped manually.")
    except Exception as e:
        logging.error(f"Error during execution: {e}\n{traceback.format_exc()}", exc_info=True)
