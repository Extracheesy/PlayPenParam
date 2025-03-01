import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

import matplotlib.pyplot as plt

import ccxt
import os
import talib
import pandas_ta as ta
import time
import threading
import input_data
from datetime import datetime, timezone
from convert_to_xcel import convert_csv_for_excel
import itertools

import utils
import concurrent.futures
import vectorbtpro as vbt

import warnings
warnings.filterwarnings("ignore")

def ichimoku(data, params):
    """
    Computes Ichimoku indicator components (Tenkan-sen, Kijun-sen, Senkou spans,
    Chikou span, and color-coded cloud info).
    """
    high = data.get('high')
    low = data.get('low')
    close = data.get('close')

    conv_period = params['conversion_line_period']
    base_period = params['base_line_periods']
    lag_span = params['lagging_span']
    displacement = params['displacement']

    tenkan_sen = (high.rolling(window=conv_period).max() + low.rolling(window=conv_period).min()) / 2
    kijun_sen = (high.rolling(window=base_period).max() + low.rolling(window=base_period).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    senkou_span_b = ((high.rolling(window=lag_span).max() + low.rolling(window=lag_span).min()) / 2).shift(displacement)
    chikou_span = close.shift(-displacement)

    future_green = (senkou_span_a > senkou_span_b).astype(int)
    future_red = (senkou_span_a < senkou_span_b).astype(int)

    cloud_top = senkou_span_a.combine(senkou_span_b, np.maximum)
    cloud_bottom = senkou_span_a.combine(senkou_span_b, np.minimum)

    dct_ichimoku = {
        'close': close,
        'high': high,
        'low': low,
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span,
        'future_green': future_green,
        'future_red': future_red,
        'cloud_top': cloud_top,
        'cloud_bottom': cloud_bottom
    }

    return dct_ichimoku

def kalman_filter_numpy(series, process_variance=1e-5, measurement_variance=1e-1):
    x = series.values  # Convert to a Numpy array
    n = x.size
    estimates = np.empty(n)

    # Initialize with the first observation
    estimates[0] = x[0]
    posteri_estimate = x[0]
    posteri_error_estimate = 1.0

    for t in range(1, n):
        # Prediction step
        priori_estimate = posteri_estimate
        priori_error_estimate = posteri_error_estimate + process_variance

        # Update step
        blending_factor = priori_error_estimate / (priori_error_estimate + measurement_variance)
        posteri_estimate = priori_estimate + blending_factor * (x[t] - priori_estimate)
        posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        estimates[t] = posteri_estimate

    return pd.Series(estimates, index=series.index)

# Kalman Filter
def kalman_filter(series, process_variance=1e-5, measurement_variance=1e-1):
    n = len(series)
    estimates = [0.0] * n

    # Initialize with the first observation
    posteri_estimate = series.iloc[0]
    posteri_error_estimate = 1.0
    estimates[0] = posteri_estimate

    for t in range(1, n):
        # Prediction step
        priori_estimate = posteri_estimate
        priori_error_estimate = posteri_error_estimate + process_variance

        # Update step
        blending_factor = priori_error_estimate / (priori_error_estimate + measurement_variance)
        posteri_estimate = priori_estimate + blending_factor * (series.iloc[t] - priori_estimate)
        posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        estimates[t] = posteri_estimate

    return pd.Series(estimates, index=series.index)

def ssl_atr(data, period=7):
    """
    Computes the SSL-ATR lines (SSL Down & SSL Up) based on ATR and SMA.

    Returns:
        ssl_down_series (pd.Series)
        ssl_up_series   (pd.Series)
    """
    try:
        high = data.get('high')
        low = data.get('low')
        close = data.get('close')

        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        # Calculate ATR using TA-Lib
        atr = talib.ATR(high, low, close, timeperiod=14)

        # Calculate SMA with ATR adjustment
        sma_high = talib.SMA(high, timeperiod=period) + atr
        sma_low = talib.SMA(low, timeperiod=period) - atr

        # Determine HLV (Higher-Low Value)
        hlv = np.where(close > sma_high, 1, np.where(close < sma_low, -1, np.nan))
        hlv = pd.Series(hlv, index=close.index).ffill()  # Forward fill to handle NaN

        # Calculate SSL Down and SSL Up
        ssl_down = np.where(hlv < 0, sma_high, sma_low)
        ssl_up = np.where(hlv < 0, sma_low, sma_high)

        ssl_down_series = pd.Series(ssl_down, index=close.index)
        ssl_up_series = pd.Series(ssl_up, index=close.index)
    except:
        exit(888)

    return ssl_down_series, ssl_up_series

class CryptoBacktest:
    def __init__(self, id, symbol, start_date, end_date, timeframe='1h', low_timeframe='1m', high_timeframe='1h', save_dir='results', ma_type='SMA', trend_type='ICHIMOKU', params=None,
                 trading_fee=0.001, initial_balance=10000, stop_loss=0.02, print_all=False):
        self.id = id
        self.symbol = symbol
        self.start_date = start_date
        # self.end_date = datetime.utcnow().replace(tzinfo=timezone.utc)
        self.end_date = end_date
        self.timeframe = timeframe
        self.low_timeframe = low_timeframe
        self.high_timeframe = high_timeframe
        self.save_dir = save_dir
        self.data_dir = os.path.join(save_dir, "database")
        self.ma_type = ma_type
        self.trend_type = trend_type
        self.params = params if params else {'zema_len_buy': 20, 'zema_len_sell': 20, 'low_offset': 0.99,
                                             'high_offset': 1.01}
        self.trading_fee = trading_fee  # Trading fee (default 0.1%)
        self.initial_balance = initial_balance
        self.stop_loss = stop_loss  # Stop loss percentage (default 2%)
        self.data = None
        self.trades = []  # Stores trade details

        self.VBT_PLOT = False

        self.save_trades = False
        self.print_all = print_all
        self.show_plot = self.print_all

        self.stats ={}
        self.stats["ID"] = self.id
        self.stats['SYMBOL'] = self.symbol
        self.stats['START_DATE'] = self.start_date
        self.stats['END_DATE'] = self.end_date
        self.stats['TIMEFRAME'] = self.timeframe
        self.stats['MA_TYPE'] = self.ma_type
        self.stats['TREND_TYPE'] = self.trend_type
        self.stats['LOW_TIMEFRAME'] = self.low_timeframe
        self.stats['HIGH_TIMEFRAME'] = self.high_timeframe
        self.stats['STOP_LOSS'] = self.stop_loss
        self.stats['FEES'] = self.trading_fee

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_data_batch(self, exchange, timeframe, since, all_ohlcv, lock):
        limit = 1000
        while True:
            ohlcv = exchange.fetch_ohlcv(self.symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            with lock:
                all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Move to the next timestamp to avoid overlap
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limit

    def fetch_data(self, data_attr, reverse=False):
        """
        Fetches OHLCV data for the specified timeframe and saves it to the specified data attribute.

        Args:
            data_attr (str): The attribute name to store the fetched data (e.g., 'data', 'low_tf_data', 'high_tf_data').
        """
        # Determine the timeframe based on the data attribute
        if data_attr == 'data':
            timeframe = self.timeframe
        elif data_attr == 'low_tf_data':
            timeframe = self.low_timeframe
        elif data_attr == 'high_tf_data':
            timeframe = self.high_timeframe
        else:
            raise ValueError(f"Invalid data_attr: {data_attr}. Must be 'data', 'low_tf_data', or 'high_tf_data'.")

        # Determine the end time
        if self.end_date is None:
            dt_end_date = datetime.utcnow()
        else:
            if isinstance(self.end_date, str):
                dt_end_date = datetime.strptime(self.end_date, "%Y-%m-%dT%H:%M:%SZ")
            else:
                dt_end_date = self.end_date  # Assume it's already a datetime object

            # Round the end date to the nearest timeframe and format it
            self.end_date = utils.round_time(dt_end_date, timeframe)
            self.end_date = self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Format the file name
        rounded_time = utils.round_time(dt_end_date, timeframe)
        formatted_time = rounded_time.strftime('%Y-%m-%d-%H-%M')  # Replace ':' with '-'
        date_with_minute = self.start_date.replace('T', '-')[:16].replace(':', '-')  # Replace ':' with '-'
        data_file = os.path.join(
            self.data_dir,
            f'{self.symbol.replace("/", "_")}_data_{timeframe}_{date_with_minute}_{formatted_time}.csv'
        )

        # Load data from file if it exists
        if os.path.exists(data_file):
            print("Loading data from file...")
            setattr(self, data_attr, pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp'))
            return

        # Initialize the exchange
        exchange = ccxt.binance()

        # Parse the start date
        since = exchange.parse8601(self.start_date)

        # Initialize an empty list to store all OHLCV data
        all_ohlcv = []

        # Create a lock for thread-safe operations
        lock = threading.Lock()

        def fetch_data_batch(exchange, timeframe, since, all_ohlcv, lock, end_timestamp=None):
            """
            Fetches OHLCV data in batches and appends it to the all_ohlcv list.
            """
            while True:
                with lock:
                    # Fetch OHLCV data
                    ohlcv = exchange.fetch_ohlcv(
                        self.symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=1000,  # Adjust the limit as needed
                        params={'endTime': end_timestamp} if end_timestamp else None
                    )
                    if not ohlcv:
                        break  # No more data to fetch
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1  # Update since to the next timestamp

        # If end_date is provided, parse it into a timestamp
        end_timestamp = exchange.parse8601(self.end_date) if self.end_date is not None else None

        # Create and start the thread
        thread = threading.Thread(
            target=fetch_data_batch,
            args=(exchange, timeframe, since, all_ohlcv, lock, end_timestamp)
        )
        thread.start()
        thread.join()

        # Convert the OHLCV data into a DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)

        # Filter the DataFrame based on end_date (if provided)
        if self.end_date is not None:
            df = df[df.index <= pd.to_datetime(self.end_date)]

        if reverse:
            backup_index = df.index.copy()
            df = df[::-1].reset_index(drop=False)
            df.index = backup_index
            df = df.drop('timestamp', axis=1)
            df.rename(columns={'open': 'close', 'close': 'open'}, inplace=True)

        # Assign the DataFrame to the specified attribute
        setattr(self, data_attr, df)

        # Save the fetched data to a file
        print("Saving fetched data to file...")
        getattr(self, data_attr).to_csv(data_file)

    def apply_indicator(self):
        if self.data is None:
            raise ValueError("No data available. Fetch data first.")
        close_data = self.data['close']

        if self.ma_type == "ZLEMA":
            ma_buy = talib.EMA(close_data, timeperiod=self.params['zema_len_buy']) * self.params['low_offset']
            ma_sell = talib.EMA(close_data, timeperiod=self.params['zema_len_sell']) * self.params['high_offset']
        elif self.ma_type in ["ZLMA", "TEMA", "DEMA", "ALMA", "KAMA", "HMA"]:
            if self.ma_type == "ZLMA":
                zlma_buy = ta.zlma(close_data, timeperiod=self.params['zema_len_buy'], mamode="ema")
                zlma_sell = ta.zlma(close_data, timeperiod=self.params['zema_len_sell'], mamode="ema")
                ma_buy = zlma_buy * self.params['low_offset']
                ma_sell = zlma_sell * self.params['high_offset']

            elif self.ma_type == "TEMA":
                tema_buy = ta.tema(close_data, timeperiod=self.params['zema_len_buy'])
                tema_sell = ta.tema(close_data, timeperiod=self.params['zema_len_sell'])
                ma_buy = tema_buy * self.params['low_offset']
                ma_sell = tema_sell * self.params['high_offset']

            elif self.ma_type == "DEMA":
                dema_buy = ta.dema(close_data, timeperiod=self.params['zema_len_buy'])
                dema_sell = ta.dema(close_data, timeperiod=self.params['zema_len_sell'])
                ma_buy = dema_buy * self.params['low_offset']
                ma_sell = dema_sell * self.params['high_offset']

            elif self.ma_type == "ALMA":
                alma_buy = ta.alma(close_data, timeperiod=self.params['zema_len_buy'])
                alma_sell = ta.alma(close_data, timeperiod=self.params['zema_len_sell'])

                ma_buy = alma_buy * self.params['low_offset']
                ma_sell = alma_sell * self.params['high_offset']

            elif self.ma_type == "KAMA":
                kama_buy = ta.kama(close_data, timeperiod=self.params['zema_len_buy'])
                kama_sell = ta.kama(close_data, timeperiod=self.params['zema_len_sell'])
                ma_buy = kama_buy * self.params['low_offset']
                ma_sell = kama_sell * self.params['high_offset']

            elif self.ma_type == "HMA":
                hma_buy = ta.hma(close_data, timeperiod=self.params['zema_len_buy'])
                hma_sell = ta.hma(close_data, timeperiod=self.params['zema_len_sell'])
                ma_buy = hma_buy * self.params['low_offset']
                ma_sell = hma_sell * self.params['high_offset']
        else:
            raise ValueError(f"Unknown moving average type: {self.ma_type}")

        self.data['buy_adj'] = ma_buy
        self.data['sell_adj'] = ma_sell

    def apply_high_tf_indicator(self):
        if self.high_tf_data is None:
            raise ValueError("No data available. Fetch data first.")

        if self.trend_type == 'ICHIMOKU':
            ssl_down, ssl_up = ssl_atr(self.high_tf_data, period=self.params['ssl_atr_period'])
            ssl_ok = (ssl_up > ssl_down).astype(int)
            ssl_bear = (ssl_up < ssl_down).astype(int)

            ichimoku_dct = ichimoku(self.high_tf_data, self.params['ichimoku_params'])
            ichimoku_ok = (
                    (ichimoku_dct['tenkan_sen'] > ichimoku_dct['kijun_sen']) &
                    (ichimoku_dct['close'] > ichimoku_dct['cloud_top']) &
                    (ichimoku_dct['future_green'] > 0) &
                    (
                            ichimoku_dct['chikou_span'] >
                            ichimoku_dct['cloud_top'].shift(
                                -self.params['ichimoku_params']['displacement']
                            )
                    )
            ).astype(int)

            ichimoku_bear = (
                    (ichimoku_dct['tenkan_sen'] < ichimoku_dct['kijun_sen']) &
                    (ichimoku_dct['close'] < ichimoku_dct['cloud_bottom']) &
                    (ichimoku_dct['future_red'] > 0) &
                    (
                            ichimoku_dct['chikou_span'] <
                            ichimoku_dct['cloud_bottom'].shift(
                                -self.params['ichimoku_params']['displacement']
                            )
                    )
            ).astype(int)

            ichimoku_valid = (~ichimoku_dct['senkou_span_b'].isna()).astype(int).fillna(0)
            trend_pulse = ((ichimoku_ok > 0) & (ssl_ok > 0)).astype(int).fillna(0)
            bear_trend_pulse = ((ichimoku_bear > 0) & (ssl_bear > 0)).astype(int).fillna(0)

            self.ichimoku_valid = ichimoku_valid.reindex(self.data.index, method='ffill')
            self.trend_pulse = trend_pulse.reindex(self.data.index, method='ffill')
            self.bear_trend_pulse = bear_trend_pulse.reindex(self.data.index, method='ffill')

            self.data["ichimoku_valid"] = self.ichimoku_valid
            self.data["trend_pulse"] = self.trend_pulse
            self.data["bear_trend_pulse"] = self.bear_trend_pulse
        else:
            if self.trend_type == 'TSI':
                # True Strength Index (TSI)
                df_tsi = ta.tsi(self.high_tf_data['close'], long=25, short=13)
                # Dynamically grab the first two columns regardless of their names
                raw_tsi_col, signal_tsi_col = df_tsi.columns[:2]
                # Create a trend difference column
                df_tsi['trend_diff'] = df_tsi[raw_tsi_col] - df_tsi[signal_tsi_col]

                # Create a trend column: 1 if uptrend (raw TSI > signal), -1 if downtrend
                trend_array = np.where(df_tsi[raw_tsi_col] > df_tsi[signal_tsi_col], 1, -1)

            elif self.trend_type == 'FISHER':
                # Fisher Transform
                df_fisher = ta.fisher(self.high_tf_data['high'], self.high_tf_data['low'], length=9)

                # Dynamically grab the first two columns regardless of their names
                raw_fisher_col, signal_fisher_col = df_fisher.columns[:2]

                # Create a trend difference column
                df_fisher['trend_diff'] = df_fisher[raw_fisher_col] - df_fisher[signal_fisher_col]

                # Create a trend column: 1 if uptrend (raw Fisher > signal Fisher), -1 if downtrend
                trend_array = np.where(df_fisher[raw_fisher_col] > df_fisher[signal_fisher_col], 1, -1)

            elif self.trend_type == 'KALMAN':
                # Kalman Filter: often you compare the price to the filter output.
                self.high_tf_data['Kalman'] = kalman_filter_numpy(self.high_tf_data['close'])
                # Bullish if the current close is above the Kalman filter value, bearish otherwise.
                trend_array = np.where(self.high_tf_data['close'] > self.high_tf_data['Kalman'], 1, -1)

            elif self.trend_type == 'SAR':
                # Compute Parabolic SAR using TA-Lib.
                self.high_tf_data["SAR"] = talib.SAR(self.high_tf_data["high"], self.high_tf_data["low"], acceleration=0.02,
                                                     maximum=0.2)
                # For SAR, if the close is above the SAR value, the trend is bullish; if below, bearish.
                trend_array = np.where(self.high_tf_data['close'] > self.high_tf_data['SAR'], 1, -1)
            elif self.trend_type == 'PRICE_ACTION':
                # Compute highest and lowest window over 10 periods
                self.high_tf_data["highest_window"] = self.high_tf_data["high"].rolling(window=10).max()
                self.high_tf_data["lowest_window"] = self.high_tf_data["low"].rolling(window=10).min()

                # Identify buy and sell signals
                buy_signal = self.high_tf_data["close"] > self.high_tf_data["highest_window"].shift(1)
                sell_signal = self.high_tf_data["close"] < self.high_tf_data["lowest_window"].shift(1)

                # Create trend array based on price action
                trend_array = np.zeros(len(self.high_tf_data), dtype=int)
                trend_array[buy_signal] = 1  # Uptrend when close breaks highest_window
                trend_array[sell_signal] = -1  # Downtrend when close breaks lowest_window

            trend_series = pd.Series(trend_array, index=self.high_tf_data.index)
            self.trend = trend_series.reindex(self.data.index, method='ffill')
            self.data["TREND"] = self.trend

    def set_signals(self):
        if self.trend_type == 'ICHIMOKU':
            self.data['buy_signal'] = np.where(
                (self.data['ichimoku_valid'] > 0) &
                (self.data['bear_trend_pulse'] == 0) &
                (self.data['close'] < self.data['buy_adj']),
                True,
                False
            )

            # Create sell_signal column:
            # True if close > sell_adj, otherwise False
            self.data['sell_signal'] = np.where(
                self.data['close'] > self.data['sell_adj'],
                True,
                False
            )
        else:
            self.data['buy_signal'] = np.where(
                (self.data['TREND'] == 1) &
                (self.data['close'] < self.data['buy_adj']),
                True,
                False
            )

            # Create sell_signal column:
            # True if close > sell_adj, otherwise False
            self.data['sell_signal'] = np.where(
                self.data['close'] > self.data['sell_adj'],
                True,
                False
            )

    def backtest_strategy(self):
        """
        Run a backtest on the trading strategy with fees and stop-loss properly applied.

        Assumptions:
          - self.trading_fee is in percentage points (e.g., 0.04 means 0.04% fee).
          - self.stop_loss is in percentage points (e.g., 2 means a 2% stop loss).
          - All available balance is used for each trade.
        """
        # Ensure necessary indicator columns exist
        if self.data is None or 'buy_adj' not in self.data:
            raise ValueError("Apply indicator before backtesting.")

        self.set_signals()

        if self.stop_loss != 0 \
                or True:
            # Reindex self.data to 1-minute resolution and forward-fill the missing values
            self.data_reindexed = self.data.reindex(self.low_tf_data.index)

            self.data_reindexed['close_low_tf'] = self.low_tf_data['close']

            self.data_reindexed['buy_signal'] = self.data_reindexed['buy_signal'].fillna(False)
            self.data_reindexed['sell_signal'] = self.data_reindexed['sell_signal'].fillna(False)

            self.data = self.data_reindexed

        self.data['strategy_returns'] = 0.0

        self.data_for_vbt = self.data.copy()

        # Initialize performance metrics
        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        num_trades = 0
        num_wins = 0
        trade_pnls = []

        # Trading state variables
        position = 0  # 0: no position, 1: in position
        buy_price = None
        stop_loss_price = None
        position_size = None

        # Loop through the dataset
        for i in range(1, len(self.data)):
            current_close = self.data['close'].iloc[i]
            current_close_low_tf = self.data['close_low_tf'].iloc[i]

            # --- Entry Condition ---
            if position == 0:
                # Example entry conditions:
                if self.data['buy_signal'].iloc[i]:
                    position = 1
                    buy_price = current_close

                    # Calculate stop loss price using stop_loss percentage.
                    # Convert stop_loss from percentage points to a decimal.
                    stop_loss_price = buy_price * (1 - self.stop_loss / 100)

                    # Deduct fee on the buy side.
                    # Convert trading_fee from percentage points to a decimal.
                    effective_balance = balance * (1 - self.trading_fee / 100)
                    # Calculate the number of asset units purchased.
                    position_size = effective_balance / buy_price

                    # Record the buy trade.
                    self.trades.append({
                        'type': 'BUY',
                        'price': buy_price,
                        'date': self.data.index[i],
                        'position_size': position_size
                    })
                    self.data.at[self.data.index[i], 'buy_signal'] = buy_price

            # --- Exit Condition ---
            elif position == 1:
                sell_signal = False
                # Check for sell condition or stop loss hit.
                if self.data['sell_signal'].iloc[i]:
                    sell_signal = True
                    sell_price = current_close
                elif self.stop_loss != 0 \
                        and current_close_low_tf < stop_loss_price:
                    sell_signal = True
                    sell_price = current_close_low_tf

                if sell_signal:
                    # When selling, deduct fee on the sell side as well.
                    # Calculate the net multiplier of the round-trip trade.
                    net_multiplier = (sell_price / buy_price) * (1 - self.trading_fee / 100) ** 2
                    trade_return = net_multiplier - 1  # Trade return as a fraction

                    # Update overall balance.
                    balance = balance * net_multiplier

                    # Record trade performance.
                    trade_pnls.append(trade_return)
                    num_trades += 1
                    if trade_return > 0:
                        num_wins += 1

                    # Update drawdown (max drop from the peak balance).
                    peak_balance = max(peak_balance, balance)
                    drawdown = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, drawdown)

                    # Save the trade return in the data.
                    self.data.at[self.data.index[i], 'strategy_returns'] = trade_return

                    # Record the sell trade.
                    self.trades.append({
                        'type': 'SELL',
                        'price': sell_price,
                        'date': self.data.index[i],
                        'position_size': position_size
                    })
                    self.data.at[self.data.index[i], 'sell_signal'] = sell_price

                    # Reset position state
                    position = 0
                    buy_price = None
                    stop_loss_price = None
                    position_size = None

        # After looping, calculate overall performance metrics and output results.
        self._calculate_performance_metrics(balance, trade_pnls, num_wins, num_trades, max_drawdown)
        self._save_and_print_results()

    def _calculate_performance_metrics(self, balance, trade_pnls, num_wins, num_trades, max_drawdown):
        self.data['cumulative_strategy_returns'] = (1 + self.data['strategy_returns']).cumprod()
        self.data['cumulative_buy_hold'] = (1 + self.data['close'].pct_change()).cumprod()

        buy_hold_return = (self.data['close'].iloc[-1] - self.data['close'].iloc[0]) / self.data['close'].iloc[0]
        avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if num_wins > 0 else 0
        avg_loss = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if (num_trades - num_wins) > 0 else 0
        best_trade = max(trade_pnls) if trade_pnls else 0
        worst_trade = min(trade_pnls) if trade_pnls else 0

        win_rate = num_wins / num_trades if num_trades > 0 else 0
        final_balance = balance
        cumulative_return = (final_balance - self.initial_balance) / self.initial_balance

        returns = self.data['strategy_returns'].dropna()
        sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)

        if self.print_all:
            print("Trades Executed:")
            for trade in self.trades:
                print(trade)
            print(f"Final Balance: {final_balance:.2f}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Total Trades: {num_trades}")
            print(f"Cumulative Return: {cumulative_return:.2%}")
            print(f"Buy & Hold Return: {buy_hold_return:.2%}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Sortino Ratio: {sortino_ratio:.2f}")
            print(f"Average Winning Trade: {avg_win:.2%}")
            print(f"Average Losing Trade: {avg_loss:.2%}")
            print(f"Best Trade: {best_trade:.2%}")
            print(f"Worst Trade: {worst_trade:.2%}")

        self.stat_metrics ={}
        self.stat_metrics["M_FINAL_BALANCE"] = f"{final_balance:.2f}"
        self.stat_metrics['M_WIN_RATE'] = f"{win_rate:.2%}"
        self.stat_metrics['M_NUM_TRADES'] = num_trades
        self.stat_metrics['M_CUMULATIVE_RETURN'] = f"{cumulative_return:.2%}"
        self.stat_metrics['M_BUY_HOLD_RETURN'] = f"{buy_hold_return:.2%}"
        self.stat_metrics['M_MAX_DRAWDOWN'] = f"{max_drawdown:.2%}"
        self.stat_metrics['M_AVG_WIN'] = f"{avg_win:.2%}"
        self.stat_metrics['M_AVG_LOSS'] = f"{avg_loss:.2%}"
        self.stat_metrics['M_BEST_TRADE'] = f"{best_trade:.2%}"
        self.stat_metrics['M_WORST_TRADE'] = f"{worst_trade:.2%}"

        self.trade_pnls = trade_pnls

    def get_metrics(self):
        return self.stats | self.stat_metrics | self.dct_pf_stats

    def _calculate_sortino_ratio(self, returns):
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        return returns.mean() / downside_std if downside_std != 0 else 0

    def _save_and_print_results(self):
        if self.print_all:
            self.data_backtest = self.data.copy()
            self.data_backtest = self.data_backtest.set_index(self.data.index)

            saved_path = utils.save_dataframe(self.data, "output_data", "data_backtest", "csv")
            print(f"DataFrame saved at: {saved_path}")

    def plot_pnl(self):
        trade_pnls = self.trade_pnls
        plt.figure(figsize=(12, 5))
        trade_indices = range(len(trade_pnls))
        colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
        plt.scatter(trade_indices, trade_pnls, c=colors, alpha=0.6, s=80, edgecolors='black')
        plt.axhline(0, color='black', linestyle='dashed', alpha=0.5)
        plt.title("Trade PnL")
        plt.xlabel("Trade Number")
        plt.ylabel("PnL (%)")
        plt.grid()

        save_path = os.path.join(self.save_dir,
                                 f'{self.id}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_pnl.png')
        plt.savefig(save_path)

        if self.show_plot:
            plt.show()
        print(f"PnL Graph saved at {save_path}")

    def plot_results(self):
        with plot_io_lock:
            if self.data is None:
                raise ValueError("No data to plot.")

            plt.figure(figsize=(12, 6))
            plt.plot(self.data.index, self.data['cumulative_strategy_returns'], label='Strategy Returns', color='blue')
            plt.plot(self.data.index, self.data['cumulative_buy_hold'], label='Buy & Hold Returns', color='black',
                     linestyle='dashed')
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.title(f'{self.symbol} Backtest Results')
            plt.grid()

            save_path = os.path.join(self.save_dir,
                                     f'{self.id}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_returns.png')
            plt.savefig(save_path)
            if self.show_plot:
                plt.show()
            print(f"Graph saved at {save_path}")

            if False:
                plt.figure(figsize=(12, 6))
                plt.plot(self.data.index, self.data['close'], label='Close Price', color='blue', alpha=0.5)
                plt.plot(self.data.index, self.data['buy_adj'], label='Buy Adj', color='green', linestyle='dashed')
                plt.plot(self.data.index, self.data['sell_adj'], label='Sell Adj', color='red', linestyle='dashed')
                plt.scatter(self.data.index, self.data['buy_signal'], label='Buy Signal', marker='^', color='lime', s=80)
                plt.scatter(self.data.index, self.data['sell_signal'], label='Sell Signal', marker='v', color='red', s=80)
                plt.legend()
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.title(f'{self.symbol} Buy/Sell Signals with {self.ma_type} Indicator')
                plt.grid()

                save_path_signals = os.path.join(self.save_dir,
                                                 f'{self.id}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_signals.png')

                plt.savefig(save_path_signals)
                if self.show_plot:
                    plt.show()
                print(f"Graph saved at {save_path_signals}")

    def __get_frequency(self):
        inferred_freq = pd.infer_freq(self.data.index)

        # Convert inferred frequency to your mapping keys
        frequency_conversion = {
            'T': '1m',
            '5T': '5m',
            '15T': '15m',
            '30T': '30m',
            'H': '1h',
            '2H': '2h'
        }

        self.timeframe = frequency_conversion.get(inferred_freq, None)
        return self.timeframe

    # Example usage in your class or function:
    def get_frequency(self):
        timeframe = self.__get_frequency()
        if timeframe == None:
            self.timeframe = utils.infer_timeframe(self.data.index)
        else:
            self.timeframe = timeframe
        return self.timeframe

    def backtest_vbt_strategy(self):
        """
        self.buy_signal_long = (
                (self.data['ichimoku_valid'] > 0) &
                (self.data['bear_trend_pulse'] == 0) &
                (self.data['close'] < self.data['buy_adj'])
        ).astype(int)
        entries_long = self.buy_signal_long == 1

        self.sell_signal_long = (self.data['close'] > self.data['sell_adj']).astype(int)
        exits_long = self.sell_signal_long == 1
        """
        self.data_for_vbt["close_vbt"] = np.where(
            self.data_for_vbt["close"].isnull(),  # Check if "close" is NaN
            self.data_for_vbt["close_low_tf"],  # If True, use value from "close_low_tf"
            self.data_for_vbt["close"]  # If False, use value from "close"
        )

        entries_long = self.data_for_vbt["buy_signal"]
        exits_long = self.data_for_vbt["sell_signal"]

        self.get_frequency()
        freq = self.timeframe

        # Define common parameters for the portfolio
        portfolio_params = {
            'close': self.data_for_vbt['close_vbt'],
            'entries': entries_long,
            'exits': exits_long,
            'freq': freq,
            'init_cash': 10000,
            'size': 100,
            'size_type': "Percent100"
        }

        # Add trading fee and stop loss if they are not zero
        if self.trading_fee != 0:
            portfolio_params['fees'] = self.trading_fee / 100
        if self.stop_loss != 0:
            portfolio_params['sl_stop'] = self.stop_loss / 100

        # Create the portfolio
        self.pf = vbt.Portfolio.from_signals(**portfolio_params)

        if self.print_all:
            print(self.pf.stats())

        self.dct_pf_stats = dict(self.pf.stats())

        if self.save_trades:
            # Access the trades
            trades = self.pf.trades
            # Convert trades to a DataFrame
            trades_df = trades.records_readable
            trades_df.to_csv('./test_5m/trades_vbt_with_with_fees.csv', index=False)

    def plot_vbt_results(self):
        if self.VBT_PLOT:
            """
            Returns a list of stats dictionaries (one per symbol).
            Includes the vectorbt stats and any custom stats from self.lst_of_my_result.
            """
            pf = self.pf
            # stats = dict(pf.stats())  # vbt-pro stats

            fig = pf.plot()
            if False:
                fig.show()
            else:
                filename = os.path.join(self.save_dir,
                                        f'{self.id}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_vbt.html')

                # full_path = os.path.join(self.save_dir, filename)
                fig.write_html(filename)

def set_df_with_missing_data(df):
    # Define your lists
    lst_STOP_LOSS = [0, 2]
    lst_FEES = [0, 0.04]
    lst_TREND_TYPE = ["ICHIMOKU", "SAR", "TSI", "FISHER", "KALMAN", "PRICE_ACTION"]

    # This list will store all new rows
    rows_list = []

    # Iterate over each row in the original dataframe
    for _, row in df.iterrows():
        # For each combination of STOP_LOSS, FEES, and TREND_TYPE, create a new row
        for stop_loss, fee, trend in itertools.product(lst_STOP_LOSS, lst_FEES, lst_TREND_TYPE):
            new_row = row.copy()  # Copy the original row to preserve its data
            new_row["STOP_LOSS"] = stop_loss
            new_row["FEES"] = fee
            new_row["TREND_TYPE"] = trend
            rows_list.append(new_row)

    # Create a new DataFrame from the list of rows
    new_df = pd.DataFrame(rows_list)
    # Check if 'id' exists; if so, drop it.
    if 'id' in new_df.columns:
        new_df.drop(columns=['id'], inplace=True)

    # Now add the 'id' column with a sequential range starting at 0.
    new_df['id'] = range(len(new_df))
    new_df.columns = new_df.columns.str.upper()
    new_df = new_df.reset_index(drop=True)
    return new_df

# Create a lock for file I/O operations
file_io_lock = threading.Lock()
file_io_lock_2 = threading.Lock()
plot_io_lock = threading.Lock()

# Define your per-parameter processing function
def process_param(param, start_date, end_date, low_timeframe, high_timeframe, save_dir, input_data, file_io_lock):
    print(f"[process_param] ID={param['ID']} on thread {threading.current_thread().name}")
    # Unpack parameters from the param dictionary
    id = param["ID"]
    symbol = param["SYMBOL"]
    timeframe = param["TIMEFRAME"]
    ma_type = param["MA_TYPE"]
    trend_type = param["TREND_TYPE"]
    ZEMA_LEN_BUY = int(param["ZEMA_LEN_BUY"])
    ZEMA_LEN_SELL = int(param["ZEMA_LEN_SELL"])
    SSL_ATR_PERIOD = int(param["SSL_ATR_PERIOD"])

    # Convert each parameter safely
    HIGH_OFFSET = utils.safe_float(param["HIGH_OFFSET"])
    LOW_OFFSET = utils.safe_float(param["LOW_OFFSET"])
    FEES = utils.safe_float(param["FEES"])
    STOP_LOSS = utils.safe_float(param["STOP_LOSS"])

    # Build additional parameters dictionary
    params = {
        'zema_len_buy': ZEMA_LEN_BUY,
        'zema_len_sell': ZEMA_LEN_SELL,
        'low_offset': LOW_OFFSET,
        'high_offset': HIGH_OFFSET,
        'ssl_atr_period': SSL_ATR_PERIOD,
        'ichimoku_params': input_data.ichimoku_params  # assuming input_data has this attribute
    }

    # Instantiate your CryptoBacktest object (make sure that CryptoBacktest is imported)
    backtest = CryptoBacktest(
        id,
        symbol,
        start_date,
        end_date,
        timeframe,
        low_timeframe,
        high_timeframe,
        save_dir,
        ma_type,
        trend_type,
        params,
        trading_fee=FEES,
        initial_balance=10000,
        stop_loss=STOP_LOSS
    )

    REVERSE_MODE = False
    with file_io_lock:
        # Fetch and process data
        backtest.fetch_data(data_attr='data', reverse=REVERSE_MODE)
        # The condition below always evaluates to True because of "or True",
        # so adjust it if necessary.
        if STOP_LOSS != 0 or True:
            backtest.fetch_data(data_attr='low_tf_data', reverse=REVERSE_MODE)
        backtest.fetch_data(data_attr='high_tf_data', reverse=REVERSE_MODE)

    # Apply indicators and perform backtests
    backtest.apply_indicator()
    backtest.apply_high_tf_indicator()
    backtest.backtest_strategy()
    backtest.backtest_vbt_strategy()

    # Plot results
    backtest.plot_results()
    # backtest.plot_pnl()
    backtest.plot_vbt_results()

    # Retrieve and return metrics
    dct_stats = backtest.get_metrics()

    del backtest
    print("######################################################")
    return dct_stats

def main():
    MULTI_FROM_CSV = True
    FINAL_SELECTION = False

    if True:
        end_date = datetime.utcnow().replace(tzinfo=timezone.utc)
    else:
        end_date_str = "2025-02-02 18:21:45.707833"
        nd_date_str = "2025-02-07 13:06:22.392898+00:00"
        # Convert the string to a datetime object
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S.%f")

    if FINAL_SELECTION:
        start_date = '2025-01-01T00:00:00Z'
        # start_date = '2024-01-01T00:00:00Z'
        print("start date: ", start_date)
        print("end date: ", end_date)
    else:
        start_date = '2024-01-01T00:00:00Z'
        print("start date: ", start_date)
        print("end date: ", end_date)

    low_timeframe = '1m'
    high_timeframe = '1h'

    if MULTI_FROM_CSV:
        if FINAL_SELECTION:
            working_directory = "./test_multi_trend_selection"
        else:
            working_directory = "./test_multi_trend_reverse"

        working_directory = "./test_multi_trend_selection"

        input_file_csv = "input_data_excel.csv"
        input_file_csv_2 = "input_data_full.csv"
        output_file_csv = "output_data_full_test.csv"
        output_file_csv_tmp = "output_data_full_test_tmp.csv"
        save_dir = "result_test"

        # Join the directory and file name to form the full path
        file_path = os.path.join(working_directory, input_file_csv)
        file_path_2 = os.path.join(working_directory, input_file_csv_2)
        save_dir = os.path.join(working_directory, save_dir)
        file_path_output = os.path.join(save_dir, output_file_csv)
        file_path_output_tmp = os.path.join(save_dir, output_file_csv_tmp)

        # Check if the file exists
        if os.path.exists(file_path):
            print(f"File exists at: {file_path}")
        else:
            print(f"File does not exist at: {file_path}")

        lock = threading.Lock()
        if False:
            df_input_data = utils.read_csv_thread_safe(file_path, lock)
            # df_input_data = set_df_with_missing_data(df_input_data)
            # df_input_data = df_input_data.sample(frac=1, random_state=42).reset_index(drop=True)
            df_input_data.to_csv(file_path_2)

            new_excel_path = utils.add_exel_before_csv(file_path_2)
            convert_csv_for_excel(file_path_2, new_excel_path)
        else:
            df_input_data = utils.read_csv_thread_safe(file_path, lock)

            df_input_data = set_df_with_missing_data(df_input_data)
            df_input_data = df_input_data.sample(frac=1, random_state=42).reset_index(drop=True)
            df_input_data = utils.drop_zero_fees(df_input_data)
            df_input_data.to_csv(file_path_2)

            new_excel_path = utils.add_exel_before_csv(file_path_2)
            convert_csv_for_excel(file_path_2, new_excel_path)

        if False:
            df_input_data = df_input_data.head(1)  # For test
            # df_input_data = df_input_data[df_input_data['ID'] == 2]

        if True: # Filter
            directory_path = r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\test_multi_trend_selection\result_test"
            working_directory += working_directory + r"\result_test"

            print("total to perform: ", len(df_input_data))
            # Get the list of prefixes
            id_list = utils.get_numeric_prefixes(directory_path)
            id_list = list(map(int, id_list))
            print("total already performed: ", len(id_list))
            df_input_data = utils.drop_rows_by_id(df_input_data, id_list)
            print("remaining to perform: ", len(df_input_data))


        lst_params = []
        for _, row in df_input_data.iterrows():
            # Convert the row to a dictionary with column names as keys
            param = row.to_dict()
            lst_params.append(param)

        print("nb combination: ", len(df_input_data))
        # Control whether to use multithreading or sequential execution
        multi_threading = True  # Set to False for sequential execution

        # Set the maximum number of threads to run concurrently
        max_threads = 4

        lst_stats = []  # This will store the metrics from each backtest

        if multi_threading:
            print("multithreading is on: ", max_threads)
            # Use ThreadPoolExecutor with a limit on the maximum number of threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                # Submit tasks: pass all required parameters to process_param
                futures = [
                    executor.submit(
                        process_param,
                        param,
                        start_date,
                        end_date,
                        low_timeframe,
                        high_timeframe,
                        save_dir,
                        input_data,
                        file_io_lock
                    )
                    for param in lst_params  # lst_params should be defined with your parameters
                ]
                # Retrieve results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        with file_io_lock_2:
                            lst_stats.append(result)
                            df = pd.DataFrame(lst_stats)

                            success = False
                            attempt = 0
                            max_attempts = 5  # Number of times to try before giving up
                            while not success and attempt < max_attempts:
                                try:
                                    df.to_csv(file_path_output_tmp)
                                    success = True
                                    print("File saved successfully.")
                                except Exception as e:
                                    attempt += 1
                                    print(f"Attempt {attempt} failed with error: {e}")

                            new_excel_path = utils.add_exel_before_csv(file_path_output_tmp)
                            success = False
                            attempt = 0
                            max_attempts = 5  # Number of times to try before giving up
                            while not success and attempt < max_attempts:
                                try:
                                    convert_csv_for_excel(file_path_output_tmp, new_excel_path)
                                    success = True
                                    print("File saved successfully.")
                                except Exception as e:
                                    attempt += 1
                                    print(f"Attempt {attempt} failed with error: {e}")

                    except Exception as exc:
                        print(f"Exception occurred: {exc}")
        else:
            # Process parameters sequentially
            for param in lst_params:
                result = process_param(param, start_date, end_date, low_timeframe, high_timeframe, save_dir, input_data, file_io_lock)
                lst_stats.append(result)

        # Save the collected statistics to a CSV file
        df = pd.DataFrame(lst_stats)
        df.to_csv(file_path_output)

        # Convert CSV for Excel if needed
        new_excel_path = utils.add_exel_before_csv(file_path_output)
        convert_csv_for_excel(file_path_output, new_excel_path)

    else:
        test_1h = False
        test_5m = True
        if test_1h:
            timeframe = '1h'
            ma_type = "TEMA"
            HIGH_OFFSET = float(1.002)
            LOW_OFFSET = float(1)
            ZEMA_LEN_BUY = 30
            ZEMA_LEN_SELL = 90
            SSL_ATR_PERIOD = 10
        elif test_5m:
            timeframe = '5m'
            ma_type = "ALMA"
            HIGH_OFFSET = float(1.004)
            LOW_OFFSET = float(1)
            ZEMA_LEN_BUY = 50
            ZEMA_LEN_SELL = 90
            SSL_ATR_PERIOD = 10

        STOP_LOSS = 0.0
        FEES = 0.0
        # FEES = 0.04

        trend_type = "ICHOMOKU"
        trend_type = "SAR"
        trend_type = "SAR"
        trend_type = "TSI"
        trend_type = "FISHER"
        trend_type = "KALMAN"
        trend_type = "PRICE_ACTION"

        low_timeframe = '1m'
        high_timeframe = '1h'
        save_dir = "test_5m"

        params = {'zema_len_buy': ZEMA_LEN_BUY, 'zema_len_sell': ZEMA_LEN_SELL, 'low_offset': LOW_OFFSET,
                  'high_offset': HIGH_OFFSET,
                  'ssl_atr_period': SSL_ATR_PERIOD,
                  'ichimoku_params': input_data.ichimoku_params
                  }

        backtest = CryptoBacktest("1", 'BTC/USDT', start_date, end_date, timeframe, low_timeframe, high_timeframe, save_dir, ma_type, trend_type, params,
                                  trading_fee=FEES, initial_balance=10000, stop_loss=STOP_LOSS, print_all=True)

        # Fetch data for the default timeframe and store it in self.data
        backtest.fetch_data(data_attr='data')

        if STOP_LOSS != 0 \
                or True:
            # Fetch data for the low timeframe and store it in self.low_tf_data
            backtest.fetch_data(data_attr='low_tf_data')

        # Fetch data for the high timeframe and store it in self.high_tf_data
        backtest.fetch_data(data_attr='high_tf_data')

        backtest.apply_indicator()
        backtest.apply_high_tf_indicator()
        backtest.backtest_strategy()
        print("######################################################")
        backtest.backtest_vbt_strategy()
        backtest.plot_results()
        backtest.plot_vbt_results()


if __name__ == "__main__":
    main()
