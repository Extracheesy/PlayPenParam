import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

from pandas_ta.overlap.zlma import zlma

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

from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from pandas.testing import assert_series_equal

from ohlcv import get_candle_dataframe

def get_3_month_intervals(start_date_str):
    # Convert input string to datetime
    current = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
    end = datetime.utcnow().replace(tzinfo=timezone.utc)
    # end = "2025-04-02"
    # end = datetime(2025, 4, 2, tzinfo=timezone.utc)

    dates = []
    while current <= end:
        start_date_str = current.strftime("%Y-%m-%d") + 'T00:00:00Z'
        dates.append(start_date_str)
        current += relativedelta(months=3)

    return dates

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
                 trading_fee=0.001, initial_balance=10000, stop_loss=0.02, vbt_plot=False, print_all=False, bitget_data=False):
        self.id = id
        self.symbol = symbol
        self.start_date = start_date
        # self.end_date = datetime.utcnow().replace(tzinfo=timezone.utc)
        # self.end_date = end_date
        self.end_date = utils.round_time(end_date, timeframe)
        self.str_end_date = self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
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

        self.VBT_PLOT = vbt_plot

        self.bitget_data = bitget_data
        if self.bitget_data:
            self.str_bitget_data = "_BITGET"
        else:
            self.str_bitget_data = ""

        self.save_trades = False
        self.print_all = print_all
        self.show_plot = self.print_all

        self.stats ={}
        self.stats["ID"] = self.id
        self.stats['SYMBOL'] = self.symbol
        self.stats['START_DATE'] = self.start_date
        self.stats['END_DATE'] = self.str_end_date
        self.stats['TIMEFRAME'] = self.timeframe
        self.stats['MA_TYPE'] = self.ma_type
        self.stats['TREND_TYPE'] = self.trend_type
        self.stats['LOW_TIMEFRAME'] = self.low_timeframe
        self.stats['HIGH_TIMEFRAME'] = self.high_timeframe
        self.stats['STOP_LOSS'] = self.stop_loss
        self.stats['FEES'] = self.trading_fee
        self.stats['BITGET_DATA'] = str(self.bitget_data)

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

    def fetch_data_biget(self, file_lock):
        if isinstance(self.end_date, str):
            dt_end_date = datetime.strptime(self.end_date, "%Y-%m-%dT%H:%M:%SZ")
        else:
            dt_end_date = self.end_date  # Assume it's already a datetime object

        # Round the end date to the nearest timeframe and format it
        self.str_end_date = self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        self.start_date = self.end_date - relativedelta(months=1)

        # Convert start_date to a formatted string.
        # Change the format string as needed (here it's set to YYYY-MM-DD).
        self.str_start_date = self.start_date.strftime("%Y-%m-%d")

        self.timeframe = "1m"

        filename = f"bitget_1m_{self.symbol}_{self.str_start_date}_{self.str_end_date.split('T')[0]}.csv"
        path_filename = os.path.join(self.data_dir, filename)
        with file_lock:
            if os.path.exists(path_filename):
                self.data = pd.read_csv(path_filename)
                self.data['datetime'] = pd.to_datetime(self.data['datetime'],
                                                       format='%Y-%m-%d %H:%M:%S',
                                                       errors='raise')
                self.data.set_index('datetime', drop=True, inplace=True)
                self.low_tf_data = self.data.copy()
                print("read low_tf_data: ", filename)
            else:
                self.data = get_candle_dataframe(self.symbol, "1m")
                self.low_tf_data = self.data.copy()

                try:
                    self.low_tf_data.to_csv(path_filename)
                except Exception as e:
                    print("toto")

            filename = f"bitget_1H_{self.symbol}_{self.str_start_date}_{self.str_end_date.split('T')[0]}.csv"
            path_filename = os.path.join(self.data_dir, filename)
            if os.path.exists(path_filename):
                self.high_tf_data = pd.read_csv(path_filename)
                self.high_tf_data['datetime'] = pd.to_datetime(self.high_tf_data['datetime'],
                                                               format='%Y-%m-%d %H:%M:%S',
                                                               errors='raise')
                self.high_tf_data.set_index('datetime', drop=True, inplace=True)
                print("read high_tf_data: ", filename)
            else:
                self.high_tf_data = get_candle_dataframe(self.symbol, "1H")

                try:
                    self.high_tf_data.to_csv(path_filename)
                except Exception as e:
                    print("toto")

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
        # if self.end_date is None:
        if False:
            dt_end_date = datetime.utcnow()
        else:
            if isinstance(self.end_date, str):
                dt_end_date = datetime.strptime(self.end_date, "%Y-%m-%dT%H:%M:%SZ")
            else:
                dt_end_date = self.end_date  # Assume it's already a datetime object

            # Round the end date to the nearest timeframe and format it
            self.str_end_date = self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

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
        end_timestamp = exchange.parse8601(self.str_end_date) if self.str_end_date is not None else None

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

        ######################################################################################
        if timeframe == '1h':  # CEDE MODIF !!!!
            df = df.shift(1)
        ######################################################################################

        # Filter the DataFrame based on end_date (if provided)
        if self.str_end_date is not None:
            df = df[df.index <= pd.to_datetime(self.str_end_date)]

        if reverse:
            backup_index = df.index.copy()
            df = df[::-1].reset_index(drop=False)
            df.index = backup_index
            df = df.drop('timestamp', axis=1)
            df.rename(columns={'open': 'close', 'close': 'open'}, inplace=True)

        # Assign the DataFrame to the specified attribute
        setattr(self, data_attr, df)

        try:
            # Save the fetched data to a file
            print("Saving fetched data to file...")
            getattr(self, data_attr).to_csv(data_file)
        except:
            print("toto")

    def apply_indicator(self):
        if self.data is None:
            raise ValueError("No data available. Fetch data first.")

        try:
            close_data = self.data['close']

            if self.ma_type == "ZLEMA":
                ma_buy = talib.EMA(close_data, timeperiod=self.params['zema_len_buy']) * self.params['low_offset']
                ma_sell = talib.EMA(close_data, timeperiod=self.params['zema_len_sell']) * self.params['high_offset']

            elif self.ma_type in ["HMA", "HMA_1", "JMA", "ZLMA_HMA", "ZLMA_ZELMA", "ZLMA", "TEMA", "DEMA", "ALMA", "KAMA"]:
                if self.ma_type == "ZLMA_HMA":
                    # zlma_buy = ta.zlma(close_data, timeperiod=self.params['zema_len_buy'], mamode="ema")
                    # zlma_sell = ta.zlma(close_data, timeperiod=self.params['zema_len_sell'], mamode="ema")

                    zlma_buy = zlma(close_data, mamode="hma")

                    zlma_buy = ta.zlma(close_data, mamode="hma")
                    # zlma_sell = ta.zlma(close_data, mamode="hma")
                    zlma_sell = zlma_buy.copy()
                    ma_buy = zlma_buy * self.params['low_offset']
                    ma_sell = zlma_sell * self.params['high_offset']

                elif self.ma_type == "ZLMA_ZELMA":
                    zlma_buy = ta.zlma(close_data, mamode="zlema")
                    # zlma_sell = ta.zlma(close_data, mamode="zlema")
                    zlma_sell = zlma_buy.copy()
                    ma_buy = zlma_buy * self.params['low_offset']
                    ma_sell = zlma_sell * self.params['high_offset']

                elif self.ma_type == "ZLMA_HMA":
                    zlma_buy = ta.zlma(close_data, mamode="hma")
                    # zlma_sell = ta.zlma(close_data, mamode="hma")
                    zlma_sell = zlma_buy.copy()
                    ma_buy = zlma_buy * self.params['low_offset']
                    ma_sell = zlma_sell * self.params['high_offset']

                elif self.ma_type == "ZLMA":
                    zlma_buy = ta.zlma(close_data)
                    # zlma_sell = ta.zlma(close_data)
                    zlma_sell = zlma_buy.copy()
                    ma_buy = zlma_buy * self.params['low_offset']
                    ma_sell = zlma_sell * self.params['high_offset']

                elif self.ma_type == "JMA":
                    jma_buy = ta.jma(close_data)
                    # jma_sell = ta.jma(close_data)
                    jma_sell = jma_buy.copy()
                    ma_buy = jma_buy * self.params['low_offset']
                    ma_sell = jma_sell * self.params['high_offset']

                elif self.ma_type == "TEMA":
                    tema_buy = ta.tema(close_data, timeperiod=self.params['zema_len_buy'])
                    tema_sell = ta.tema(close_data, timeperiod=self.params['zema_len_sell'])
                    ma_buy = tema_buy * self.params['low_offset']
                    ma_sell = tema_sell * self.params['high_offset']

                elif self.ma_type == "DEMA":
                    # dema_buy = ta.dema(close_data, timeperiod=self.params['zema_len_buy'])
                    # dema_sell = ta.dema(close_data, timeperiod=self.params['zema_len_sell'])
                    dema_buy = ta.dema(close_data)
                    dema_sell = ta.dema(close_data)
                    ma_buy = dema_buy * self.params['low_offset']
                    ma_sell = dema_sell * self.params['high_offset']

                elif self.ma_type == "ALMA":
                    # alma_buy = ta.alma(close_data, timeperiod=self.params['zema_len_buy'])
                    # alma_sell = ta.alma(close_data, timeperiod=self.params['zema_len_sell'])
                    alma_buy = ta.alma(close_data)
                    alma_sell = ta.alma(close_data)

                    ma_buy = alma_buy * self.params['low_offset']
                    ma_sell = alma_sell * self.params['high_offset']

                elif self.ma_type == "KAMA":
                    # kama_buy = ta.kama(close_data, timeperiod=self.params['zema_len_buy'])
                    # kama_sell = ta.kama(close_data, timeperiod=self.params['zema_len_sell'])
                    kama_buy = ta.kama(close_data)
                    kama_sell = ta.kama(close_data)
                    ma_buy = kama_buy * self.params['low_offset']
                    ma_sell = kama_sell * self.params['high_offset']

                elif self.ma_type == "HMA_1":
                    hma_buy = ta.hma(close_data, timeperiod=self.params['zema_len_buy'])     # CEDE TESTED OK
                    hma_sell = ta.hma(close_data, timeperiod=self.params['zema_len_sell'])
                    ma_buy = hma_buy * self.params['low_offset']
                    ma_sell = hma_sell * self.params['high_offset']

                elif self.ma_type == "HMA":
                    # hma_buy = ta.hma(close_data, timeperiod=self.params['zema_len_buy'])     # CEDE TESTED OK
                    # hma_sell = ta.hma(close_data, timeperiod=self.params['zema_len_sell'])
                    hma_buy = ta.hma(close_data)
                    # hma_sell = ta.hma(close_data)
                    hma_sell = hma_buy.copy()
                    ma_buy = hma_buy * self.params['low_offset']
                    ma_sell = hma_sell * self.params['high_offset']

                    if False:
                        hma_buy_length = ta.hma(close_data, length=self.params['zema_len_buy'])

                        df = pd.DataFrame({
                            'hma_buy': hma_buy,
                            'hma_buy_length': hma_buy_length
                        })

                        # Save the DataFrame to a CSV file (without the index)
                        df.to_csv("hma_comparison.csv", index=False)
                        exit(0)
                        hma_buy = hma_buy_length

                        try:
                            assert_series_equal(hma_buy, hma_buy_length)
                            print("Both methods produce identical outputs!")
                        except AssertionError as e:
                            print("There is a difference between the outputs:", e)

                elif self.ma_type == "KAMA":
                    # kama_buy = ta.kama(close_data, timeperiod=self.params['zema_len_buy'])
                    # kama_sell = ta.kama(close_data, timeperiod=self.params['zema_len_sell'])
                    kama_buy = ta.kama(close_data)
                    kama_sell = ta.kama(close_data)
                    ma_buy = kama_buy * self.params['low_offset']
                    ma_sell = kama_sell * self.params['high_offset']

                else:
                    raise ValueError(f"Unknown moving average type: {self.ma_type}")
        except Exception as e:
            print('toto')

        try:
            self.data['buy_adj'] = ma_buy
            self.data['sell_adj'] = ma_sell
        except Exception as e:
            print('toto')


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

                self.high_tf_data["buy_signal"] = buy_signal
                self.high_tf_data["sell_signal"] = sell_signal

                # Create trend array based on price action
                trend_array = np.zeros(len(self.high_tf_data), dtype=int)
                trend_array[buy_signal] = 1  # Uptrend when close breaks highest_window
                trend_array[sell_signal] = -1  # Downtrend when close breaks lowest_window

            elif self.trend_type == 'MY_PRICE_ACTION':
                # Compute highest and lowest window over 10 periods
                self.high_tf_data["highest_window"] = self.high_tf_data["high"].rolling(window=10).max()
                self.high_tf_data["lowest_window"] = self.high_tf_data["low"].rolling(window=10).min()

                # Identify buy and sell signals
                # buy_signal = self.high_tf_data["close"] > self.high_tf_data["highest_window"].shift(1)
                # sell_signal = self.high_tf_data["close"] < self.high_tf_data["lowest_window"].shift(1)

                self.high_tf_data["highest_window"] = self.high_tf_data["highest_window"].shift(1)
                self.high_tf_data["lowest_window"] = self.high_tf_data["lowest_window"].shift(1)
                self.my_high_tf_data = self.high_tf_data.copy()

                self.my_high_tf_data = self.my_high_tf_data.reindex(self.data.index).ffill()
                self.my_high_tf_data["close"] = self.data["close"]
                self.my_high_tf_data["buy_signal"] = self.my_high_tf_data["close"] > self.my_high_tf_data["highest_window"]
                self.my_high_tf_data["sell_signal"] = self.my_high_tf_data["close"] < self.my_high_tf_data["lowest_window"]
                self.my_high_tf_data["TREND"] = 0
                # start with all zeros
                self.my_high_tf_data["TREND"] = 0

                # set +1 where buy_signal is True
                self.my_high_tf_data.loc[self.my_high_tf_data["buy_signal"] == True, "TREND"] = 1

                # set -1 where sell_signal is True
                self.my_high_tf_data.loc[self.my_high_tf_data["sell_signal"] == True, "TREND"] = -1

                self.data["TREND"] = self.my_high_tf_data["TREND"].copy()

                return

            self.close_high_tf = self.high_tf_data["close"].reindex(self.data.index, method='ffill')
            self.data["close_high_tf"] = self.close_high_tf

            self.highest_window = self.high_tf_data["highest_window"].reindex(self.data.index, method='ffill')
            self.data["highest_window"] = self.highest_window

            self.lowest_window = self.high_tf_data["lowest_window"].reindex(self.data.index, method='ffill')
            self.data["lowest_window"] = self.lowest_window

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

        if True:
            self.save_debug_data()

    def save_debug_data(self):
        """
        Saves self.data as a CSV in <save_dir>/debug_data/,
        tagging it with either _BITGET or _BINANCE.
        Returns the full path to the saved file.
        """
        # pick suffix
        str_source = self.str_bitget_data if self.str_bitget_data == '_BITGET' else '_BINANCE'

        # ensure debug folder exists
        self.save_dir_debug = os.path.join(self.save_dir, 'debug_data')
        os.makedirs(self.save_dir_debug, exist_ok=True)

        # build filename and save
        filename = f"{self.id}{str_source}_{self.symbol}.csv"
        filepath = os.path.join(self.save_dir_debug, filename)
        self.data.to_csv(filepath)

        filename = f"{self.id}{str_source}_{self.symbol}_high_tf.csv"
        filepath = os.path.join(self.save_dir_debug, filename)
        self.high_tf_data.to_csv(filepath)

        return

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
                                 f'{self.id}{self.str_bitget_data}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_pnl.png')
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
                                     f'{self.id}{self.str_bitget_data}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_returns.png')
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
                                                 f'{self.id}{self.str_bitget_data}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_signals.png')

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

    def plot_weekly_results(self):
        import matplotlib.pyplot as plt

        with plot_io_lock:
            df = self.data.copy()
            # For weekly returns:
            weekly = df.resample('W').last()
            # Calculate weekly returns safely
            weekly_returns = weekly[['cumulative_strategy_returns', 'cumulative_buy_hold']].pct_change()
            # Replace any infinite values (division by zero issues) with NaN
            weekly_returns.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Create the weekly returns plot and save it as a file
            fig_week, ax_week = plt.subplots(figsize=(14, 7))
            weekly_returns.plot(kind='bar', ax=ax_week)
            ax_week.set_title('Weekly Returns')
            # Hide axis
            ax_week.set_axis_off()
            plt.tight_layout()

            save_path = os.path.join(self.save_dir,
                                     f'{self.id}{self.str_bitget_data}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_weekly_returns.png')
            fig_week.savefig(save_path)  # Save the weekly returns chart to a file
            plt.close(fig_week)

            df = self.data.copy()
            # For monthly returns:
            monthly = df.resample('M').last()
            # Calculate monthly returns safely
            monthly_returns = monthly[['cumulative_strategy_returns', 'cumulative_buy_hold']].pct_change()
            # Replace any infinite values with NaN
            monthly_returns.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Create the monthly returns plot and save it as a file
            fig_month, ax_month = plt.subplots(figsize=(14, 7))
            monthly_returns.plot(kind='bar', ax=ax_month)
            ax_month.set_title('Monthly Returns')
            ax_month.set_axis_off()
            plt.tight_layout()

            save_path = os.path.join(self.save_dir,
                                     f'{self.id}{self.str_bitget_data}_{self.symbol.replace("/", "_")}_{self.timeframe}_{self.ma_type}_{self.trend_type}_{self.stop_loss}_{self.trading_fee}_monthly_returns.png')
            fig_month.savefig(save_path)  # Save the monthly returns chart to a file
            plt.close(fig_month)

            del df

def set_df_with_missing_data(df, lst_trend):
    # Define your lists
    lst_STOP_LOSS = [0, 2]
    lst_FEES = [0, 0.04]
    lst_TREND_TYPE = lst_trend

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
def process_param(param, start_date, end_date, low_timeframe, high_timeframe, save_dir, input_data, file_io_lock, vbt_plot):
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
    BITGET_DATA = (str(param["BITGET_DATA"]).lower() == "true")

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
        stop_loss=STOP_LOSS,
        vbt_plot=vbt_plot,
        bitget_data=BITGET_DATA
    )

    try:
        if BITGET_DATA:
            backtest.fetch_data_biget(file_io_lock)
        else:
            REVERSE_MODE = False
            with file_io_lock:
                # Fetch and process data
                backtest.fetch_data(data_attr='data', reverse=REVERSE_MODE)
                # The condition below always evaluates to True because of "or True",
                # so adjust it if necessary.
                if STOP_LOSS != 0 or True:
                    backtest.fetch_data(data_attr='low_tf_data', reverse=REVERSE_MODE)
                backtest.fetch_data(data_attr='high_tf_data', reverse=REVERSE_MODE)
    except:
        print("toto")

    try:
        # Apply indicators and perform backtests
        try:
            backtest.apply_indicator()
        except Exception as e:
            backtest.apply_indicator()

        backtest.apply_high_tf_indicator()
        backtest.backtest_strategy()
        backtest.backtest_vbt_strategy()
    except:
        print("backtest error")

    try:
        # Retrieve and return metrics
        dct_stats = backtest.get_metrics()
    except:
        print("dct error")

    try:
        # Plot results
        if (
                float(dct_stats['M_CUMULATIVE_RETURN'].strip('%')) > 0
                and float(dct_stats['M_CUMULATIVE_RETURN'].strip('%')) > float(dct_stats['M_BUY_HOLD_RETURN'].strip('%'))
        ):
            backtest.plot_weekly_results()
            backtest.plot_results()
            # backtest.plot_pnl()
            backtest.plot_vbt_results()
    except:
        print("Plot error")

    del backtest
    print("######################################################")
    return dct_stats

def main():

    lst_start_date = [
        '2025-04-06T00:00:00Z'
    ]

    lst_start_date = [
        '2024-01-01T00:00:00Z'
    ]

    lst_param_strategy = []

    lst_start_date = [
        '2025-03-01T00:00:00Z'
    ]

    lst_start_date = [
        '2025-03-01T00:00:00Z'
    ]

    lst_start_date = [
        '2024-01-01T00:00:00Z',
        '2025-03-01T00:00:00Z',
        '2025-02-01T00:00:00Z',
        '2025-01-01T00:00:00Z',
        '2024-12-17T00:00:00Z',
        '2025-01-20T00:00:00Z',
        '2024-03-13T00:00:00Z'
    ]

    lst_start_date = [
        '2025-03-13T00:00:00Z'
    ]

    lst_start_date = [
        '2024-01-01T00:00:00Z',
        '2025-03-01T00:00:00Z',
        '2024-12-17T00:00:00Z',
        '2024-03-13T00:00:00Z'
    ]

    lst_start_date = [
        '2024-01-01T00:00:00Z',
        '2024-03-13T00:00:00Z'
    ]

    START_AT_ONE_MONTH = False
    if START_AT_ONE_MONTH:
        # 1. get UTC “now”
        now_utc = datetime.utcnow()

        # 2. subtract one month
        one_month_ago = now_utc - relativedelta(months=1)

        # 3. if you want it at midnight (00:00:00), zero out the time fields
        if False:
            one_month_ago_midnight = one_month_ago.replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        # 4. format it exactly as '2025-03-13T00:00:00Z'
        iso_str = one_month_ago.strftime('%Y-%m-%dT%H:%M:%SZ')

        print(iso_str)  # e.g. '2025-03-17T00:00:00Z' if today is 2025‑04‑17

        lst_start_date = [
            iso_str
        ]

    NOW = True
    if not NOW:
        end = "2025-04-08"
        end_date = datetime(2025, 4, 8, tzinfo=timezone.utc)
        str_end_date = datetime(2025, 4, 8, tzinfo=timezone.utc).strftime("%Y-%m-%d")
    else:
        now_date = datetime.now(timezone.utc)
        str_now_date = now_date.strftime("%Y-%m-%d")

        end_date = now_date
        str_end_date = str_now_date

    for start_date in lst_start_date:
        str_start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00")).strftime("%Y-%m-%d")

        print("start date: ", start_date)
        print("end date: ", end_date)
        low_timeframe = '1m'
        high_timeframe = '1h'

        working_directory = "./test_multi_trend_selection_test_7"

        input_file_csv = "input_data_excel.csv"
        input_file_csv_2 = "input_data_full.csv"
        output_file_csv = "output_data_full_test.csv"
        output_file_csv_tmp = "output_data_full_test_tmp.csv"
        save_dir = "result_test"

        result_directory = r"\result_test" + f"_{str_start_date}_{str_end_date}"
        directory_path = working_directory + result_directory

        param_strategy = {
            "start_date": start_date,
            "end_date" : end_date,
            "low_timeframe": low_timeframe,
            "high_timeframe": high_timeframe,
            "working_directory" : working_directory,
            "input_file_csv": input_file_csv,
            "input_file_csv_2": input_file_csv_2,
            "output_file_csv": output_file_csv,
            "output_file_csv_tmp": output_file_csv_tmp,
            "save_dir": directory_path,
            "vbt_plot": True
        }
        lst_param_strategy.append(param_strategy)

    for param in lst_param_strategy:
        run_strategy(param)

def run_strategy(param_strategy):
    start_date = param_strategy.get("start_date")
    end_date = param_strategy.get("end_date")
    low_timeframe = param_strategy.get("low_timeframe")
    high_timeframe = param_strategy.get("high_timeframe")
    working_directory = param_strategy.get("working_directory")
    input_file_csv = param_strategy.get("input_file_csv")
    input_file_csv_2 = param_strategy.get("input_file_csv_2")
    output_file_csv = param_strategy.get("output_file_csv")
    output_file_csv_tmp = param_strategy.get("output_file_csv_tmp")
    save_dir = param_strategy.get("save_dir")
    directory_path = save_dir
    vbt_plot = param_strategy.get("vbt_plot")

    # Join the directory and file name to form the full path
    file_path = os.path.join(working_directory, input_file_csv)
    file_path_2 = os.path.join(working_directory, input_file_csv_2)
    # save_dir = os.path.join(working_directory, save_dir)
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
        # df_input_data = set_df_with_missing_data(df_input_data, lst_trend)
        # df_input_data = df_input_data.sample(frac=1, random_state=42).reset_index(drop=True)
        df_input_data.to_csv(file_path_2)

        new_excel_path = utils.add_exel_before_csv(file_path_2)
        convert_csv_for_excel(file_path_2, new_excel_path)
    else:
        df_input_data = utils.read_csv_thread_safe(file_path, lock)
        lst_trend = ["SAR", "TSI", "FISHER", "KALMAN", "PRICE_ACTION"]
        # lst_trend = ["PRICE_ACTION"]
        df_input_data = set_df_with_missing_data(df_input_data, lst_trend)
        df_input_data = df_input_data.sample(frac=1, random_state=42).reset_index(drop=True)
        df_input_data = utils.drop_zero_fees(df_input_data)

        FILTER_ID = False
        FILTER_SYMBOL = True
        FILTER_INDICATOR = False
        FILTER_BITGET = True
        if FILTER_ID:
            lst_ids = [65, 69, 73, 75, 77]
            lst_ids = [49, 129]
            lst_ids = [49, 51]
            df_input_data = utils.keep_lst_ids(df_input_data, lst_ids)
        if FILTER_SYMBOL:
            lst_ids = ["SOLUSDT", "BTCUSDT"]
            df_input_data = utils.keep_lst_symbols(df_input_data, lst_ids)
        if FILTER_INDICATOR:
            lst_ids = ["HMA"]
            df_input_data = utils.keep_lst_indicators(df_input_data, lst_ids)
        if FILTER_BITGET:
            KEEP_BITGET = False
            df_input_data = utils.keep_lst_indicators_bitget(df_input_data, KEEP_BITGET)

        # df_input_data["SYMBOL"].iloc[0] = "BTCUSDT"

        # df_input_data["TREND_TYPE"] = "MY_PRICE_ACTION"
        try:
            df_input_data.to_csv(file_path_2)
        except:
            print("toto")

        new_excel_path = utils.add_exel_before_csv(file_path_2)
        convert_csv_for_excel(file_path_2, new_excel_path)

    if False:
        df_input_data = df_input_data.head(1)  # For test
        # df_input_data = df_input_data[df_input_data['ID'] == 2]

    if True: # Filter
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

    # Set the maximum number of threads to run concurrently
    max_threads = 4

    lst_stats = []  # This will store the metrics from each backtest

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
                file_io_lock,
                vbt_plot
            )
            for param in lst_params  # lst_params should be defined with your parameters
        ]
        # Retrieve results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"Exception occurred: {exc}")
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

    # Save the collected statistics to a CSV file
    df = pd.DataFrame(lst_stats)
    df.to_csv(file_path_output)

    # Convert CSV for Excel if needed
    new_excel_path = utils.add_exel_before_csv(file_path_output)
    convert_csv_for_excel(file_path_output, new_excel_path)



if __name__ == "__main__":
    main()
