import os
import sys
import csv
import shutil
import hashlib
import random
import threading
import multiprocessing
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import talib

import vectorbtpro as vbt

import input_data
from data_loader import DataLoaderVBT
from my_trading_simulator import TradingSimulator

from convert_to_xcel import convert_csv_for_excel

from pathlib import Path

# ============================
# Utility Functions
# ============================
def ensure_id_first_column(df):
    if "id" not in df.columns:
        # Insert 'id' as the first column, counting from 0 to len(df)-1
        df.insert(0, "id", range(len(df)))
    return df

def add_exel_before_csv(path_str):
    p = Path(path_str)            # Convert string to Path object
    if p.suffix.lower() == ".csv":
        # If the file ends with .csv, insert _exel before .csv
        return str(p.with_name(f"{p.stem}_exel{p.suffix}"))
    else:
        # If it doesn't end with .csv, return the original path (or handle differently)
        return path_str

def get_unique_filename(base_path, base_name, extension):
    """
    Generate a unique filename by appending _n to the base name if the file exists.
    """
    n = 0
    while True:
        unique_filename = f"{base_path}/{base_name}{f'_{n}' if n > 0 else ''}{extension}"
        if not os.path.exists(unique_filename):
            return unique_filename
        n += 1


def detect_delimiter(file_path):
    """
    Detects the delimiter of a CSV file, defaulting to ',' if detection fails.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(1024)  # Read a small part of the file
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except csv.Error:
            print("Warning: Could not determine delimiter; defaulting to manual check.")
            return None


def read_csv_thread_safe(file_path, lock):
    """
    Reads a CSV file safely with automatic delimiter detection.
    Uses a lock to ensure thread-safe reading.
    """
    delimiter = detect_delimiter(file_path)
    with lock:
        if delimiter:
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            # Try `;` first (common for Excel), fallback to `,`
            try:
                df = pd.read_csv(file_path, delimiter=";")
            except pd.errors.ParserError:
                df = pd.read_csv(file_path, delimiter=",")
    return df


def safe_convert(value, target_type):
    """
    Converts 'value' to 'target_type' (float or int) only if it's a string.
    Returns the original value if conversion fails or is not needed.
    """
    if isinstance(value, str):
        try:
            return target_type(value)
        except ValueError:
            print(f"Warning: Could not convert {value} to {target_type.__name__}")
            return value
    return value


def is_column_string(df, col):
    """
    Checks if a DataFrame column contains string (object) values.
    """
    return df[col].dtype == object


def split_list(lst, n):
    """
    Split a list into sublists, each with a maximum length of n.
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def save_and_merge_csv(df, directory_path, file_name):
    """
    Save a DataFrame to a CSV file and then merge all CSVs with the same base name
    in the directory into one file.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    new_file_path = os.path.join(directory_path, f"{file_name}.csv")
    df.to_csv(new_file_path, index=False)

    # Gather all CSV files with the same base name in the directory
    matching_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.startswith(file_name) and file.endswith(".csv")
    ]
    merged_df = pd.concat(pd.read_csv(file) for file in matching_files)

    merged_file_path = os.path.join(directory_path, f"{file_name}_merged.csv")
    merged_df.to_csv(merged_file_path, index=False)

    print(f"Data saved to {new_file_path} and merged file created at {merged_file_path}.")


# ============================
# Indicator Functions
# ============================

def ssl_atr(data, period=7):
    """
    Computes the SSL-ATR lines (SSL Down & SSL Up) based on ATR and SMA.

    Returns:
        ssl_down_series (pd.Series)
        ssl_up_series   (pd.Series)
    """
    try:
        high = data.get('High')
        low = data.get('Low')
        close = data.get('Close')

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


def ichimoku(data, params):
    """
    Computes Ichimoku indicator components (Tenkan-sen, Kijun-sen, Senkou spans,
    Chikou span, and color-coded cloud info).
    """
    high = data.get('High')
    low = data.get('Low')
    close = data.get('Close')

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


# ============================
# Strategy Class
# ============================

class IchimokuZemaStrategy:
    def __init__(
        self,
        id,
        symbols,
        timeframe,
        start_date,
        end_date,
        trade_type,
        ma_type,
        lst_combined,
        low_offset,
        high_offset,
        zema_len_buy,
        zema_len_sell,
        ssl_atr_period
    ):
        self.id = id
        self.symbols = symbols
        self.timeframe = timeframe
        self.informative_timeframe = '1h'
        self.trade_type = trade_type
        self.tf = timeframe
        self.ma_type = ma_type
        self.lst_of_my_result = []
        self.lst_combined = lst_combined

        # Fetch data
        data_loader = DataLoaderVBT(input_data.extended_start_date, end_date, "vbt_data")
        self.data = data_loader.fetch_data([symbols], self.tf).loc[start_date:]
        self.high_tf_data = data_loader.fetch_data([symbols], self.informative_timeframe).loc[start_date:]

        self.params = {
            'low_offset': low_offset,
            'high_offset': high_offset,
            'zema_len_buy': zema_len_buy,
            'zema_len_sell': zema_len_sell,
            'ssl_atr_period': ssl_atr_period,
            'ichimoku_params': input_data.ichimoku_params
        }

        self.calculate_indicators()
        self.generate_signals()

        # Optionally run custom simulator
        if input_data.my_trading_sim:
            self.run_my_backtest()

        # Run vectorbt backtest
        self.run_backtest()

    # --- Indicator Calculations ---
    def calculate_indicators(self):
        """
        Calculate and store all relevant indicators (ZLEMA, ZLMA, TEMA, DEMA, ALMA, KAMA, HMA, Ichimoku, SSL ATR).
        """
        zlema_buy_adj = zlema_sell_adj = None
        zlma_buy_adj = zlma_sell_adj = None
        tema_buy_adj = tema_sell_adj = None
        dema_buy_adj = dema_sell_adj = None
        alma_buy_adj = alma_sell_adj = None
        kama_buy_adj = kama_sell_adj = None
        hma_buy_adj = hma_sell_adj = None

        # Select close data
        close_data = self.data.get('Close')
        if isinstance(close_data, pd.DataFrame) and self.symbols in close_data.columns:
            close_data = close_data[self.symbols]

        # Calculate the chosen MA type (or COMBINED)
        try:
            if self.ma_type == "ZLEMA":
                # Using TA-Lib EMA as a proxy for "ZLEMA" from original code
                zlema_buy = talib.EMA(close_data, timeperiod=self.params['zema_len_buy'])
                zlema_buy_adj = zlema_buy * self.params['low_offset']

                zlema_sell = talib.EMA(close_data, timeperiod=self.params['zema_len_sell'])
                zlema_sell_adj = zlema_sell * self.params['high_offset']

            elif self.ma_type == "ZLMA":
                zlma_buy = vbt.pandas_ta("ZLMA").run(
                    close_data,
                    zlma_lengh=[self.params['zema_len_buy']],
                    zlma_mode="ema"
                ).zlma
                zlma_sell = vbt.pandas_ta("ZLMA").run(
                    close_data,
                    zlma_lengh=self.params['zema_len_sell'],
                    zlma_mode="ema"
                ).zlma
                zlma_buy_adj = zlma_buy * self.params['low_offset']
                zlma_sell_adj = zlma_sell * self.params['high_offset']

            elif self.ma_type == "TEMA":
                tema_buy = vbt.pandas_ta("TEMA").run(close_data, timeperiod=[self.params['zema_len_buy']]).tema
                tema_sell = vbt.pandas_ta("TEMA").run(close_data, timeperiod=self.params['zema_len_sell']).tema
                tema_buy_adj = tema_buy * self.params['low_offset']
                tema_sell_adj = tema_sell * self.params['high_offset']

            elif self.ma_type == "DEMA":
                dema_buy = vbt.pandas_ta("DEMA").run(close_data, timeperiod=[self.params['zema_len_buy']]).dema
                dema_sell = vbt.pandas_ta("DEMA").run(close_data, timeperiod=self.params['zema_len_sell']).dema
                dema_buy_adj = dema_buy * self.params['low_offset']
                dema_sell_adj = dema_sell * self.params['high_offset']

            elif self.ma_type == "ALMA":
                alma_buy = vbt.pandas_ta("ALMA").run(close_data, timeperiod=[self.params['zema_len_buy']]).alma
                alma_sell = vbt.pandas_ta("ALMA").run(close_data, timeperiod=self.params['zema_len_sell']).alma
                alma_buy_adj = alma_buy * self.params['low_offset']
                alma_sell_adj = alma_sell * self.params['high_offset']

            elif self.ma_type == "KAMA":
                kama_buy = vbt.pandas_ta("KAMA").run(close_data, timeperiod=[self.params['zema_len_buy']]).kama
                kama_sell = vbt.pandas_ta("KAMA").run(close_data, timeperiod=self.params['zema_len_sell']).kama
                kama_buy_adj = kama_buy * self.params['low_offset']
                kama_sell_adj = kama_sell * self.params['high_offset']

            elif self.ma_type == "HMA":
                hma_buy = vbt.pandas_ta("HMA").run(close_data, timeperiod=[self.params['zema_len_buy']]).hma
                hma_sell = vbt.pandas_ta("HMA").run(close_data, timeperiod=self.params['zema_len_sell']).hma
                hma_buy_adj = hma_buy * self.params['low_offset']
                hma_sell_adj = hma_sell * self.params['high_offset']

            elif self.ma_type == "COMBINED":
                # Handled separately in the signal generation phase
                pass

        except:
            exit(111)

        # SSL ATR Calculation on high_tf_data
        try:
            ssl_down, ssl_up = ssl_atr(self.high_tf_data, period=self.params['ssl_atr_period'])
        except:
            exit(222)

        ssl_ok = (ssl_up > ssl_down).astype(int)
        ssl_bear = (ssl_up < ssl_down).astype(int)

        # Ichimoku on high_tf_data
        try:
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

        except:
            exit(333)

        # Store computed indicators
        self.dct_indicators = {
            "zlema_buy_adj": zlema_buy_adj,
            "zlema_sell_adj": zlema_sell_adj,
            "zlma_buy_adj": zlma_buy_adj,
            "zlma_sell_adj": zlma_sell_adj,
            "tema_buy_adj": tema_buy_adj,
            "tema_sell_adj": tema_sell_adj,
            "dema_buy_adj": dema_buy_adj,
            "dema_sell_adj": dema_sell_adj,
            "alma_buy_adj": alma_buy_adj,
            "alma_sell_adj": alma_sell_adj,
            "kama_buy_adj": kama_buy_adj,
            "kama_sell_adj": kama_sell_adj,
            "hma_buy_adj": hma_buy_adj,
            "hma_sell_adj": hma_sell_adj,
            "ichimoku_valid": self.ichimoku_valid,
            "trend_pulse": self.trend_pulse,
            "bear_trend_pulse": self.bear_trend_pulse
        }

    # --- Signal Generation ---
    def generate_signals(self):
        """
        Generate buy/sell signals for long/short entries based on
        the chosen MA line and Ichimoku + SSL ATR conditions.
        """
        adj_map = {
            "ZLEMA": ("zlema_buy_adj", "zlema_sell_adj"),
            "ZLMA": ("zlma_buy_adj", "zlma_sell_adj"),
            "TEMA": ("tema_buy_adj", "tema_sell_adj"),
            "DEMA": ("dema_buy_adj", "dema_sell_adj"),
            "ALMA": ("alma_buy_adj", "alma_sell_adj"),
            "KAMA": ("kama_buy_adj", "kama_sell_adj"),
            "HMA": ("hma_buy_adj", "hma_sell_adj"),
        }

        if isinstance(self.data.close, pd.DataFrame):
            data_close = self.data.close.iloc[:, 0]
        else:
            data_close = self.data.close

        buy_signal_long = sell_signal_long = None
        buy_signal_short = sell_signal_short = None

        # If user selects a single MA type
        if self.ma_type in adj_map:
            try:
                buy_key, sell_key = adj_map[self.ma_type]
                buy_adj = self.dct_indicators[buy_key]
                sell_adj = self.dct_indicators[sell_key]

                buy_adj_aligned = buy_adj.reindex(self.data.close.index).ffill()
                sell_adj_aligned = sell_adj.reindex(self.data.close.index).ffill()

                if isinstance(buy_adj_aligned, pd.DataFrame):
                    buy_adj_aligned = buy_adj_aligned.iloc[:, 0]
                if isinstance(sell_adj_aligned, pd.DataFrame):
                    sell_adj_aligned = sell_adj_aligned.iloc[:, 0]

                # Generate signals for long
                buy_signal_long = (
                    (self.dct_indicators['ichimoku_valid'] > 0) &
                    (self.dct_indicators['bear_trend_pulse'] == 0) &
                    (data_close < buy_adj_aligned)
                ).astype(int)
                sell_signal_long = (data_close > sell_adj_aligned).astype(int)

                # Generate signals for short
                buy_signal_short = (
                    (self.dct_indicators['ichimoku_valid'] > 0) &
                    (self.dct_indicators['trend_pulse'] == 0) &
                    (data_close > buy_adj_aligned)
                ).astype(int)
                sell_signal_short = (data_close < sell_adj_aligned).astype(int)

            except:
                exit(444)

        # If user selects "COMBINED" (multiple MAs)
        elif self.ma_type == "COMBINED":
            cond_ichimoku = (self.dct_indicators['ichimoku_valid'] > 0)
            cond_bear_pulse_0 = (self.dct_indicators['bear_trend_pulse'] == 0)
            cond_trend_pulse_0 = (self.dct_indicators['trend_pulse'] == 0)

            buy_signal_long_agg = (data_close * 0).astype(int)
            sell_signal_long_agg = (data_close * 0).astype(int)
            buy_signal_short_agg = (data_close * 0).astype(int)
            sell_signal_short_agg = (data_close * 0).astype(int)

            for ma_type in self.lst_combined:
                if ma_type in adj_map:
                    buy_key, sell_key = adj_map[ma_type]
                    buy_line = self.dct_indicators[buy_key]
                    sell_line = self.dct_indicators[sell_key]

                    buy_line = buy_line.reindex(self.data.close.index).ffill()
                    sell_line = sell_line.reindex(self.data.close.index).ffill()
                    if isinstance(buy_line, pd.DataFrame):
                        buy_line = buy_line.iloc[:, 0]
                    if isinstance(sell_line, pd.DataFrame):
                        sell_line = sell_line.iloc[:, 0]

                    tmp_buy_long = (cond_ichimoku & cond_bear_pulse_0 & (data_close < buy_line)).astype(int)
                    tmp_sell_long = (data_close > sell_line).astype(int)

                    tmp_buy_short = (cond_ichimoku & cond_trend_pulse_0 & (data_close > buy_line)).astype(int)
                    tmp_sell_short = (data_close < sell_line).astype(int)

                    buy_signal_long_agg |= tmp_buy_long
                    sell_signal_long_agg |= tmp_sell_long
                    buy_signal_short_agg |= tmp_buy_short
                    sell_signal_short_agg |= tmp_sell_short

            buy_signal_long = buy_signal_long_agg
            sell_signal_long = sell_signal_long_agg
            buy_signal_short = buy_signal_short_agg
            sell_signal_short = sell_signal_short_agg

        # Store signals
        self.dct_signals = {
            "buy_signal_long": buy_signal_long,
            "sell_signal_long": sell_signal_long,
            "buy_signal_short": buy_signal_short,
            "sell_signal_short": sell_signal_short
        }

    def run_my_backtest(self):
        """
        Runs a custom backtest using 'TradingSimulator' from 'my_trading_simulator.py'.
        """
        self.lst_of_my_result = []
        for symbol in self.symbols:
            Close = self.data.get("Close")[symbol]
            Entry = self.dct_signals['buy_signal_long'][symbol]
            Exit = self.dct_signals['sell_signal_long'][symbol]
            Entry_short = self.dct_signals['buy_signal_short'][symbol]
            Exit_short = self.dct_signals['sell_signal_short'][symbol]

            simulator = TradingSimulator(
                symbol=symbol,
                timeframe=self.timeframe,
                initial_budget=10000,
                fee=0.0,
                direction=self.trade_type,
                ma_type=self.ma_type
            )
            results = simulator.simulate_trading(Close, Entry, Exit, Entry_short, Exit_short)
            self.lst_of_my_result.append(results)

    def run_backtest(self):
        """
        Runs a vectorbt backtest (Portfolio.from_signals) with the given signals.
        """
        entries_long = self.dct_signals['buy_signal_long'] == 1
        exits_long = self.dct_signals['sell_signal_long'] == 1
        entries_short = self.dct_signals['buy_signal_short'] == 1
        exits_short = self.dct_signals['sell_signal_short'] == 1

        # Map timeframe to a pandas frequency
        frequency_mapping = {
            '1m': '1T',   # 1 minute
            '5m': '5T',   # 5 minutes
            '15m': '15T', # 15 minutes
            '30m': '30T', # 30 minutes
            '1h': '1H',   # 1 hour
            '2h': '2H',   # 2 hours
        }
        freq = frequency_mapping.get(self.timeframe, None)

        # Setup the portfolio
        if self.trade_type == 'long':
            self.pf = vbt.Portfolio.from_signals(
                close=self.data.get("Close"),
                entries=entries_long,
                exits=exits_long,
                freq=freq,
                init_cash=10000,
                size=100,
                size_type="Percent100"
            )
        elif self.trade_type == 'short':
            self.pf = vbt.Portfolio.from_signals(
                close=self.data.get("Close"),
                entries=False,
                exits=False,
                short_entries=entries_short,
                short_exits=exits_short,
                freq=freq,
                init_cash=10000,
                size=100,
                size_type="Percent100"
            )
        else:  # both long and short
            self.pf = vbt.Portfolio.from_signals(
                close=self.data.get("Close"),
                entries=entries_long,
                exits=exits_long,
                short_entries=entries_short,
                short_exits=exits_short,
                freq=freq,
                init_cash=10000,
                size=100,
                size_type="Percent100"
            )

    def get_results(self):
        """
        Returns a list of stats dictionaries (one per symbol).
        Includes the vectorbt stats and any custom stats from self.lst_of_my_result.
        """
        stats_list = []
        for symbol in self.data.columns:
            pf = self.pf
            stats = dict(pf.stats())  # vbt-pro stats

            # Additional fields
            stats['Type'] = "vbt_pro"
            stats["id"] = self.id
            stats['Symbol'] = symbol
            stats['Timeframe'] = self.timeframe
            stats['Trade_Type'] = self.trade_type
            stats['MA_Type'] = self.ma_type

            # Add parameters
            if hasattr(self, 'params') and isinstance(self.params, dict):
                stats.update(self.params)

            stats_list.append(stats)

        return stats_list + self.get_my_results()

    def get_my_results(self):
        """
        Returns custom simulator results if 'my_trading_sim' is enabled in input_data.
        """
        if input_data.my_trading_sim:
            return self.lst_of_my_result
        else:
            return []


# ============================
# Parallel Execution
# ============================

def run_strategy(
    id,
    symbols,
    timeframe,
    start_date,
    end_date,
    trade_type,
    ma_type,
    lst_combined,
    low_offset,
    high_offset,
    zema_len_buy,
    zema_len_sell,
    ssl_atr_period
):
    """
    Helper function that initializes and runs the IchimokuZemaStrategy,
    returning its results.
    """
    print(f"Running strategy for {symbols} on {timeframe} "
          f"timeframe with trade_type='{trade_type}' and MA='{ma_type}'.")
    strategy = IchimokuZemaStrategy(
        id=id,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        trade_type=trade_type,
        ma_type=ma_type,
        lst_combined=lst_combined,
        low_offset=low_offset,
        high_offset=high_offset,
        zema_len_buy=zema_len_buy,
        zema_len_sell=zema_len_sell,
        ssl_atr_period=ssl_atr_period
    )
    return strategy.get_results()


def parallel_execute(scored_results_path, data, batch_results_folder_path):
    """
    Executes the strategy in parallel based on configurations
    read from the CSV file at 'scored_results_path'.
    Results are saved incrementally.
    """
    print("Computing parameter combination list...")

    if not os.path.exists(scored_results_path):
        print(f"Error: {scored_results_path} does not exist.")
        sys.exit(1)  # Exit with an error code

    lock = threading.Lock()
    if False:
        df = read_csv_thread_safe(scored_results_path, lock)
    else:
        df = pd.read_csv(scored_results_path)

    df = ensure_id_first_column(df)

    # Convert columns where necessary
    cols_to_float = ["LOW_OFFSET", "HIGH_OFFSET"]
    cols_to_int = ["ZEMA_LEN_BUY", "ZEMA_LEN_SELL", "SSL_ATR_PERIOD"]

    # Float columns
    for col in cols_to_float:
        if is_column_string(df, col):
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Integer columns
    for col in cols_to_int:
        if is_column_string(df, col):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Example filter (can be modified or removed)
    # filtered_df = df[df['TYPE'] == 'max']
    filtered_df = df.copy()

    # Build tasks
    rows_as_dicts = filtered_df.to_dict(orient='records')
    tasks = []
    for row in rows_as_dicts:
        task = (
            row["id"],
            row["SYMBOL"],
            row["TIMEFRAME"],
            input_data.start_date,
            input_data.end_date,
            "long",  # or short, or from row if needed
            row["MA_TYPE"],
            [],
            safe_convert(row["LOW_OFFSET"], float),
            safe_convert(row["HIGH_OFFSET"], float),
            safe_convert(row["ZEMA_LEN_BUY"], int),
            safe_convert(row["ZEMA_LEN_SELL"], int),
            safe_convert(row["SSL_ATR_PERIOD"], int)
        )
        tasks.append(task)

    max_workers = multiprocessing.cpu_count()
    total_tasks = len(tasks)
    grouped_tasks = split_list(tasks, 2 * max_workers)

    results_list = []
    print("Number of combinations:", total_tasks)
    cpt = 0

    for chunk in grouped_tasks:
        batch_results_list = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(run_strategy, *task): task for task in chunk}

            for i, future in enumerate(as_completed(futures), start=1):
                results = future.result()
                results_list.extend(results)
                batch_results_list.extend(results)
                print(f"Completed {i}/{total_tasks} tasks - index:{cpt}")
                cpt += 1

        # After each chunk, save the intermediate batch
        print("Batch processes completed.")
        batch_common_keys = set.intersection(*(set(d.keys()) for d in batch_results_list))
        batch_filtered_list = [{k: d[k] for k in batch_common_keys} for d in batch_results_list]
        batch_stats_df = pd.DataFrame(batch_filtered_list)

        batch_stats_df = batch_stats_df[input_data.desired_columns]
        batch_stats_df.columns = [col.upper() for col in batch_stats_df.columns]
        batch_stats_df["COMPARE RETURN"] = (
            batch_stats_df["TOTAL RETURN [%]"] > batch_stats_df["BENCHMARK RETURN [%]"]
        )
        save_and_merge_csv(batch_stats_df, batch_results_folder_path, "batch_stats_df")

    print("All processes completed.")

    # Final aggregation
    common_keys = set.intersection(*(set(d.keys()) for d in results_list))
    filtered_list = [{k: d[k] for k in common_keys} for d in results_list]
    all_stats_df = pd.DataFrame(filtered_list)

    all_stats_df = all_stats_df[input_data.desired_columns]
    all_stats_df.columns = [col.upper() for col in all_stats_df.columns]
    all_stats_df["COMPARE RETURN"] = (
        all_stats_df["TOTAL RETURN [%]"] > all_stats_df["BENCHMARK RETURN [%]"]
    )
    return all_stats_df


# ============================
# Main Script Execution
# ============================
if __name__ == "__main__":

    if False:
        # Example path definition now in 'main' instead of 'parallel_execute'
        input_directory = './test_param'
        folder_path = input_directory
        scored_results_path = os.path.join(input_directory, 'test_param_combo.csv')
    else:
        input_directory = './new_dataset'
        folder_path = input_directory
        scored_results_path = os.path.join(input_directory, 'merged_output.csv')

        # scored_results_path = os.path.join(input_directory, 'merged_output - Copie.csv')

    # Optional: Clear the results folder if needed
    # if os.path.exists(input_data.batch_results_folder_path):
    #     shutil.rmtree(input_data.batch_results_folder_path)
    # os.makedirs(input_data.batch_results_folder_path, exist_ok=True)

    if input_data.multi_treading:
        # Pass scored_results_path into parallel_execute
        all_stats_df = parallel_execute(scored_results_path, {}, input_data.batch_results_folder_path)
    else:
        # Example single-threaded approach
        lst_stats = []
        for trade_type in input_data.lst_trade_type:
            for timeframe in input_data.tf:
                for ma_type in input_data.lst_ma_type:
                    for low_offset in input_data.lst_low_offset:
                        for high_offset in input_data.lst_high_offset:
                            for zema_len_buy in input_data.lst_zema_len_buy:
                                for zema_len_sell in input_data.lst_zema_len_sell:
                                    for ssl_atr_period in input_data.lst_ssl_atr_period:
                                        id = 0
                                        strategy = IchimokuZemaStrategy(
                                            id=id,
                                            symbols=input_data.symbols,
                                            timeframe=timeframe,
                                            start_date=input_data.start_date,
                                            end_date=input_data.end_date,
                                            trade_type=trade_type,
                                            ma_type=ma_type,
                                            lst_combined=input_data.lst_combined,
                                            low_offset=low_offset,
                                            high_offset=high_offset,
                                            zema_len_buy=zema_len_buy,
                                            zema_len_sell=zema_len_sell,
                                            ssl_atr_period=ssl_atr_period
                                        )
                                        print(f"Completed strategy for {input_data.symbols} "
                                              f"on {timeframe} timeframe with trade_type='{trade_type}'.")
                                        lst_stats.extend(strategy.get_results())

        all_stats_df = pd.DataFrame(lst_stats)


    os.makedirs(folder_path, exist_ok=True)
    stats_csv_path = get_unique_filename(folder_path, "portfolio_stats_summary", ".csv")
    all_stats_df.to_csv(stats_csv_path, index=False)
    print(f"Portfolio statistics saved to {stats_csv_path}.")

    new_excel_path = add_exel_before_csv(stats_csv_path)
    convert_csv_for_excel(stats_csv_path, new_excel_path)


