import os

# import vectorbt as vbt
from numpy import nan as npNaN
import numpy as np
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import concat_data
import input_data
from data_loader import DataLoaderVBT
from my_trading_simulator import TradingSimulator
from itertools import product
from vectorbtpro import *
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import random

import os
import shutil

def get_unique_filename(base_path, base_name, extension):
    """
    Generate a unique filename by appending _n to the base name if the file exists.

    Parameters:
    - base_path (str): The directory path where the file will be saved.
    - base_name (str): The base name of the file.
    - extension (str): The file extension (e.g., '.csv').

    Returns:
    - str: Unique filename with full path.
    """
    n = 0
    while True:
        unique_filename = f"{base_path}/{base_name}{f'_{n}' if n > 0 else ''}{extension}"
        if not os.path.exists(unique_filename):
            return unique_filename
        n += 1

# === Indicator Functions ===
def ssl_atr__(dataframe, length=7):
    atr = vbt.ATR.run(dataframe['High'], dataframe['Low'], dataframe['Close'], window=14).atr
    sma_high = dataframe['High'].rolling(window=length).mean() + atr
    sma_low = dataframe['Low'].rolling(window=length).mean() - atr

    hlv = np.where(dataframe['Close'] > sma_high, 1, np.where(dataframe['Close'] < sma_low, -1, np.nan))
    hlv = pd.Series(hlv, index=dataframe.index).ffill()

    ssl_down = np.where(hlv < 0, sma_high, sma_low)
    ssl_up = np.where(hlv < 0, sma_low, sma_high)

    return pd.Series(ssl_down, index=dataframe.index), pd.Series(ssl_up, index=dataframe.index)

def ssl_atr(data, period=7):
    high = data.get('High')
    low = data.get('Low')
    close = data.get('Close')

    atr = vbt.ATR.run(high, low, close, window=14).atr
    sma_high = vbt.talib("sma").run(high,timeperiod=period) + atr
    sma_low = vbt.talib("sma").run(low,timeperiod=period) - atr

    # close.vbt.set(1, every=close.vbt > sma_high.vbt, inplace=True)

    hlv = np.where(close.vbt > sma_high.vbt, 1, np.where(close.vbt < sma_low.vbt, -1, np.nan))

    hlv = pd.DataFrame(hlv, index=close.index).ffill()

    ssl_down = np.where(hlv < 0, sma_high, sma_low)
    ssl_up = np.where(hlv < 0, sma_low, sma_high)

    return pd.DataFrame(ssl_down, index=close.index, columns=high.columns), pd.DataFrame(ssl_up, index=close.index, columns=high.columns)

def ichimoku(data, params):
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

    # Compute cloud_top by taking the element-wise maximum
    cloud_top = senkou_span_a.combine(senkou_span_b, np.maximum)
    # Compute cloud_bottom by taking the element-wise minimum
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

# === Strategy Class ===
class IchimokuZemaStrategy:
    def __init__(self, symbols, timeframe, start_date, end_date, trade_type, ma_type, lst_combined,
                 low_offset,
                 high_offset,
                 zema_len_buy,
                 zema_len_sell,
                 ssl_atr_period
                 ):
        self.symbols = symbols
        self.timeframe = timeframe
        self.informative_timeframe = '1h'
        self.trade_type = trade_type
        self.tf = timeframe
        self.ma_type = ma_type
        self.lst_of_my_result = []
        self.lst_combined = lst_combined

        # Fetch data starting from extended_start_date
        dataLoaderVBT = DataLoaderVBT(input_data.extended_start_date, end_date, "vbt_data")
        self.data = dataLoaderVBT.fetch_data(symbols, self.tf).loc[start_date:]
        self.high_tf_data = dataLoaderVBT.fetch_data(symbols, self.informative_timeframe).loc[start_date:]

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
        if input_data.my_trading_sim:
            self.run_my_backtest()
        self.run_backtest()

    # --- Indicator Calculations ---
    def calculate_indicators(self):
        # print(vbt.IF.list_indicators("vbt"))
        # print(vbt.IF.list_indicators("talib"))
        # print(vbt.IF.list_indicators("*ma", location="pandas_ta"))
        # self.data['zema_buy'] = vbt.MA.run(self.data['Close'], window=self.params['zema_len_buy'], ewm=True).ma

        # "ZLEMA", "ZLMA", "TEMA", "DEMA", "ALMA", "KAMA", "HMA"
        zlema_buy_adj = zlema_sell_adj = zlma_buy_adj = zlma_sell_adj = tema_buy_adj = tema_sell_adj = \
            dema_buy_adj = dema_sell_adj = alma_buy_adj = alma_sell_adj = kama_buy_adj = kama_sell_adj = \
            hma_buy_adj = hma_sell_adj = None

        if self.ma_type == "ZLEMA":
            zlema_buy = vbt.talib("EMA").run(self.data.get('Close'), timeperiod=[self.params['zema_len_buy']]).ema
            zlema_buy.columns = zlema_buy.columns.droplevel('ema_timeperiod')
            zlema_buy_adj = zlema_buy * self.params['low_offset']
            zlema_sell = vbt.talib("EMA").run(self.data.get('Close'), timeperiod=self.params['zema_len_sell']).ema
            zlema_sell.columns = zlema_sell.columns.droplevel('ema_timeperiod')
            zlema_sell_adj = zlema_sell * self.params['high_offset']
        elif self.ma_type == "ZLMA":
            zlma_buy = vbt.pandas_ta("ZLMA").run(self.data.get('Close'), zlma_lengh=[self.params['zema_len_buy']], zlma_mode="ema").zlma
            zlma_sell = vbt.pandas_ta("ZLMA").run(self.data.get('Close'), zlma_lengh=self.params['zema_len_sell'], zlma_mode="ema").zlma
            zlma_buy_adj = zlma_buy * self.params['low_offset']
            zlma_sell_adj = zlma_sell * self.params['high_offset']
        elif self.ma_type == "TEMA":
            tema_buy = vbt.pandas_ta("TEMA").run(self.data.get('Close'), timeperiod=[self.params['zema_len_buy']]).tema
            tema_sell = vbt.pandas_ta("TEMA").run(self.data.get('Close'), timeperiod=self.params['zema_len_sell']).tema
            tema_buy_adj = tema_buy * self.params['low_offset']
            tema_sell_adj = tema_sell * self.params['high_offset']
        elif self.ma_type == "DEMA":
            dema_buy = vbt.pandas_ta("DEMA").run(self.data.get('Close'), timeperiod=[self.params['zema_len_buy']]).dema
            dema_sell = vbt.pandas_ta("DEMA").run(self.data.get('Close'), timeperiod=self.params['zema_len_sell']).dema
            dema_buy_adj = dema_buy * self.params['low_offset']
            dema_sell_adj = dema_sell * self.params['high_offset']
        elif self.ma_type == "ALMA":
            alma_buy = vbt.pandas_ta("ALMA").run(self.data.get('Close'), timeperiod=[self.params['zema_len_buy']]).alma
            alma_sell = vbt.pandas_ta("ALMA").run(self.data.get('Close'), timeperiod=self.params['zema_len_sell']).alma
            alma_buy_adj = alma_buy * self.params['low_offset']
            alma_sell_adj = alma_sell * self.params['high_offset']
        elif self.ma_type == "KAMA":
            kama_buy = vbt.pandas_ta("KAMA").run(self.data.get('Close'), timeperiod=[self.params['zema_len_buy']]).kama
            kama_sell = vbt.pandas_ta("KAMA").run(self.data.get('Close'), timeperiod=self.params['zema_len_sell']).kama
            kama_buy_adj = kama_buy * self.params['low_offset']
            kama_sell_adj = kama_sell * self.params['high_offset']
        elif self.ma_type == "HMA":
            hma_buy = vbt.pandas_ta("HMA").run(self.data.get('Close'), timeperiod=[self.params['zema_len_buy']]).hma
            hma_sell = vbt.pandas_ta("HMA").run(self.data.get('Close'), timeperiod=self.params['zema_len_sell']).hma
            hma_buy_adj = hma_buy * self.params['low_offset']
            hma_sell_adj = hma_sell * self.params['high_offset']
        elif self.ma_type == "COMBINED":
            # For each MA type in self.lst_combined, compute the corresponding buy/sell and buy_adj/sell_adj
            for TYPE in self.lst_combined:
                if TYPE == "ZLEMA":
                    zlema_buy = vbt.talib("EMA").run(self.data.get('Close'), timeperiod=[self.params['zema_len_buy']]).ema
                    zlema_buy.columns = zlema_buy.columns.droplevel('ema_timeperiod')
                    zlema_buy_adj = zlema_buy * self.params['low_offset']
                    zlema_sell = vbt.talib("EMA").run(self.data.get('Close'), timeperiod=self.params['zema_len_sell']).ema
                    zlema_sell.columns = zlema_sell.columns.droplevel('ema_timeperiod')
                    zlema_sell_adj = zlema_sell * self.params['high_offset']

                elif TYPE == "ZLMA":
                    zlma_buy = vbt.pandas_ta("ZLMA").run(self.data.get('Close'), zlma_lengh=[self.params['zema_len_buy']], zlma_mode="ema").zlma
                    zlma_sell = vbt.pandas_ta("ZLMA").run(
                        self.data.get('Close'),
                        zlma_lengh=self.params['zema_len_sell'],
                        zlma_mode="ema"
                    ).zlma

                    zlma_buy_adj = zlma_buy * self.params['low_offset']
                    zlma_sell_adj = zlma_sell * self.params['high_offset']

                elif TYPE == "TEMA":
                    tema_buy = vbt.pandas_ta("TEMA").run(self.data.get('Close'),
                                                         timeperiod=[self.params['zema_len_buy']]).tema
                    tema_sell = vbt.pandas_ta("TEMA").run(self.data.get('Close'),
                                                          timeperiod=self.params['zema_len_sell']).tema
                    tema_buy_adj = tema_buy * self.params['low_offset']
                    tema_sell_adj = tema_sell * self.params['high_offset']

                elif TYPE == "DEMA":
                    dema_buy = vbt.pandas_ta("DEMA").run(self.data.get('Close'),
                                                         timeperiod=[self.params['zema_len_buy']]).dema
                    dema_sell = vbt.pandas_ta("DEMA").run(self.data.get('Close'),
                                                          timeperiod=self.params['zema_len_sell']).dema
                    dema_buy_adj = dema_buy * self.params['low_offset']
                    dema_sell_adj = dema_sell * self.params['high_offset']

                elif TYPE == "ALMA":
                    alma_buy = vbt.pandas_ta("ALMA").run(self.data.get('Close'),
                                                         timeperiod=[self.params['zema_len_buy']]).alma
                    alma_sell = vbt.pandas_ta("ALMA").run(self.data.get('Close'),
                                                          timeperiod=self.params['zema_len_sell']).alma
                    alma_buy_adj = alma_buy * self.params['low_offset']
                    alma_sell_adj = alma_sell * self.params['high_offset']

                elif TYPE == "KAMA":
                    kama_buy = vbt.pandas_ta("KAMA").run(self.data.get('Close'),
                                                         timeperiod=[self.params['zema_len_buy']]).kama
                    kama_sell = vbt.pandas_ta("KAMA").run(self.data.get('Close'),
                                                          timeperiod=self.params['zema_len_sell']).kama
                    kama_buy_adj = kama_buy * self.params['low_offset']
                    kama_sell_adj = kama_sell * self.params['high_offset']

                elif TYPE == "HMA":
                    hma_buy = vbt.pandas_ta("HMA").run(self.data.get('Close'),
                                                       timeperiod=[self.params['zema_len_buy']]).hma
                    hma_sell = vbt.pandas_ta("HMA").run(self.data.get('Close'),
                                                        timeperiod=self.params['zema_len_sell']).hma
                    hma_buy_adj = hma_buy * self.params['low_offset']
                    hma_sell_adj = hma_sell * self.params['high_offset']

                # After computing for TYPE, store the results in self.dct_indicators
                # We must ensure these assignments don't overwrite each other for different TYPES.
                # One approach: store each adjusted line in the dictionary keyed by its TYPE.
                # For example:
                # self.dct_indicators[f"{TYPE.lower()}_buy_adj"] = locals().get(f"{TYPE.lower()}_buy_adj", None)
                # self.dct_indicators[f"{TYPE.lower()}_sell_adj"] = locals().get(f"{TYPE.lower()}_sell_adj", None)

        ssl_down, ssl_up = ssl_atr(self.high_tf_data, period=self.params['ssl_atr_period'])
        # self.high_tf_data['ssl_down'] = ssl_down
        # self.high_tf_data['ssl_up'] = ssl_up
        # self.high_tf_data['ssl_ok'] = (ssl_up > ssl_down).astype(int)
        ssl_ok = (ssl_up > ssl_down).astype(int)
        ssl_bear = (ssl_up < ssl_down).astype(int)

        ichimoku_dct = ichimoku(self.high_tf_data, self.params['ichimoku_params'])
        # self.high_tf_data = pd.concat([self.high_tf_data, ichimoku_df], axis=1)

        ichimoku_ok = (
                (ichimoku_dct['tenkan_sen'] > ichimoku_dct['kijun_sen']) &
                (ichimoku_dct['close'] > ichimoku_dct['cloud_top']) &
                (ichimoku_dct['future_green'] > 0) &
                (ichimoku_dct['chikou_span'] > ichimoku_dct['cloud_top'].shift(
                    -self.params['ichimoku_params']['displacement']))
        ).astype(int)

        ichimoku_bear = (
                (ichimoku_dct['tenkan_sen'] < ichimoku_dct['kijun_sen']) &
                (ichimoku_dct['close'] < ichimoku_dct['cloud_bottom']) &
                (ichimoku_dct['future_red'] > 0) &
                (ichimoku_dct['chikou_span'] < ichimoku_dct['cloud_bottom'].shift(
                    -self.params['ichimoku_params']['displacement']))
        ).astype(int)

        self.ichimoku_valid = (~ichimoku_dct['senkou_span_b'].isna()).astype(int).fillna(0)

        self.trend_pulse = (
                (ichimoku_ok > 0) &
                (ssl_ok > 0)
        ).astype(int).fillna(0)

        self.bear_trend_pulse = (
                (ichimoku_bear > 0) &
                (ssl_bear > 0)
        ).astype(int).fillna(0)

        self.ichimoku_valid = self.ichimoku_valid.reindex(self.data.index, method='ffill')
        self.trend_pulse = self.trend_pulse.reindex(self.data.index, method='ffill')
        self.bear_trend_pulse = self.bear_trend_pulse.reindex(self.data.index, method='ffill')

        # "ZLEMA", "ZLMA", "TEMA", "DEMA", "ALMA", "KAMA", "HMA"
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
        # "ZLEMA", "ZLMA", "TEMA", "DEMA", "ALMA", "KAMA", "HMA"
        # A dictionary mapping each ma_type to the corresponding buy and sell adjustment keys
        adj_map = {
            "ZLEMA": ("zlema_buy_adj", "zlema_sell_adj"),
            "ZLMA": ("zlma_buy_adj", "zlma_sell_adj"),
            "TEMA": ("tema_buy_adj", "tema_sell_adj"),
            "DEMA": ("dema_buy_adj", "dema_sell_adj"),
            "ALMA": ("alma_buy_adj", "alma_sell_adj"),
            "KAMA": ("kama_buy_adj", "kama_sell_adj"),
            "HMA": ("hma_buy_adj", "hma_sell_adj"),
        }

        if self.ma_type in adj_map:
            buy_key, sell_key = adj_map[self.ma_type]
            buy_adj = self.dct_indicators[buy_key]
            sell_adj = self.dct_indicators[sell_key]

            buy_signal_long = (
                    (self.dct_indicators['ichimoku_valid'] > 0) &
                    (self.dct_indicators['bear_trend_pulse'] == 0) &
                    # (self.dct_indicators['trend_pulse'] == 1) &
                    (self.data.close < buy_adj)
            ).astype(int)

            sell_signal_long = (
                (self.data.close > sell_adj)
            ).astype(int)

            buy_signal_short = (
                    (self.dct_indicators['ichimoku_valid'] > 0) &
                    (self.dct_indicators['trend_pulse'] == 0) &
                    # (self.dct_indicators['bear_trend_pulse'] == 1) &
                    (self.data.close > buy_adj)
            ).astype(int)

            sell_signal_short = (
                (self.data.close < sell_adj)
            ).astype(int)

        elif self.ma_type == "COMBINED":
            combined_dict = {
                symbol: {
                    "buy_adj": None,
                    "sell_adj": None,
                    "buy_signal_long": None,
                    "sell_signal_long": None,
                    "buy_signal_short": None,
                    "sell_signal_short": None
                }
                for symbol in self.lst_combined
            }

            # Common conditions reused for all symbols
            cond_ichimoku = (self.dct_indicators['ichimoku_valid'] > 0)
            cond_bear_pulse_0 = (self.dct_indicators['bear_trend_pulse'] == 0)
            cond_trend_pulse_0 = (self.dct_indicators['trend_pulse'] == 0)
            price_close = self.data.close

            # Initialize aggregate signals as arrays of zeros
            # Assuming `price_close` is a NumPy array or Pandas Series
            buy_signal_long_agg = (price_close * 0).astype(int)
            sell_signal_long_agg = (price_close * 0).astype(int)
            buy_signal_short_agg = (price_close * 0).astype(int)
            sell_signal_short_agg = (price_close * 0).astype(int)

            for type in self.lst_combined:
                if type in adj_map:
                    buy_key, sell_key = adj_map[type]
                    combined_dict[type]["buy_adj"] = self.dct_indicators[buy_key]
                    combined_dict[type]["sell_adj"] = self.dct_indicators[sell_key]

                    buy_line = combined_dict[type]["buy_adj"]
                    sell_line = combined_dict[type]["sell_adj"]

                    combined_dict[type]["buy_signal_long"] = (
                            cond_ichimoku & cond_bear_pulse_0 & (price_close < buy_line)
                    ).astype(int)

                    combined_dict[type]["sell_signal_long"] = (
                        (price_close > sell_line)
                    ).astype(int)

                    combined_dict[type]["buy_signal_short"] = (
                            cond_ichimoku & cond_trend_pulse_0 & (price_close > buy_line)
                    ).astype(int)

                    combined_dict[type]["sell_signal_short"] = (
                        (price_close < sell_line)
                    ).astype(int)

                    # Aggregate signals with OR operation across all symbols
                    buy_signal_long_agg |= combined_dict[type]["buy_signal_long"]
                    sell_signal_long_agg |= combined_dict[type]["sell_signal_long"]
                    buy_signal_short_agg |= combined_dict[type]["buy_signal_short"]
                    sell_signal_short_agg |= combined_dict[type]["sell_signal_short"]

            # After processing all symbols, assign the aggregated signals
            buy_signal_long = buy_signal_long_agg
            sell_signal_long = sell_signal_long_agg
            buy_signal_short = buy_signal_short_agg
            sell_signal_short = sell_signal_short_agg


        self.dct_signals = {
            "buy_signal_long": buy_signal_long,
            "sell_signal_long": sell_signal_long,
            "buy_signal_short": buy_signal_short,
            "sell_signal_short": sell_signal_short
        }

    def run_my_backtest(self):
        self.lst_of_my_result = []
        for symbol in self.symbols:
            Close = self.data.get("Close")[symbol]
            Entry = self.dct_signals['buy_signal_long'][symbol]
            Exit = self.dct_signals['sell_signal_long'][symbol]
            Entry_short = self.dct_signals['buy_signal_short'][symbol]
            Exit_short = self.dct_signals['sell_signal_short'][symbol]
            simulator = TradingSimulator(symbol, self.timeframe, initial_budget=10000, fee=0.0,  direction=self.trade_type, ma_type=self.ma_type)
            self.lst_of_my_result.append(simulator.simulate_trading(Close, Entry, Exit, Entry_short, Exit_short))

    def run_backtest(self):
        entries_long = self.dct_signals['buy_signal_long'] == 1
        exits_long = self.dct_signals['sell_signal_long'] == 1

        entries_short = self.dct_signals['buy_signal_short'] == 1
        exits_short = self.dct_signals['sell_signal_short'] == 1

        # Stop loss as 10% of the entry price
        stop_loss = 0.20

        # Set frequency dynamically based on timeframe
        frequency_mapping = {
            '1m': '1T',  # 1 minutes
            '5m': '5T',  # 5 minutes
            '15m': '15T',  # 15 minutes
            '30m': '30T',  # 30 minutes
            '1h': '1H',  # 1 hour
            '2h': '2H',  # 2 hours
        }
        freq = frequency_mapping.get(self.timeframe, None)

        if self.trade_type == 'long':
            self.pf = vbt.Portfolio.from_signals(
                close=self.data.get("Close"),
                entries=entries_long,
                exits=exits_long,
                freq=freq,  # Set frequency here
                init_cash=10000,
                size=100,
                size_type="Percent100",
                # fees=0.001,
                # slippage=0.001
            )
        elif self.trade_type == 'short':
            self.pf = vbt.Portfolio.from_signals(
                close=self.data.get("Close"),
                entries=False,  # No long entries
                exits=False,
                short_entries=entries_short,
                short_exits=exits_short,
                freq=freq,  # Set frequency here
                init_cash=10000,
                size=100,
                size_type="Percent100",
                # fees=0.001,
                # slippage=0.001
            )
        else:  # Both long and short
            self.pf = vbt.Portfolio.from_signals(
                close=self.data.get("Close"),
                entries=entries_long,
                exits=exits_long,
                short_entries=entries_short,
                short_exits=exits_short,
                freq=freq,  # Set frequency here
                init_cash=10000,
                size=100,
                size_type="Percent100",
                # fees=0.001,
                # slippage=0.001
            )

    def get_results(self):
        stats_list = []
        for symbol in self.data.columns:
            pf = self.pf[symbol]
            stats = pf.stats()  # Assuming this returns a dictionary or similar object
            # Explicitly copy stats to ensure modifications are applied
            stats = dict(stats)
            # Add additional fields
            stats['Type'] = "vbt_pro"
            stats['Symbol'] = symbol
            stats['Timeframe'] = self.timeframe
            stats['Trade_Type'] = self.trade_type
            stats['MA_Type'] = self.ma_type

            # Add parameters from self.params
            if hasattr(self, 'params') and isinstance(self.params, dict):
                stats.update(self.params)

            stats_list.append(stats)

        return stats_list + self.get_my_results()

    def get_my_results(self):
        if input_data.my_trading_sim:
            return self.lst_of_my_results
        else:
            return []


# all_data = concat_data.concat_data(symbols, lst_trade_type, tf)
# high_tf_data = concat_data.concat_data(symbols, lst_trade_type, tf)

# Assuming you have something like this:
# class IchimokuZemaStrategy:
#     def __init__(self, symbol, timeframe, start_date, end_date, trade_type):
#         # initialize your strategy
#         pass
#
#     def get_results(self, timeframe, trade_type):
#         # return a list of results (dicts or other structures)
#         return [{"symbol": "example_symbol", "timeframe": timeframe, "trade_type": trade_type, "result": 42}]

def run_strategy(symbols, timeframe, start_date, end_date, trade_type, ma_type, lst_combined,
                 low_offset,
                 high_offset,
                 zema_len_buy,
                 zema_len_sell,
                 ssl_atr_period
                 ):
    """
    Helper function that runs the strategy and returns the results.
    """
    print(f"Running strategy for {symbols} on {timeframe} timeframe with trade_type='{trade_type} : {ma_type}'.")
    strategy = IchimokuZemaStrategy(symbols, timeframe, start_date, end_date, trade_type, ma_type, lst_combined,
                                    low_offset,
                                    high_offset,
                                    zema_len_buy,
                                    zema_len_sell,
                                    ssl_atr_period
                                    )
    return strategy.get_results()


def split_list(lst, n):
    """
    Split a list into sublists, each with a maximum length of n.

    Parameters:
        lst (list): The list to split.
        n (int): The maximum length of each sublist.

    Returns:
        list: A list of sublists.
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def save_and_merge_csv(df, directory_path, file_name):
    """
    Save a DataFrame to a CSV file and merge all CSVs with the same name in the directory into one file.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        directory_path (str): The directory path where the CSV will be saved.
        file_name (str): The name of the CSV file (without extension).
    """
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Define the full path for the new CSV file
    new_file_path = os.path.join(directory_path, f"{file_name}.csv")

    # Save the current DataFrame to the CSV file
    df.to_csv(new_file_path, index=False)

    # Gather all CSV files with the same name in the directory
    matching_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.startswith(file_name) and file.endswith(".csv")
    ]

    # Merge all matching CSV files into a single DataFrame
    merged_df = pd.concat(pd.read_csv(file) for file in matching_files)

    # Save the merged DataFrame back to the original file path
    merged_file_path = os.path.join(directory_path, f"{file_name}_merged.csv")
    merged_df.to_csv(merged_file_path, index=False)

    print(f"Data saved to {new_file_path} and merged file created at {merged_file_path}.")

def parallel_execute(data, batch_results_folder_path):
    # Create all tasks using a comprehension and itertools.product
    print("computing param combination list...")
    tasks = [
        (
            data["symbols"],
            timeframe,
            data["start_date"],
            data["end_date"],
            trade_type,
            ma_type,
            data["lst_combined"],
            low_offset,
            high_offset,
            zema_len_buy,
            zema_len_sell,
            ssl_atr_period
        )
        for trade_type, timeframe, ma_type, low_offset, high_offset, zema_len_buy, zema_len_sell, ssl_atr_period in product(
            data["lst_trade_type"],
            data["tf"],
            data["lst_ma_type"],
            data["lst_low_offset"],
            data["lst_high_offset"],
            data["lst_zema_len_buy"],
            data["lst_zema_len_sell"],
            data["lst_ssl_atr_period"]
        )
    ]
    random.shuffle(tasks)

    max_workers = multiprocessing.cpu_count()
    total_tasks = len(tasks)
    lst_tasks = split_list(tasks, 2 * max_workers)
    results_list = []

    print("nb combination: ", len(tasks))

    lst_tasks = [lst_tasks[0], lst_tasks[1], lst_tasks[2], lst_tasks[3]]

    for tasks in lst_tasks:
        batch_results_list = []
        # Execute tasks in parallel
        # with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(run_strategy, *task): task for task in tasks}
            # Collect results as they complete
            for i, future in enumerate(as_completed(futures), start=1):
                results = future.result()
                results_list.extend(results)
                batch_results_list.extend(results)
                print(f"Completed {i}/{total_tasks} tasks")

        print(" Batch processes completed")
        # Identify common keys across all result dictionaries
        batch_common_keys = set.intersection(*(set(d.keys()) for d in batch_results_list))
        # Filter each result dict to include only the common keys
        batch_filtered_list = [{k: d[k] for k in batch_common_keys} for d in batch_results_list]
        # Combine all filtered results into a single DataFrame
        batch_stats_df = pd.DataFrame(batch_filtered_list)

        batch_stats_df = batch_stats_df[input_data.desired_columns]
        batch_stats_df.columns = [col.upper() for col in batch_stats_df.columns]

        batch_stats_df["COMPARE RETURN"] = batch_stats_df["Total Return [%]"] > batch_stats_df["Benchmark Return [%]"]

        save_and_merge_csv(batch_stats_df, batch_results_folder_path, "batch_stats_df")

    print(" All processes completed")
    # Identify common keys across all result dictionaries
    common_keys = set.intersection(*(set(d.keys()) for d in results_list))

    # Filter each result dict to include only the common keys
    filtered_list = [{k: d[k] for k in common_keys} for d in results_list]

    # Combine all filtered results into a single DataFrame
    all_stats_df = pd.DataFrame(filtered_list)

    all_stats_df = all_stats_df[input_data.desired_columns]
    all_stats_df.columns = [col.upper() for col in all_stats_df.columns]

    all_stats_df["COMPARE RETURN"] = all_stats_df["Total Return [%]"] > all_stats_df["Benchmark Return [%]"]

    return all_stats_df

if __name__ == "__main__":
    results = {}
    stats_list = []  # To collect stats for all symbols and timeframes

    # data = fetch_data(symbols, tf[0], start_date, end_date)
    # Ensure the results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Check if the folder exists
    if os.path.exists(input_data.batch_results_folder_path):
        # Clear the folder by deleting its contents
        shutil.rmtree(input_data.batch_results_folder_path)
    # Recreate the folder
    os.makedirs(input_data.batch_results_folder_path)

    if input_data.multi_treading:
        input_datas = {
            "symbols": input_data.symbols,
            "tf": input_data.tf,
            "start_date": input_data.start_date,
            "end_date": input_data.end_date,
            "lst_trade_type": input_data.lst_trade_type,
            "lst_ma_type": input_data.lst_ma_type,
            "lst_combined": input_data.lst_combined,
            "lst_low_offset": input_data.lst_low_offset,
            "lst_high_offset": input_data.lst_high_offset,
            "lst_zema_len_buy": input_data.lst_zema_len_buy,
            "lst_zema_len_sell": input_data.lst_zema_len_sell,
            "lst_ssl_atr_period": input_data.lst_ssl_atr_period
            }

        all_stats_df = parallel_execute(input_datas, input_data.batch_results_folder_path)

    else:
        lst_stats = []
        lst_my_stats = []
        for trade_type in input_data.lst_trade_type:
            for timeframe in input_data.tf:
                for ma_type in input_data.lst_ma_type:
                    for low_offset in input_data.lst_low_offset:
                        for high_offset in input_data.lst_high_offset:
                            for zema_len_buy in input_data.lst_zema_len_buy:
                                for zema_len_sell in input_data.lst_zema_len_sell:
                                    for ssl_atr_period in input_data.lst_ssl_atr_period:
                                        strategy = IchimokuZemaStrategy(input_data.symbols, timeframe, input_data.start_date, input_data.end_date, trade_type, ma_type, input_data.lst_combined,
                                                                        low_offset,
                                                                        high_offset,
                                                                        zema_len_buy,
                                                                        zema_len_sell,
                                                                        ssl_atr_period
                                                                        )
                                        print(f"Completed strategy for {input_data.symbols} on {timeframe} timeframe with trade_type='{trade_type}'.")
                                        lst_stats.extend(strategy.get_results(timeframe, trade_type, ma_type))

            # Combine all stats into a single DataFrame
            all_stats_df = pd.DataFrame(lst_stats)

    if False:
        column_order = [
            'Symbol', 'Timeframe', 'Trade_Type',
            'End Value', 'Total Return [%]', 'Benchmark Return [%]',
            'Max Drawdown [%]',
            'Total Orders', 'Total Fees Paid', 'Total Trades', 'Win Rate [%]',
            'Sharpe Ratio', 'Calmar Ratio', 'Omega Ratio', 'Sortino Ratio',
            'Profit Factor', 'Expectancy',
            'Min Value', 'Max Value',  'Position Coverage [%]',
            'Max Gross Exposure [%]',  'Max Drawdown Duration',
            'Best Trade [%]', 'Worst Trade [%]', 'Avg Winning Trade [%]',
            'Avg Losing Trade [%]', 'Avg Winning Trade Duration',
            'Avg Losing Trade Duration',
            'Start Index', 'End Index', 'Total Duration', 'Start Value'
            ]

        # ['Total Closed Trades', 'Total Open Trades', 'Open Trade PnL', 'Start', 'End', 'Period']

        # Reorder columns
        all_stats_df = all_stats_df[column_order]

    folder_path = "results_vbtpro"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Get a unique filename
    stats_csv_path = get_unique_filename(folder_path, "portfolio_stats_summary", ".csv")

    # all_stats_df = all_stats_df.iloc[:, ::-1]
    """
    all_stats_df = all_stats_df[[
        "Symbol",
        "Type",
        "Trade_Type",
        "MA_Type",
        "Timeframe",
        "End Value",
        "Total Return [%]",
        "Benchmark Return [%]",
        "Max Drawdown [%]",
        "Win Rate [%]",
        "Sharpe Ratio",
        "Total Trades"
    ]]
    """

    # Save the combined stats to a CSV file
    all_stats_df.to_csv(stats_csv_path, index=False)

    print(f"Portfolio statistics saved to {stats_csv_path}.")


