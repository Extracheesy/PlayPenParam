import pandas as pd
import numpy as np

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


class CryptoBacktest:
    def __init__(self, symbol, start_date, end_date=None, lst_timeframe=["1m", "5m", "15m", "30m", "1h"], save_dir='results'):
        self.symbol = symbol
        self.end_date = end_date
        self.start_date = start_date

        self.data_dir = save_dir

        self.lst_timeframe = lst_timeframe

        for timeframe in self.lst_timeframe:
            self.fetch_data(data_attr='lst_of_tf', timeframe=timeframe, reverse=False)

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

    def fetch_data(self, data_attr, timeframe="1m", reverse=False):
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
        elif data_attr == 'lst_of_tf':
            timeframe = timeframe
        else:
            data_attr = 'lst_of_tf'
            timeframe = timeframe

        # Determine the end time
        # if self.end_date is None:
        if self.end_date is None:
            dt_end_date = datetime.utcnow()
            self.end_date = dt_end_date
            self.str_end_date = self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
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
        # if timeframe == '1h':  # CEDE MODIF !!!!
        #    df = df.shift(1)
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