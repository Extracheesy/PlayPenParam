import asyncio
import platform

import os
from vectorbtpro import *
import hashlib
import numpy as np
import pandas as pd

class DataLoaderVBT:
    def __init__(self, start_date, end_date, path):
        self.start_date = start_date
        self.end_date = end_date
        self.path = path

    # === Data Fetching and Storage ===
    def fetch_data(self, symbols, tf):
        # Convert the list of symbols into a string separated by '-'

        lst_symbols_binance = vbt.BinanceData.list_symbols("*")

        # kas_symbols = [symbol for symbol in lst_symbols_binance if symbol.startswith('KAS')]

        # Filter symbols to include only those present in lst_symbols_binance
        filtered_symbols = [symbol for symbol in symbols if symbol in lst_symbols_binance]

        # Identify symbols that have been removed
        # removed_symbols = [symbol for symbol in symbols if symbol not in lst_symbols_binance]
        symbols = filtered_symbols
        # print("Filtered symbols:", symbols)
        # print("Removed symbols:", removed_symbols)

        symbols_str = '-'.join(symbols)
        timeframe_str = '-'.join(tf)


        # Remove any unwanted characters from start_date and end_date
        start_date_str = self.start_date.replace('-', '')
        end_date_str = self.end_date.replace('-', '')

        # Ensure the directory exists
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Generate the filename
        filename = os.path.join(self.path, f"{symbols_str}_{timeframe_str}_{start_date_str}_{end_date_str}.h5")

        # Create a hash of the symbols for uniqueness
        symbols_hash = hashlib.md5(symbols_str.encode()).hexdigest()

        # Generate the filename with the hash
        filename = os.path.join(self.path, f"{symbols_hash}_{timeframe_str}_{start_date_str}_{end_date_str}.h5")

        if os.path.exists(filename):
            # data = vbt.HDFData.pull(filename)
            data = vbt.BinanceData.from_hdf(filename)
        else:
            data = vbt.BinanceData.pull(
                symbols,
                start=self.start_date,
                end=self.end_date,
                timeframe=tf
            )

            if not os.path.exists('data_pro'):
                os.makedirs('data_pro')

            data.to_hdf(filename)

        return data