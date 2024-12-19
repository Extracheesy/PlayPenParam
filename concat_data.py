import os
# import vectorbt as vbt
from vectorbtpro import *
import numpy as np
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class concat_data:
    def __init__(self, symbols, trade_types, timeframes):
        self.concat_data = {}
        for type in trade_types:
            self.concat_data[type] = {}
            for tf in timeframes:
                self.concat_data[type][tf] = {}
                for symbol in symbols:
                    df = pd.DataFrame()
                    self.concat_data[type][tf][symbol] = df

    def add_columns(self, symbols, trade_type, timeframe, data, column_name):
        for symbol in symbols:
            df = self.concat_data[trade_type][timeframe][symbol]

            if isinstance(data.get(column_name), pd.DataFrame):
                df[column_name.lower()] = data.get(column_name)[symbol]
            elif isinstance(data, pd.DataFrame):
                if column_name in data.columns:
                    df[column_name.lower()] = data.get(column_name)
                else:
                    df[column_name.lower()] = data.get(symbol)
            elif isinstance(data, pd.Series):
                df[column_name.lower()] = data

    def save_all_data(self, prefix=''):
        # Ensure the directory exists where you want to save the files
        os.makedirs('output', exist_ok=True)

        # Loop through the nested dictionary
        for trade_type, timeframes in self.concat_data.items():
            for timeframe, symbols in timeframes.items():
                for symbol, df in symbols.items():
                    # Construct the filename with the prefix
                    filename = f"{prefix}_{trade_type}_{timeframe}_{symbol}.csv"
                    filepath = os.path.join('output', filename)
                    df = df.rename_axis('date')
                    # Save the DataFrame to CSV
                    df.to_csv(filepath)
