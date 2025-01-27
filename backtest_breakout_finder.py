import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
import os
from datetime import datetime


class CryptoBacktest:
    def __init__(self, symbol, start_date, timeframe='1h', save_dir='results'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.timeframe = timeframe
        self.save_dir = save_dir
        self.data = None

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def fetch_data(self):
        exchange = ccxt.binance()
        since = exchange.parse8601(self.start_date)
        ohlcv = exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[df.index <= pd.to_datetime(self.end_date)]
        self.data = df

    def apply_indicator(self):
        if self.data is None:
            raise ValueError("No data available. Fetch data first.")
        self.data['SMA_20'] = self.data['close'].rolling(window=20).mean()

    def backtest_strategy(self):
        if self.data is None or 'SMA_20' not in self.data:
            raise ValueError("Apply indicator before backtesting.")
        self.data['signal'] = np.where(self.data['close'] > self.data['SMA_20'], 1, -1)
        self.data['buy_signal'] = np.where((self.data['signal'] == 1) & (self.data['signal'].shift(1) == -1),
                                           self.data['close'], np.nan)
        self.data['sell_signal'] = np.where((self.data['signal'] == -1) & (self.data['signal'].shift(1) == 1),
                                            self.data['close'], np.nan)

    def plot_results(self):
        if self.data is None:
            raise ValueError("No data to plot.")

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['close'], label='Close Price', color='blue', alpha=0.5)
        plt.plot(self.data.index, self.data['SMA_20'], label='SMA 20', color='red', linestyle='dashed')
        plt.scatter(self.data.index, self.data['buy_signal'], label='Buy Signal', marker='^', color='green', alpha=1)
        plt.scatter(self.data.index, self.data['sell_signal'], label='Sell Signal', marker='v', color='red', alpha=1)

        plt.title(f'{self.symbol} Backtest with SMA Indicator')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()

        save_path = os.path.join(self.save_dir, f'{self.symbol.replace("/", "_")}_backtest.png')
        plt.savefig(save_path)
        plt.show()
        print(f"Graph saved at {save_path}")


def main():
    symbol = 'BTC/USDT'
    start_date = '2023-01-01T00:00:00Z'
    timeframe = input("Enter timeframe (e.g., '1m', '5m', '1h', '1d'): ")
    save_dir = input("Enter directory to save the graph: ")

    backtest = CryptoBacktest(symbol, start_date, timeframe, save_dir)
    backtest.fetch_data()
    backtest.apply_indicator()
    backtest.backtest_strategy()
    backtest.plot_results()


if __name__ == "__main__":
    main()
