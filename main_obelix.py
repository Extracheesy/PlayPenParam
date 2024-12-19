import os
import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Adjustable Parameters ===
timeframes = ['5m', '15m', '30m', '1h', '2h']
timeframes = ['5m', '15m']
symbols = ['PEPEUSDT', 'BTCUSDT', 'SOLUSDT', 'ETHUSDT']

symbol_meme = [
    "DOGEUSDT",
    "SHIBUSDT",
    "PEPEUSDT",
    "WIFUSDT",
    "BONKUSDT",
    "FLOKIUSDT",
    "BOMEUSDT",
    # "MOGUSDT"
    # "POPCATUSDT"
]

symbols_binance = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT', 'SHIBUSDT', 'TRXUSDT', 'AVAXUSDT', 'UNIUSDT', 'ATOMUSDT', 'LINKUSDT', 'ETCUSDT', 'XMRUSDT', 'BCHUSDT', 'APTUSDT', 'FILUSDT', 'LDOUSDT', 'ARBUSDT', 'QNTUSDT', 'NEARUSDT', 'VETUSDT', 'ALGOUSDT', 'ICPUSDT', 'GRTUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'EGLDUSDT', 'AAVEUSDT', 'AXSUSDT', 'THETAUSDT', 'XTZUSDT', 'IMXUSDT', 'RUNEUSDT', 'KLAYUSDT', 'ZILUSDT', 'CHZUSDT', 'GALAUSDT', 'ENJUSDT', 'STXUSDT', 'KAVAUSDT', 'SNXUSDT', 'CRVUSDT', '1INCHUSDT', 'BATUSDT']

symbols = list(set(symbol_meme + symbols + symbols_binance))
lst_trade_type = ['long', 'short', 'both']

# lst_trade_type = ['long']
symbols = ["PEPEUSDT"]
timeframes = ['1m', "5m", "15m"]

print("list symbols: ", symbols)

low_offset = 0.964
high_offset = 1.004
zema_len_buy = 51
zema_len_sell = 72
ssl_atr_period = 10

ichimoku_params = {
    'conversion_line_period': 20,
    'base_line_periods': 60,
    'lagging_span': 120,
    'displacement': 30
}

start_date = '2024-03-01'
# start_date = '2024-01-01'
end_date = '2024-11-07'

# Maximum lookback period based on the longest indicator
max_lookback_days = max(
    ichimoku_params['conversion_line_period'],
    ichimoku_params['base_line_periods'],
    ichimoku_params['lagging_span'],
    ssl_atr_period,
    zema_len_buy,
    zema_len_sell
)

extended_start_date = (pd.Timestamp(start_date) - timedelta(days=max_lookback_days)).strftime('%Y-%m-%d')


# === Data Fetching and Storage ===
def fetch_data(symbol, timeframe, start_date, end_date):
    filename = f"data/{symbol}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"

    if os.path.exists(filename):
        dataframe = pd.read_csv(filename, parse_dates=True)
        if 'timestamp' in dataframe.columns:
            dataframe.set_index('timestamp', inplace=True)
        else:
            dataframe.index = pd.to_datetime(dataframe.index)
            dataframe.index.name = 'timestamp'
    else:
        data = vbt.CCXTData.download(
            symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            show_progress=True
        )
        dataframe = data.get()

        if not os.path.exists('data'):
            os.makedirs('data')

        dataframe.to_csv(filename, index_label='timestamp')

        #####################################################
        dataframe = pd.read_csv(filename, parse_dates=True)
        if 'timestamp' in dataframe.columns:
            dataframe.set_index('timestamp', inplace=True)
        else:
            dataframe.index = pd.to_datetime(dataframe.index)
            dataframe.index.name = 'timestamp'
        #####################################################

    return dataframe


# === Indicator Functions ===
def ssl_atr(dataframe, length=7):
    atr = vbt.ATR.run(dataframe['High'], dataframe['Low'], dataframe['Close'], window=14).atr
    sma_high = dataframe['High'].rolling(window=length).mean() + atr
    sma_low = dataframe['Low'].rolling(window=length).mean() - atr
    hlv = np.where(dataframe['Close'] > sma_high, 1, np.where(dataframe['Close'] < sma_low, -1, np.nan))
    hlv = pd.Series(hlv, index=dataframe.index).ffill()
    ssl_down = np.where(hlv < 0, sma_high, sma_low)
    ssl_up = np.where(hlv < 0, sma_low, sma_high)
    return pd.Series(ssl_down, index=dataframe.index), pd.Series(ssl_up, index=dataframe.index)


def ichimoku(dataframe, params):
    high = dataframe['High']
    low = dataframe['Low']
    close = dataframe['Close']
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
    cloud_top = pd.concat([senkou_span_a, senkou_span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_span_a, senkou_span_b], axis=1).min(axis=1)

    ichimoku_df = pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span,
        'future_green': future_green,
        'future_red': future_red,
        'cloud_top': cloud_top,
        'cloud_bottom': cloud_bottom
    }, index=dataframe.index)

    return ichimoku_df


# === Strategy Class ===
class IchimokuZemaStrategy:
    def __init__(self, symbol, timeframe, start_date, end_date, trade_type):
        self.symbol = symbol
        self.timeframe = timeframe
        self.informative_timeframe = '1h'
        self.trade_type = trade_type

        # Fetch data starting from extended_start_date
        self.data = fetch_data(symbol, timeframe, extended_start_date, end_date).loc[start_date:]
        self.high_tf_data = fetch_data(symbol, self.informative_timeframe, extended_start_date, end_date).loc[
                            start_date:]

        self.params = {
            'low_offset': low_offset,
            'high_offset': high_offset,
            'zema_len_buy': zema_len_buy,
            'zema_len_sell': zema_len_sell,
            'ssl_atr_period': ssl_atr_period,
            'ichimoku_params': ichimoku_params
        }
        self.calculate_indicators()
        self.generate_signals()
        self.run_backtest()

    # --- Indicator Calculations ---
    def calculate_indicators(self):
        self.data['zema_buy'] = vbt.MA.run(self.data['Close'], window=self.params['zema_len_buy'], ewm=True).ma
        self.data['zema_sell'] = vbt.MA.run(self.data['Close'], window=self.params['zema_len_sell'], ewm=True).ma
        self.data['zema_buy_adj'] = self.data['zema_buy'] * self.params['low_offset']
        self.data['zema_sell_adj'] = self.data['zema_sell'] * self.params['high_offset']

        ssl_down, ssl_up = ssl_atr(self.high_tf_data, length=self.params['ssl_atr_period'])
        self.high_tf_data['ssl_down'] = ssl_down
        self.high_tf_data['ssl_up'] = ssl_up
        self.high_tf_data['ssl_ok'] = (ssl_up > ssl_down).astype(int)
        self.high_tf_data['ssl_bear'] = (ssl_up < ssl_down).astype(int)

        ichimoku_df = ichimoku(self.high_tf_data, self.params['ichimoku_params'])
        self.high_tf_data = pd.concat([self.high_tf_data, ichimoku_df], axis=1)

        self.high_tf_data['ichimoku_ok'] = (
                (self.high_tf_data['tenkan_sen'] > self.high_tf_data['kijun_sen']) &
                (self.high_tf_data['Close'] > self.high_tf_data['cloud_top']) &
                (self.high_tf_data['future_green'] > 0) &
                (self.high_tf_data['chikou_span'] > self.high_tf_data['cloud_top'].shift(
                    -self.params['ichimoku_params']['displacement']))
        ).astype(int)

        self.high_tf_data['ichimoku_bear'] = (
                (self.high_tf_data['tenkan_sen'] < self.high_tf_data['kijun_sen']) &
                (self.high_tf_data['Close'] < self.high_tf_data['cloud_bottom']) &
                (self.high_tf_data['future_red'] > 0) &
                (self.high_tf_data['chikou_span'] < self.high_tf_data['cloud_bottom'].shift(
                    -self.params['ichimoku_params']['displacement']))
        ).astype(int)

        self.high_tf_data['ichimoku_valid'] = (~self.high_tf_data['senkou_span_b'].isna()).astype(int)

        self.high_tf_data['trend_pulse'] = (
                (self.high_tf_data['ichimoku_ok'] > 0) &
                (self.high_tf_data['ssl_ok'] > 0)
        ).astype(int)

        self.high_tf_data['bear_trend_pulse'] = (
                (self.high_tf_data['ichimoku_bear'] > 0) &
                (self.high_tf_data['ssl_bear'] > 0)
        ).astype(int)

        self.high_tf_data = self.high_tf_data.reindex(self.data.index, method='ffill')
        self.data = self.data.join(self.high_tf_data[['ichimoku_valid', 'trend_pulse', 'bear_trend_pulse']],
                                   how='left').fillna(0)

    # --- Signal Generation ---
    def generate_signals(self):
        self.data['buy_signal_long'] = (
                (self.data['ichimoku_valid'] > 0) &
                (self.data['bear_trend_pulse'] == 0) &
                # (self.data['trend_pulse'] == 1) &
                (self.data['Close'] < self.data['zema_buy_adj'])
        ).astype(int)

        self.data['sell_signal_long'] = (
            (self.data['Close'] > self.data['zema_sell_adj'])
        ).astype(int)

        self.data['buy_signal_short'] = (
                (self.data['ichimoku_valid'] > 0) &
                (self.data['trend_pulse'] == 0) &
                # (self.data['bear_trend_pulse'] == 1) &
                (self.data['Close'] > self.data['zema_buy_adj'])
        ).astype(int)

        self.data['sell_signal_short'] = (
            (self.data['Close'] < self.data['zema_sell_adj'])
        ).astype(int)

    def run_backtest(self):
        entries_long = self.data['buy_signal_long'] == 1
        exits_long = self.data['sell_signal_long'] == 1

        entries_short = self.data['buy_signal_short'] == 1
        exits_short = self.data['sell_signal_short'] == 1

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

        if trade_type == 'long':
            self.pf = vbt.Portfolio.from_signals(
                close=self.data['Close'],
                entries=entries_long,
                exits=exits_long,
                freq=freq,  # Set frequency here
                init_cash=10000,
                fees=0.001,
                slippage=0.001
            )
        elif trade_type == 'short':
            self.pf = vbt.Portfolio.from_signals(
                close=self.data['Close'],
                entries=False,  # No long entries
                exits=False,
                short_entries=entries_short,
                short_exits=exits_short,
                freq=freq,  # Set frequency here
                init_cash=10000,
                fees=0.001,
                slippage=0.001
            )
        else:  # Both long and short
            self.pf = vbt.Portfolio.from_signals(
                close=self.data['Close'],
                entries=entries_long,
                exits=exits_long,
                short_entries=entries_short,
                short_exits=exits_short,
                freq=freq,  # Set frequency here
                init_cash=10000,
                fees=0.001,
                slippage=0.001
            )

        print("toto")

    def plot_strategy(self, save_path=None, show=False):
        """
        Plot the strategy chart with indicators and signals in different windows.

        Parameters:
        - save_path (str): Path to save the chart.
        - show (bool): Display the chart interactively.
        """
        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=7, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Close Price & Signals",
                "ZEMA Sell Adj & Close",
                "ZEMA Buy Adj & Close",
                "ZEMA Sell Adj & ZEMA Buy Adj",
                "Close + ZEMA Buy Adj + ZEMA Sell Adj",
                "Ichimoku & Other Indicators",
                "Trend Indicators"
            )
        )

        # === Subplot 1: Close price and entry/exit signals ===
        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['Close'],
            mode='lines', name='Close Price'
        ), row=1, col=1)

        if self.trade_type in ['long', 'both']:
            fig.add_trace(go.Scatter(
                x=self.data.index[self.data['buy_signal_long'] == 1],
                y=self.data['Close'][self.data['buy_signal_long'] == 1],
                mode='markers', marker=dict(color='green', size=10),
                name='Long Entry'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index[self.data['sell_signal_long'] == 1],
                y=self.data['Close'][self.data['sell_signal_long'] == 1],
                mode='markers', marker=dict(color='red', size=10),
                name='Long Exit'
            ), row=1, col=1)

        if self.trade_type in ['short', 'both']:
            fig.add_trace(go.Scatter(
                x=self.data.index[self.data['buy_signal_short'] == 1],
                y=self.data['Close'][self.data['buy_signal_short'] == 1],
                mode='markers', marker=dict(color='blue', size=10),
                name='Short Entry'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index[self.data['sell_signal_short'] == 1],
                y=self.data['Close'][self.data['sell_signal_short'] == 1],
                mode='markers', marker=dict(color='orange', size=10),
                name='Short Exit'
            ), row=1, col=1)

        # === Subplot 2: ZEMA Sell Adj & Close ===
        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['Close'],
            mode='lines', name='Close Price'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['zema_sell_adj'],
            mode='lines', name='ZEMA Sell Adj', line=dict(color='red', dash='dot')
        ), row=2, col=1)

        # === Subplot 3: ZEMA Buy Adj & Close ===
        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['Close'],
            mode='lines', name='Close Price'
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['zema_buy_adj'],
            mode='lines', name='ZEMA Buy Adj', line=dict(color='blue', dash='dot')
        ), row=3, col=1)

        # === Subplot 4: ZEMA Sell Adj & ZEMA Buy Adj ===
        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['zema_sell_adj'],
            mode='lines', name='ZEMA Sell Adj', line=dict(color='red', dash='dot')
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['zema_buy_adj'],
            mode='lines', name='ZEMA Buy Adj', line=dict(color='blue', dash='dot')
        ), row=4, col=1)

        # === Subplot 5: Close + ZEMA Buy Adj + ZEMA Sell Adj ===
        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['Close'],
            mode='lines', name='Close Price'
        ), row=5, col=1)

        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['zema_buy_adj'],
            mode='lines', name='ZEMA Buy Adj', line=dict(color='blue', dash='dot')
        ), row=5, col=1)

        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['zema_sell_adj'],
            mode='lines', name='ZEMA Sell Adj', line=dict(color='red', dash='dot')
        ), row=5, col=1)

        # === Subplot 6: Ichimoku indicators ===
        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['ichimoku_valid'],
            mode='lines', name='Ichimoku Valid', line=dict(color='green', dash='dot')
        ), row=6, col=1)

        # === Subplot 7: Trend indicators ===
        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['trend_pulse'],
            mode='lines', name='Trend Pulse', line=dict(color='orange', dash='dot')
        ), row=7, col=1)

        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data['bear_trend_pulse'],
            mode='lines', name='Bear Trend Pulse', line=dict(color='red', dash='dot')
        ), row=7, col=1)

        # Update layout for better visualization
        fig.update_layout(
            height=1400,  # Adjust height to fit all subplots
            title=f"Strategy Chart for {self.symbol} - {self.timeframe} ({self.trade_type})",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark"
        )

        if save_path:
            fig.write_image(save_path)
            print(f"Strategy chart saved to {save_path}.")

        if show:
            fig.show()

        return fig

    def plot_ohlcv(self, save_path=None, show=False):
        """
        Plot the OHLCV chart using Plotly.

        Parameters:
        - save_path (str): Path to save the chart.
        - show (bool): Display the chart interactively.
        """
        fig = go.Figure(data=[
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='OHLC'
            )
        ])

        fig.update_layout(
            title=f"OHLCV Chart for {self.symbol} - {self.timeframe}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )

        if save_path:
            fig.write_image(save_path)
            print(f"OHLCV chart saved to {save_path}.")

        if show:
            fig.show()

        return fig

    # --- Results ---
    def get_results(self):
        return self.pf.stats()

    # --- Portfolio Visualization ---
    def plot_portfolio(self, save_path=None, show=False):
        """
        Plot the portfolio and optionally save or display the graph.

        Parameters:
        - save_path (str): Path to save the graph. If None, the graph will not be saved.
        - show (bool): Whether to display the graph interactively.
        """
        fig = self.pf.plot()

        if save_path:
            fig.write_image(save_path)
            print(f"Graph saved to {save_path}.")

        if show:
            fig.show()

        return fig


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



if __name__ == "__main__":
    results = {}
    stats_list = []  # To collect stats for all symbols and timeframes

    # Ensure the results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    for symbol in symbols:
        for trade_type in lst_trade_type:
            for timeframe in timeframes:
                print(f"Running strategy for {symbol} on {timeframe} timeframe with trade_type='{trade_type}'.")
                strategy = IchimokuZemaStrategy(symbol, timeframe, start_date, end_date, trade_type)
                stats = strategy.get_results()

                # Add stats to the list with additional metadata
                stats['Symbol'] = symbol
                stats['Timeframe'] = timeframe
                stats['Trade_Type'] = trade_type
                stats_list.append(stats)

                # Save the OHLCV graph
                ohlcv_path = f"results/{symbol}_{timeframe}_{trade_type}_ohlcv_graph.png"
                # strategy.plot_ohlcv(save_path=ohlcv_path, show=False)
                strategy.plot_ohlcv(show=False)

                # Save the strategy graph
                strategy_path = f"results/{symbol}_{timeframe}_{trade_type}_strategy_graph.png"
                # strategy.plot_strategy(save_path=strategy_path, show=False)
                strategy.plot_strategy(show=False)

                # Display and save stats for each iteration
                results[(symbol, timeframe)] = stats
                print(stats)
                print("-" * 50)

                # Save the portfolio graph
                graph_path = f"results/{symbol}_{timeframe}_{trade_type}_portfolio_graph.png"
                # strategy.plot_portfolio(save_path=graph_path, show=False)
                strategy.plot_portfolio(show=False)

    # Combine all stats into a single DataFrame
    all_stats_df = pd.DataFrame(stats_list)

    # Desired column order
    column_order = [
        'Symbol', 'Timeframe', 'Trade_Type', 'Start Value', 'End Value',
        'Total Return [%]', 'Benchmark Return [%]', 'Win Rate [%]',
        'Sharpe Ratio', 'Calmar Ratio', 'Omega Ratio', 'Sortino Ratio',
        'Max Drawdown [%]', 'Total Trades', 'Best Trade [%]', 'Worst Trade [%]',
        'Avg Winning Trade [%]', 'Avg Losing Trade [%]', 'Total Fees Paid',
        'Total Closed Trades', 'Total Open Trades', 'Open Trade PnL',
        'Avg Winning Trade Duration', 'Avg Losing Trade Duration',
        'Profit Factor', 'Start', 'End', 'Period', 'Max Gross Exposure [%]',
        'Max Drawdown Duration', 'Expectancy'
    ]

    # Reorder columns
    all_stats_df = all_stats_df[column_order]

    # Get a unique filename
    stats_csv_path = get_unique_filename("results", "portfolio_stats_summary", ".csv")

    # Save the combined stats to a CSV file
    all_stats_df.to_csv(stats_csv_path, index=False)

    print(f"Portfolio statistics saved to {stats_csv_path}.")
