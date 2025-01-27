import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # hides certain ADX warnings

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TA imports
from ta.trend import IchimokuIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator  # Not used here, but left if you want to expand

###############################################################################
# 1. Data Fetching
###############################################################################
def get_data(symbol="BTC-USD", period="2y", interval="1d"):
    """
    Fetches historical data from Yahoo Finance using yfinance.
    Returns a DataFrame with columns: [open, high, low, close, volume].
    """
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    return df

###############################################################################
# 2. Strategies
###############################################################################
def strategy_1_ichimoku_only(df):
    """
    Strategy 1: Ichimoku ONLY (very simplistic).
      - Buy if close > cloud top
      - Sell if close < cloud bottom
    """
    ichi = IchimokuIndicator(
        high=df["high"],
        low=df["low"],
        window1=9,
        window2=26,
        window3=52,
        visual=False,
        fillna=False
    )
    df["tenkan_sen"] = ichi.ichimoku_conversion_line()
    df["kijun_sen"]  = ichi.ichimoku_base_line()
    df["senkou_a"]   = ichi.ichimoku_a()
    df["senkou_b"]   = ichi.ichimoku_b()

    df["cloud_top"]    = np.where(df["senkou_a"] > df["senkou_b"], df["senkou_a"], df["senkou_b"])
    df["cloud_bottom"] = np.where(df["senkou_a"] < df["senkou_b"], df["senkou_a"], df["senkou_b"])

    buy_signal  = df["close"] > df["cloud_top"]
    sell_signal = df["close"] < df["cloud_bottom"]

    df["strategy_1_signal"] = 0
    df.loc[buy_signal,  "strategy_1_signal"] = 1
    df.loc[sell_signal, "strategy_1_signal"] = -1
    return df

def strategy_6_price_action(df, window=20):
    """
    Strategy 6: Simple Price Action using Donchian-style breakout.
      - Buy if close crosses above the (window)-day high
      - Sell if close crosses below the (window)-day low
    """
    df["highest_window"] = df["high"].rolling(window=window).max()
    df["lowest_window"]  = df["low"].rolling(window=window).min()

    buy_signal  = df["close"] > df["highest_window"].shift(1)
    sell_signal = df["close"] < df["lowest_window"].shift(1)

    df["strategy_6_signal"] = 0
    df.loc[buy_signal,  "strategy_6_signal"] = 1
    df.loc[sell_signal, "strategy_6_signal"] = -1
    return df

###############################################################################
# 3. Simple Backtest & Buy-and-Hold
###############################################################################
def simple_backtest(df, signal_col, initial_capital=10000):
    """
    A very basic backtest:
      - +1 signal => go long with entire capital (if not already in position)
      - -1 signal => exit/flat (if in position)
      - No partial sizing, no fees, no slippage, etc.
    """
    df = df.copy()
    df["position"] = 0
    df["cash"] = initial_capital
    df["holdings"] = 0
    df["total_equity"] = initial_capital

    in_position = False
    shares_held = 0
    cash = initial_capital

    for i in range(1, len(df)):
        signal = df.iloc[i][signal_col]
        price = df.iloc[i]["close"]

        # Enter
        if signal == 1 and not in_position:
            shares_held = cash / price
            cash = 0
            in_position = True
        # Exit
        elif signal == -1 and in_position:
            cash = shares_held * price
            shares_held = 0
            in_position = False

        # Update daily equity
        df.at[df.index[i], "position"] = 1 if in_position else 0
        holdings_value = shares_held * price
        df.at[df.index[i], "cash"] = cash
        df.at[df.index[i], "holdings"] = holdings_value
        df.at[df.index[i], "total_equity"] = cash + holdings_value

    # Final performance
    final_equity = df["total_equity"].iloc[-1]
    return_pct = (final_equity - initial_capital) / initial_capital * 100

    # Calculate drawdown
    df["peak_equity"] = df["total_equity"].cummax()
    df["drawdown"] = (df["total_equity"] - df["peak_equity"]) / df["peak_equity"]
    max_drawdown = df["drawdown"].min() * 100

    return {
        "final_equity": final_equity,
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown,
        "backtest_history": df
    }

def buy_and_hold(df, initial_capital=10000):
    """
    Buy & Hold from the first date to the last date:
      - invests all capital at first close
      - final equity at last close
    """
    df = df.copy()
    first_close = df["close"].iloc[0]
    shares_held = initial_capital / first_close

    # Each day's equity
    df["bh_equity"] = df["close"] * shares_held
    df["peak_equity"] = df["bh_equity"].cummax()
    df["drawdown"] = (df["bh_equity"] - df["peak_equity"]) / df["peak_equity"]

    final_equity = df["bh_equity"].iloc[-1]
    return_pct = (final_equity - initial_capital) / initial_capital * 100
    max_drawdown = df["drawdown"].min() * 100

    return {
        "final_equity": final_equity,
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown,
        "history": df
    }

###############################################################################
# 4. Plotting Functions (with file saving)
###############################################################################
import math

def plot_signals(df, signal_col, title="Strategy Signals", save_dir="charts"):
    """
    Plots the close price of BTC and overlays buy/sell signal markers.
    Optionally saves the plot to `save_dir` (charts/ by default).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["close"], label="BTC Close", color="blue")

    # Buy signals
    buy_signals = df[df[signal_col] == 1]
    # Sell signals
    sell_signals = df[df[signal_col] == -1]

    # Plot buy/sell markers
    plt.scatter(buy_signals.index, buy_signals["close"], marker="^", color="green", label="Buy", s=100)
    plt.scatter(sell_signals.index, sell_signals["close"], marker="v", color="red", label="Sell", s=100)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()

    # Save to PNG
    filename = f"{title.replace(' ', '_')}_signals.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def plot_price_and_trend(df, signal_col, title="Price + Trend Shading", save_dir="charts"):
    """
    Plots BTC's close price and:
      - Shades green where position=1
      - Marks buy/sell signals from `signal_col`
    Saves the plot to `save_dir`.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["close"], label="BTC Close", color="blue")

    # Shade where position == 1
    plt.fill_between(
        df.index, df["close"].min(), df["close"].max(),
        where=(df["position"] == 1), color="green", alpha=0.1,
        label="Bullish (in position)"
    )

    # Mark the buy/sell signals
    buy_signals = df[df[signal_col] == 1]
    sell_signals = df[df[signal_col] == -1]

    plt.scatter(buy_signals.index, buy_signals["close"], marker="^", color="green", label="Buy", s=100)
    plt.scatter(sell_signals.index, sell_signals["close"], marker="v", color="red", label="Sell", s=100)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()

    # Save to PNG
    filename = f"{title.replace(' ', '_')}_trend.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def plot_equity_curves(strategies_dict, save_dir="charts"):
    """
    Takes a dict of {name: backtest_df} and plots each df['total_equity']
    (or 'bh_equity') on the same chart. Saves to `save_dir`.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12,6))
    for name, history_df in strategies_dict.items():
        if "total_equity" in history_df.columns:
            plt.plot(history_df.index, history_df["total_equity"], label=name)
        elif "bh_equity" in history_df.columns:
            plt.plot(history_df.index, history_df["bh_equity"], label=name)

    plt.title("Equity Curves Comparison")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()

    filename = "Equity_Curves_Comparison.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def plot_all_strategies_trends(all_strategies_info, save_dir="charts"):
    """
    Creates a single figure with subplots for each strategy's price + trend shading.
    We'll make a 1 row x 2 columns layout for 2 strategies.
    :param all_strategies_info: a list of tuples [(title, result_dict, signal_col), ...]
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_strats = len(all_strategies_info)  # likely 2
    rows = 1
    cols = num_strats

    fig, axes = plt.subplots(rows, cols, figsize=(7 * num_strats, 6), sharex=True)
    if num_strats == 1:
        axes = [axes]  # ensure it's iterable
    else:
        axes = axes.flatten()

    for i, (title, result, sig_col) in enumerate(all_strategies_info):
        ax = axes[i]
        df = result["backtest_history"]

        ax.plot(df.index, df["close"], label="BTC Close", color="blue")

        # Shade where position == 1
        ax.fill_between(
            df.index, df["close"].min(), df["close"].max(),
            where=(df["position"] == 1), color="green", alpha=0.1
        )

        # Mark the buy/sell signals
        buy_signals = df[df[sig_col] == 1]
        sell_signals = df[df[sig_col] == -1]
        ax.scatter(buy_signals.index, buy_signals["close"], marker="^", color="green", s=50)
        ax.scatter(sell_signals.index, sell_signals["close"], marker="v", color="red", s=50)

        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    filename = "All_Strategies_Trend_Subplots.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

###############################################################################
# 5. Main Execution
###############################################################################
if __name__ == "__main__":
    # 1) Get data (2 years of BTC-USD)
    df_main = get_data("BTC-USD", "2y", "1d")

    # ---------------------------------------------------------------------------
    # Strategy 1: Ichimoku Only
    df_s1 = df_main.copy()
    df_s1 = strategy_1_ichimoku_only(df_s1)
    result_s1 = simple_backtest(df_s1, "strategy_1_signal")

    plot_signals(df_s1, "strategy_1_signal", title="Strategy 1 Ichimoku Only")
    plot_price_and_trend(result_s1["backtest_history"], "strategy_1_signal",
                         title="Strategy 1 Ichimoku Only Trend")

    # Strategy 6: Price Action (Donchian breakout, 20-day)
    df_s6 = df_main.copy()
    df_s6 = strategy_6_price_action(df_s6, window=20)
    result_s6 = simple_backtest(df_s6, "strategy_6_signal")

    plot_signals(df_s6, "strategy_6_signal", title="Strategy 6 Price Action")
    plot_price_and_trend(result_s6["backtest_history"], "strategy_6_signal",
                         title="Strategy 6 Price Action Trend")

    # ---------------------------------------------------------------------------
    # Buy & Hold Benchmark
    df_bh = df_main.copy()
    bh_result = buy_and_hold(df_bh)

    # 2) Print results summary
    summary = pd.DataFrame({
        "Strategy": [
            "S1_Ichimoku_Only",
            "S6_PriceAction",
            "Buy_Hold"
        ],
        "Final_Equity": [
            result_s1["final_equity"],
            result_s6["final_equity"],
            bh_result["final_equity"]
        ],
        "Return_%": [
            result_s1["return_pct"],
            result_s6["return_pct"],
            bh_result["return_pct"]
        ],
        "Max_Drawdown_%": [
            result_s1["max_drawdown_pct"],
            result_s6["max_drawdown_pct"],
            bh_result["max_drawdown_pct"]
        ]
    })

    print("=== Strategy Comparison ===")
    print(summary)

    # ---------------------------------------------------------------------------
    # 3) Equity Curves Plot (Strategy 1, Strategy 6, and Buy & Hold)
    s1_hist = result_s1["backtest_history"].rename(columns={"total_equity": "total_equity_s1"})
    s6_hist = result_s6["backtest_history"].rename(columns={"total_equity": "total_equity_s6"})
    df_bh   = bh_result["history"].rename(columns={"bh_equity": "total_equity_bh"})

    s1_hist["total_equity"] = s1_hist["total_equity_s1"]
    s6_hist["total_equity"] = s6_hist["total_equity_s6"]
    df_bh["total_equity"]   = df_bh["total_equity_bh"]

    strategy_histories = {
        "S1_Ichimoku_Only": s1_hist,
        "S6_PriceAction":   s6_hist,
        "BuyHold":          df_bh
    }

    plot_equity_curves(strategy_histories, save_dir="charts")

    # ---------------------------------------------------------------------------
    # 4) Create a composite figure: subplots for the 2 strategies' trend shading
    #    (Buy & Hold doesn't have signals, so we skip it.)
    all_strategies_info = [
        ("Strategy 1 Ichimoku Only", result_s1, "strategy_1_signal"),
        ("Strategy 6 Price Action", result_s6, "strategy_6_signal"),
    ]

    plot_all_strategies_trends(all_strategies_info, save_dir="charts")
