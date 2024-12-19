import os
import numpy as np
import pandas as pd
import vectorbt as vbt

def load_data(symbol, start_date, end_date, timeframe):
    data_dir = 'data_grid'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    safe_symbol = symbol.replace('/', '_')
    filename = f"{safe_symbol}_{start_date}_{end_date}_{timeframe}.csv"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        data = pd.read_csv(filepath, parse_dates=True, index_col=0)
    else:
        data = vbt.CCXTData.download(
            symbols=symbol,
            start=start_date,
            end=end_date,
            timeframe=timeframe,
            exchange='binance',
            show_progress=True
        ).get()
        data.to_csv(filepath, index=True)
    return data

def generate_grid_levels(grid_low, grid_high, grid_size):
    grid_levels = [grid_low]
    while grid_levels[-1] * (1 + grid_size / 100) <= grid_high:
        grid_levels.append(grid_levels[-1] * (1 + grid_size / 100))
    return np.array(grid_levels)

def run_grid_trading_strategy(
    start_date,
    end_date,
    symbol,
    grid_size,
    grid_high,
    grid_low,
    leverage,
    timeframe,
    cash
):
    data = load_data(symbol, start_date, end_date, timeframe)
    close = data['Close']

    grid_levels = generate_grid_levels(grid_low, grid_high, grid_size)

    grid_levels_df = pd.DataFrame(
        np.tile(grid_levels, (len(close), 1)),
        index=close.index,
        columns=grid_levels
    )

    close_df = pd.DataFrame(
        np.repeat(close.values.reshape(-1, 1), len(grid_levels), axis=1),
        index=close.index,
        columns=grid_levels
    )

    entries = close_df.vbt.crossed_below(grid_levels_df)
    exits = close_df.vbt.crossed_above(grid_levels_df)

    entries_merged = entries.any(axis=1)
    exits_merged = exits.any(axis=1)

    num_levels = len(grid_levels)
    position_size = (cash * leverage) / num_levels

    # Create a group mapping that assigns all columns to a single group
    group_by = pd.Index(['Combined'] * len(close_df.columns))

    # Create the portfolio with grouping
    portfolio = vbt.Portfolio.from_signals(
        # close=close_df,
        close=close,
        # entries=entries,
        # exits=exits,
        entries=entries_merged,
        exits=exits_merged,
        size=position_size,
        init_cash=cash,
        fees=0.0,
        slippage=0.0,
        # group_by=group_by
    )

    return portfolio, close, grid_levels, entries, exits

# Example usage
start_date = '2024-01-01'
end_date = '2024-10-07'
symbol = 'PEPE/USDT'
grid_size = 1.0  # 1%
grid_high = 70000
grid_low = 30000
leverage = 1
# timeframe = '1h'
timeframe = '5m'
cash = 10000

portfolio, close, grid_levels, entries, exits = run_grid_trading_strategy(
    start_date,
    end_date,
    symbol,
    grid_size,
    grid_high,
    grid_low,
    leverage,
    timeframe,
    cash
)

# Display portfolio statistics
print(portfolio.stats())

# Plot the combined portfolio performance
portfolio.plot(title='Combined Portfolio Performance').show()
