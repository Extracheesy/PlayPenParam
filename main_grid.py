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
    # Charger les données
    data = load_data(symbol, start_date, end_date, timeframe)
    close = data['Close']

    # Générer les niveaux de grille
    grid_levels = generate_grid_levels(grid_low, grid_high, grid_size)

    # Créer les DataFrames des niveaux de grille et des prix de clôture
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

    # Déterminer les entrées (achats) lorsque le prix croise en dessous des niveaux de grille
    entries = close_df.vbt.crossed_below(grid_levels_df)

    # Les exits seront basés sur un Take Profit de 1% à partir du prix d'entrée
    tp_pct = grid_size / 100  # 1% en fraction décimale

    # Calculer la taille de chaque position
    num_levels = len(grid_levels)
    position_size = (cash * leverage) / num_levels

    # Créer un mapping pour regrouper toutes les colonnes
    group_by = pd.Index(['Combined'] * len(close_df.columns))

    entries_merged = entries.any(axis=1)

    # Créer le portefeuille en utilisant le Take Profit et le regroupement
    portfolio = vbt.Portfolio.from_signals(
        # close=close_df,
        close=close,
        # entries=entries,
        entries=entries_merged,
        exits=None,  # Pas d'exits basés sur les niveaux de grille
        size=position_size,
        size_type='value',
        init_cash=cash,
        fees=0.0,
        slippage=0.0,
        tp_stop=tp_pct,
        # group_by=group_by
    )

    return portfolio, close, grid_levels, entries

# Exemple d'utilisation
# start_date = '2021-01-01'  # Utilisez des dates historiques valides
# end_date = '2021-12-31'

start_date = '2024-03-01'  # Utilisez des dates historiques valides
end_date = '2024-11-07'

if True:
    symbol = 'BTC/USDT'
    grid_high = 70000
    grid_low = 30000
else:
    symbol = 'PEPE/USDT'
    grid_high = 1.718e-05
    grid_low = 2.68e-06

grid_size = 1.0  # 1%
leverage = 10
timeframe = '5m'
cash = 10000

portfolio, close, grid_levels, entries = run_grid_trading_strategy(
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

# Afficher les statistiques du portefeuille
print(portfolio.stats())

# Tracer la performance du portefeuille
portfolio.plot(title='Performance du Portefeuille avec Take Profit de 1%').show()
