# ====== Importation des bibliothèques nécessaires ======
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import os

# ====== Paramètres globaux ======
CRYPTOS = ['PEPEUSDT', 'SOLUSDT', 'ETHUSDT', 'BTCUSDT']
TIMEFRAMES = ['5m', '15m', '30m', '1h', '2h']
RESULTS_DIR = 'results'
DATA_DIR = 'data'  # Répertoire pour stocker les données

# Création des répertoires pour les résultats et les données
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ====== Définition des fonctions utilitaires ======

def zema(series, period):
    """Calcule la Zero-Lag Exponential Moving Average (ZEMA)."""
    lag = (period - 1) // 2
    series_adjusted = series + (series - series.shift(lag))
    zema = series_adjusted.ewm(span=period, adjust=False).mean()
    return zema


def ssl_atr(high, low, close, atr, length=7):
    """Calcule les indicateurs SSL basés sur l'ATR."""
    sma_high = high.rolling(window=length).mean() + atr
    sma_low = low.rolling(window=length).mean() - atr
    hlv = np.where(close > sma_high, 1, np.where(close < sma_low, -1, np.nan))
    hlv = pd.Series(hlv, index=close.index).ffill()
    ssl_down = np.where(hlv < 0, sma_high, sma_low)
    ssl_up = np.where(hlv < 0, sma_low, sma_high)
    return pd.Series(ssl_down, index=close.index), pd.Series(ssl_up, index=close.index)


def ichimoku_cloud(high, low, close, displacement=30):
    """Calcule les composantes de l'indicateur Ichimoku."""
    # Calcul des lignes de conversion et de base
    conversion_line = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    base_line = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2

    # Calcul des Span A et B
    leading_span_a = (conversion_line + base_line) / 2
    leading_span_b = (high.rolling(window=52).max() + low.rolling(window=52).min()) / 2

    # Décalage des Span
    leading_span_a = leading_span_a.shift(displacement)
    leading_span_b = leading_span_b.shift(displacement)

    # Calcul du lagging span
    lagging_span = close.shift(-displacement)

    return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span


def calculate_indicators(dataframe, zema_length_buy, zema_length_sell, low_offset, high_offset):
    """Calcule tous les indicateurs nécessaires pour la stratégie."""
    # ====== Indicateurs rapides ======
    zema_buy = zema(dataframe['Close'], period=zema_length_buy)
    zema_sell = zema(dataframe['Close'], period=zema_length_sell)

    dataframe['zema_buy'] = zema_buy * low_offset
    dataframe['zema_sell'] = zema_sell * high_offset

    # ====== Indicateurs lents (Timeframe informatif) ======
    # Resample en timeframe 1h pour les indicateurs Ichimoku
    df_informative = dataframe.copy()
    df_informative = df_informative.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calcul de l'ATR
    atr = vbt.ATR.run(
        high=df_informative['High'],
        low=df_informative['Low'],
        close=df_informative['Close'],
        window=14
    ).atr

    # Calcul des SSL ATR
    ssl_down, ssl_up = ssl_atr(
        high=df_informative['High'],
        low=df_informative['Low'],
        close=df_informative['Close'],
        atr=atr,
        length=10
    )

    df_informative['ssl_down'] = ssl_down
    df_informative['ssl_up'] = ssl_up

    # Calcul des composantes Ichimoku
    conversion_line, base_line, leading_span_a, leading_span_b, lagging_span = ichimoku_cloud(
        high=df_informative['High'],
        low=df_informative['Low'],
        close=df_informative['Close']
    )

    df_informative['tenkan_sen'] = conversion_line
    df_informative['kijun_sen'] = base_line
    df_informative['senkou_span_a'] = leading_span_a
    df_informative['senkou_span_b'] = leading_span_b
    df_informative['chikou_span'] = lagging_span

    # Fusion des indicateurs informatifs avec le dataframe principal
    dataframe = dataframe.join(df_informative[['ssl_down', 'ssl_up', 'tenkan_sen', 'kijun_sen',
                                               'senkou_span_a', 'senkou_span_b', 'chikou_span']], how='left')
    dataframe.fillna(method='ffill', inplace=True)

    return dataframe


def generate_signals(dataframe):
    """Génère les signaux d'achat et de vente basés sur les indicateurs calculés."""
    # Conditions Ichimoku pour tendance haussière
    ichimoku_bull = (
            (dataframe['tenkan_sen'] > dataframe['kijun_sen']) &
            (dataframe['Close'] > dataframe['senkou_span_a']) &
            (dataframe['Close'] > dataframe['senkou_span_b']) &
            (dataframe['chikou_span'] > dataframe['Close'].shift(26))
    )

    # Conditions SSL pour tendance haussière
    ssl_bull = dataframe['ssl_up'] > dataframe['ssl_down']

    # Condition d'achat
    buy_condition = (
            ichimoku_bull &
            ssl_bull &
            (dataframe['Close'] < dataframe['zema_buy'])
    )

    # Condition de vente
    sell_condition = dataframe['Close'] > dataframe['zema_sell']

    # Retourne des booléens
    return buy_condition.astype(bool), sell_condition.astype(bool)


def backtest_strategy(dataframe, buy_signal, sell_signal, timeframe):
    """Exécute le backtest de la stratégie."""
    portfolio = vbt.Portfolio.from_signals(
        close=dataframe['Close'],
        entries=buy_signal,
        exits=sell_signal,
        direction='longonly',
        fees=0.001,  # Ajustez les frais si nécessaire
        freq=timeframe  # Définir la fréquence explicitement
    )
    return portfolio


# ====== Définition des fonctions de traçage ======

def plot_strategy(close, entries, exits, title="Stratégie"):
    """Superpose les signaux d'achat et de vente sur le graphique des prix de clôture."""
    fig = close.vbt.plot(title=title)
    entries.vbt.signals.plot_as_entries(close, fig=fig)
    exits.vbt.signals.plot_as_exits(close, fig=fig)
    return fig


def plot_strategy_signal(symbol, data, close, entries, exits, title="Stratégie"):
    """Superpose les signaux d'achat et de vente avec des marques distinctes."""
    new_exits = exits.vbt.signals.first_after(entries, reset_wait=0)
    new_entries = entries.vbt.signals.first_after(exits)
    """
    fig = data.plot(
        symbol=symbol,
        ohlc_trace_kwargs=dict(opacity=0.5),
        plot_volume=False
    )
    """
    df_filtered = data[["Open", "Close", "High", "Low"]]
    df_filtered = data[["Close"]]
    fig = df_filtered.plot()

    entries[symbol].vbt.signals.plot_as_entries(
        y=df_filtered.get("Close", symbol), fig=fig)
    exits[symbol].vbt.signals.plot_as_exits(
        y=df_filtered.get("Close", symbol), fig=fig)
    new_entries[symbol].vbt.signals.plot_as_entry_marks(
        y=df_filtered.get("Close", symbol), fig=fig,
        trace_kwargs=dict(name="New entries"))
    new_exits[symbol].vbt.signals.plot_as_exit_marks(
        y=df_filtered.get("Close", symbol), fig=fig,
        trace_kwargs=dict(name="New exits"))
    fig.show()

    return fig


def plot_portfolio(portfolio, title="Portefeuille"):
    """Trace la performance du portefeuille."""
    fig = portfolio.plot()
    return fig


# ====== Fonction pour récupérer les données avec mise en cache ======

def get_data(crypto, timeframe, start, end, limit=1000):
    """
    Récupère les données pour une crypto et un timeframe.
    Si les données sont déjà stockées localement avec les dates correspondantes, les lit depuis le fichier.
    Sinon, les télécharge et les sauvegarde localement.
    """
    # Formater les dates pour le nom de fichier (YYYY-MM-DD)
    start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end).strftime('%Y-%m-%d')

    filename = f"{DATA_DIR}/{crypto}_{timeframe}_{start_str}_{end_str}.h5"
    if os.path.exists(filename):
        print(f"  Lecture des données depuis {filename}...")
        try:
            data = vbt.HDFData.pull(filename)
            return data.get()
        except Exception as e:
            print(f"  Erreur lors de la lecture du fichier {filename}: {e}")
            print(f"  Tentative de téléchargement des données pour {crypto}...")

    print(f"  Téléchargement des données pour {crypto}...")
    try:
        data = vbt.BinanceData.pull(
            crypto,
            start=start,  # Date de début
            end=end,  # Date de fin
            timeframe=timeframe,
            limit=limit  # Ajustez le limit si nécessaire
        )
        print(f"  Sauvegarde des données dans {filename}...")
        data.to_hdf(filename)
        return data.get()
    except Exception as e:
        print(f"  Erreur lors de la récupération des données pour {crypto}: {e}")
        return None


# ====== Exécution de la stratégie pour chaque crypto et timeframe ======

results = {}

# Définir les dates de début et de fin globales
start_date = '2024-01-01'
end_date = '2024-10-25'

for timeframe in TIMEFRAMES:
    print(f"Traitement du timeframe : {timeframe}")
    results[timeframe] = {}
    for crypto in CRYPTOS:
        print(f"  Récupération des données pour {crypto}...")

        # Récupérer les données avec mise en cache
        df = get_data(crypto, timeframe, start=start_date, end=end_date, limit=1000)
        if df is None:
            continue

        df = df.dropna()

        if df.empty:
            print(f"  Aucune donnée disponible pour {crypto} sur le timeframe {timeframe}.")
            continue

        print(f"  Calcul des indicateurs pour {crypto}...")
        df = calculate_indicators(
            dataframe=df,
            zema_length_buy=51,  # Paramètres optimisés
            zema_length_sell=72,
            low_offset=0.964,
            high_offset=1.004
        )

        print(f"  Génération des signaux pour {crypto}...")
        buy_signal, sell_signal = generate_signals(df)

        # Vérification des types de signaux
        assert buy_signal.dtype == bool, "buy_signal doit être de type booléen"
        assert sell_signal.dtype == bool, "sell_signal doit être de type booléen"

        print(f"  Exécution du backtest pour {crypto}...")
        portfolio = backtest_strategy(df, buy_signal, sell_signal, timeframe)

        print(portfolio.stats())

        # Sauvegarde des résultats
        results[timeframe][crypto] = {
            'data': df,
            'portfolio': portfolio
        }

        # Tracer le portefeuille
        fig_portfolio = plot_portfolio(portfolio, title=f"{crypto} - {timeframe} Portfolio")
        # fig_portfolio.write_image(f"{RESULTS_DIR}/{crypto}_{timeframe}_portfolio.png")

        """
        # Génération des graphiques
        print(f"  Génération des graphiques pour {crypto}...")
        # Tracer les signaux avec des marques distinctes
        fig_signals = plot_strategy_signal(crypto, df, df['Close'], buy_signal, sell_signal,
                                           title=f"{crypto} - {timeframe} Signals")
        fig_signals.write_image(f"{RESULTS_DIR}/{crypto}_{timeframe}_signals.png")

        # Tracer les signaux de manière standard (optionnel)
        # Vous pouvez commenter cette ligne si vous ne souhaitez pas générer deux images de signaux
        fig_signals_standard = plot_strategy(df['Close'], buy_signal, sell_signal,
                                             title=f"{crypto} - {timeframe} Signals")
        fig_signals_standard.write_image(f"{RESULTS_DIR}/{crypto}_{timeframe}_signals_standard.png")
        """


# ====== Comparaison des résultats ======

# Génération des heatmaps pour les paramètres
print("Génération des heatmaps pour les paramètres...")
zema_lengths = np.arange(30, 91, 5)
offsets = np.linspace(0.95, 1.05, 11)

for timeframe in TIMEFRAMES:
    for crypto in CRYPTOS:
        if crypto not in results[timeframe]:
            continue  # Passer si aucune donnée disponible

        df = results[timeframe][crypto]['data']
        heatmap_data = []

        print(f"  Génération de la heatmap pour {crypto} - {timeframe}...")
        for zema_length in zema_lengths:
            row = []
            zema_series = zema(df['Close'], period=zema_length)
            for offset in offsets:
                zema_current = zema_series * offset
                buy_signal = (df['Close'] < zema_current).astype(bool)
                sell_signal = (df['Close'] > zema_current).astype(bool)
                portfolio = backtest_strategy(df, buy_signal, sell_signal, timeframe)
                row.append(portfolio.total_return())
            heatmap_data.append(row)

        # Création de la heatmap
        heatmap_df = pd.DataFrame(
            data=heatmap_data,
            index=zema_lengths,
            columns=offsets
        )

        fig_heatmap = heatmap_df.vbt.heatmap(
            x_title='Offset',
            y_title='ZEMA Length',
            title=f'Heatmap pour {crypto} - {timeframe}'
        )
        fig_heatmap.write_image(f"{RESULTS_DIR}/{crypto}_{timeframe}_heatmap.png")

# ====== Détermination du meilleur timeframe ======

performance_summary = {}

for timeframe in TIMEFRAMES:
    total_returns = []
    for crypto in CRYPTOS:
        if crypto not in results[timeframe]:
            continue  # Passer si aucune donnée disponible

        portfolio = results[timeframe][crypto]['portfolio']
        total_returns.append(portfolio.total_return())
    if total_returns:
        average_return = np.mean(total_returns)
        performance_summary[timeframe] = average_return

# Création du graphique comparatif des timeframes
performance_df = pd.Series(performance_summary)
fig_comparison = performance_df.vbt.plot(
    kind='bar',
    title='Performance moyenne par timeframe'
)
fig_comparison.write_image(f"{RESULTS_DIR}/timeframe_comparison.png")

print("Analyse terminée. Les résultats sont disponibles dans le répertoire 'results'.")
