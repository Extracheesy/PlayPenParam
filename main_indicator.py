import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import ta
import os
from datetime import datetime

# Initialisation de l'API Binance via ccxt
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
})

# Paramètres
symbol = 'PEPE/USDT'
timeframe = '5m'  # Modifier par '5m', '15m', ou '1h'
start_date = '2024-03-01'
end_date = '2024-12-14'

# Nom du fichier pour stocker les données
data_directory = './data'
os.makedirs(data_directory, exist_ok=True)
data_file = f'{data_directory}/{symbol.replace("/", "_")}_{timeframe}_{start_date}_{end_date}.csv'


# Fonction pour récupérer ou charger les données
def fetch_or_load_data():
    if os.path.exists(data_file):
        print(f"Chargement des données depuis {data_file}...")
        df = pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')
    else:
        print("Téléchargement des données depuis Binance...")
        since = exchange.parse8601(f'{start_date}T00:00:00Z')
        until = exchange.parse8601(f'{end_date}T23:59:59Z')

        ohlcv = []
        while since < until:
            data = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if len(data) == 0:
                break
            ohlcv += data
            since = data[-1][0] + 1

        # Conversion en DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Sauvegarde dans un fichier CSV
        df.to_csv(data_file)
        print(f"Données sauvegardées dans {data_file}.")

    return df


# Récupération ou chargement des données
df = fetch_or_load_data()

# Supprimer les valeurs nulles avant le calcul des indicateurs
df = df.dropna()

# Calcul des indicateurs techniques
df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

macd_indicator = ta.trend.MACD(df['close'])
df['MACD'] = macd_indicator.macd()  # Ligne MACD
df['Signal'] = macd_indicator.macd_signal()  # Ligne de Signal

df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
bollinger = ta.volatility.BollingerBands(df['close'])
df['Bollinger High'] = bollinger.bollinger_hband()
df['Bollinger Low'] = bollinger.bollinger_lband()
df['Force Index'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()
df['Momentum'] = df['close'].diff(10)  # Différence sur 10 périodes

# Nouveaux indicateurs
df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

short_window = 5
long_window = 20
df['Volume MA Short'] = df['volume'].rolling(window=short_window).mean()
df['Volume MA Long'] = df['volume'].rolling(window=long_window).mean()
df['Volume Oscillator'] = ((df['Volume MA Short'] - df['Volume MA Long']) / df['Volume MA Long']) * 100

df['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()


# Fonction pour afficher les graphiques
def plot_indicators(df):
    fig, axes = plt.subplots(9, 1, figsize=(12, 30), sharex=True)

    axes[0].plot(df.index, df['close'], label='Close Price', color='blue')
    axes[0].set_title('Prix de Clôture et Bandes de Bollinger')
    axes[0].fill_between(df.index, df['Bollinger High'], df['Bollinger Low'], color='gray', alpha=0.3)

    axes[1].plot(df.index, df['RSI'], label='RSI', color='orange')
    axes[1].axhline(70, color='red', linestyle='--', linewidth=0.5)
    axes[1].axhline(30, color='green', linestyle='--', linewidth=0.5)
    axes[1].set_title('RSI')

    axes[2].plot(df.index, df['MACD'], label='MACD', color='purple')
    axes[2].plot(df.index, df['Signal'], label='Signal', color='red')
    axes[2].set_title('MACD')

    axes[3].plot(df.index, df['ADX'], label='ADX', color='brown')
    axes[3].axhline(25, color='blue', linestyle='--', linewidth=0.5)
    axes[3].set_title('ADX')

    axes[4].plot(df.index, df['Force Index'], label='Force Index', color='teal')
    axes[4].set_title('Force Index')

    axes[5].plot(df.index, df['Momentum'], label='Momentum', color='black')
    axes[5].set_title('Momentum')

    axes[6].plot(df.index, df['OBV'], label='On-Balance Volume (OBV)', color='magenta')
    axes[6].set_title('On-Balance Volume (OBV)')

    axes[7].plot(df.index, df['Volume Oscillator'], label='Volume Oscillator', color='cyan')
    axes[7].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axes[7].set_title('Volume Oscillator')

    axes[8].plot(df.index, df['MFI'], label='Money Flow Index (MFI)', color='darkgreen')
    axes[8].axhline(80, color='red', linestyle='--', linewidth=0.5)
    axes[8].axhline(20, color='blue', linestyle='--', linewidth=0.5)
    axes[8].set_title('Money Flow Index (MFI)')

    for ax in axes:
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()


plot_indicators(df)
