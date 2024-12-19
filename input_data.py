import pandas as pd
from datetime import timedelta
from datetime import date

symbols = [
    'PEPEUSDT',
    'BTCUSDT',
    'SOLUSDT',
    'ETHUSDT'
]

"""
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
"""

multi_treading = False
my_trading_sim = False
init_param = False

start_date = '2024-01-01'
# start_date = '2024-03-01'
# end_date = '2024-12-14'
end_date = date.today().strftime("%Y-%m-%d")
end_date = '2024-12-18'

# tf = '1h'
# tf = '15m'
# tf = ['5m']
tf = ['1m', '5m', "15m"]

# lst_trade_type = ['long', 'short', 'both']
lst_trade_type = ['long']
# lst_trade_type = ['short']

# lst_ma_type = ["ZLEMA", "ZLMA", "TEMA", "DEMA", "ALMA", "KAMA", "HMA", "COMBINED"]
# lst_ma_type = ["ZLEMA", "ZLMA", "TEMA", "DEMA", "ALMA", "HMA", "COMBINED"]
lst_ma_type = ["COMBINED"]
lst_combined = ['ALMA', 'ZLEMA', "DEMA", "HMA", "TEMA"]

if init_param:
    lst_low_offset = [0.964]
    lst_high_offset = [1.004]
    lst_zema_len_buy = [51]
    lst_zema_len_sell = [72]
    lst_ssl_atr_period = [10]
else:
    lst_low_offset = [round(x, 3) for x in [0.90 + i*0.01 for i in range(11)]]  # 0.90 to 1.00 in steps of 0.01
    lst_high_offset = [round(x, 3) for x in [1.000 + i*0.002 for i in range(6)]]  # 1.000 to 1.010 in steps of 0.002
    lst_zema_len_buy = list(range(30, 71, 5))  # 30,35,40,45,50,55,60,65,70
    lst_zema_len_sell = list(range(50, 91, 5)) # 50,55,60,65,70,75,80,85,90
    lst_ssl_atr_period = [5, 10, 15, 20]

ichimoku_params = {
    'conversion_line_period': 20,
    'base_line_periods': 60,
    'lagging_span': 120,
    'displacement': 30
}

# Maximum lookback period based on the longest indicator
max_lookback_days = max(
    ichimoku_params['conversion_line_period'],
    ichimoku_params['base_line_periods'],
    ichimoku_params['lagging_span'],
)

extended_start_date = (pd.Timestamp(start_date) - timedelta(days=max_lookback_days)).strftime('%Y-%m-%d')