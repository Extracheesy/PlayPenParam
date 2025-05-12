import pandas as pd
import numpy as np
import glob
from ohlcv_binance import CryptoBacktest
import os
from datetime import datetime
import os
import glob
import shutil

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

try:
    import pandas_ta as ta
except ImportError:
    # If pandas_ta is not installed, the user will need to install it.
    raise ImportError("The pandas_ta library is required for this code. Install via pip (pip install pandas_ta).")

# Trend detection functions using various indicator combinations.
# Each function returns a pandas Series of trend signals (1 for uptrend, -1 for downtrend, 0 for no trend)
# corresponding to each row of the input DataFrame.
# The DataFrame `df` is expected to have columns: 'open', 'high', 'low', 'close', 'volume'.

def trend_kama(df, length=10, fast=2, slow=30):
    """
    Determine trend using the Kaufman Adaptive Moving Average (KAMA).
    - If price is above KAMA and KAMA is rising, trend = 1 (uptrend).
    - If price is below KAMA and KAMA is falling, trend = -1 (downtrend).
    - Otherwise, trend = 0 (no clear trend).
    Parameters:
        length: period for efficiency ratio calculation (e.g., 10).
        fast: fast EMA period for KAMA (e.g., 2).
        slow: slow EMA period for KAMA (e.g., 30).
    """
    close = df['close']
    # Compute KAMA (Kaufman's Adaptive Moving Average)
    kama_series = ta.kama(close=close, length=length, fast=fast, slow=slow)
    # In case pandas_ta returns a DataFrame (some versions might), ensure we have a Series:
    if isinstance(kama_series, pd.DataFrame):
        # pandas_ta.kama might return a DataFrame with one column named like "KAMA_{length}_{fast}_{slow}"
        kama_series = kama_series.iloc[:, 0]
    # Calculate the change in KAMA to determine its slope
    kama_diff = kama_series.diff()
    # Initialize trend signal series with 0 (no trend by default)
    trend = pd.Series(0, index=df.index, dtype=int)
    # Uptrend condition: price above KAMA and KAMA rising
    up_cond = (close > kama_series) & (kama_diff > 0)
    # Downtrend condition: price below KAMA and KAMA falling
    down_cond = (close < kama_series) & (kama_diff < 0)
    trend[up_cond] = 1
    trend[down_cond] = -1
    return trend

def trend_adx_tsi(df, adx_length=14, adx_threshold=20, tsi_fast=13, tsi_slow=25):
    """
    Determine trend using ADX (Average Directional Index) combined with TSI and +DI/-DI.
    - Uses ADX to confirm a trending market (ADX above threshold indicates a trend).
    - If ADX >= threshold and +DI > -DI and TSI > 0: trend = 1 (uptrend).
    - If ADX >= threshold and -DI > +DI and TSI < 0: trend = -1 (downtrend).
    - Otherwise (ADX below threshold or conflicting indicators): trend = 0.
    Parameters:
        adx_length: period for ADX and DI calculation (e.g., 14).
        adx_threshold: minimum ADX value to consider a trend (e.g., 20 or 25).
        tsi_fast: fast period for True Strength Index (e.g., 13).
        tsi_slow: slow period for True Strength Index (e.g., 25).
    """
    high = df['high']
    low = df['low']
    close = df['close']
    # Compute ADX, +DI, -DI
    adx_df = ta.adx(high=high, low=low, close=close, length=adx_length)
    # pandas_ta.adx returns a DataFrame with ADX, +DI, -DI. Column names typically: 'ADX_{adx_length}', 'DMP_{adx_length}', 'DMN_{adx_length}'
    # Extract the series:
    adx_val = adx_df[f"ADX_{adx_length}"] if f"ADX_{adx_length}" in adx_df.columns else adx_df.iloc[:, 0]
    plus_di = adx_df[f"DMP_{adx_length}"] if f"DMP_{adx_length}" in adx_df.columns else adx_df.iloc[:, 1]
    minus_di = adx_df[f"DMN_{adx_length}"] if f"DMN_{adx_length}" in adx_df.columns else adx_df.iloc[:, 2]
    # Compute TSI (True Strength Index)
    tsi_series = ta.tsi(close=close, fast=tsi_fast, slow=tsi_slow)
    if isinstance(tsi_series, pd.DataFrame):
        tsi_series = tsi_series.iloc[:, 0]
    # Initialize trend signals
    trend = pd.Series(0, index=df.index, dtype=int)
    # Trend conditions (only consider trend if ADX is above threshold)
    up_cond = (adx_val >= adx_threshold) & (plus_di > minus_di) & (tsi_series > 0)
    down_cond = (adx_val >= adx_threshold) & (minus_di > plus_di) & (tsi_series < 0)
    trend[up_cond] = 1
    trend[down_cond] = -1
    return trend

def trend_aroon_tsi(df, aroon_length=25, aroon_threshold=70, tsi_fast=13, tsi_slow=25):
    """
    Determine trend using Aroon indicator combined with TSI and +DI/-DI.
    - Uses Aroon Up/Down to gauge trend direction (values range 0-100).
    - If Aroon Up > threshold and Aroon Down < (100 - threshold), plus TSI > 0 and +DI > -DI: trend = 1.
    - If Aroon Down > threshold and Aroon Up < (100 - threshold), plus TSI < 0 and -DI > +DI: trend = -1.
    - Otherwise: trend = 0.
    Parameters:
        aroon_length: period for Aroon calculation (e.g., 25).
        aroon_threshold: threshold for strong Aroon signal (e.g., 70).
        tsi_fast, tsi_slow: periods for TSI (like in trend_adx_tsi).
    """
    high = df['high']
    low = df['low']
    close = df['close']
    # Compute Aroon Up and Down
    aroon_df = ta.aroon(high=high, low=low, length=aroon_length)
    # pandas_ta.aroon returns a DataFrame with Aroon Up and Down; names like 'AROONU_{length}', 'AROOND_{length}'
    aroon_up = aroon_df[f"AROONU_{aroon_length}"] if f"AROONU_{aroon_length}" in aroon_df.columns else aroon_df.iloc[:, 0]
    aroon_down = aroon_df[f"AROOND_{aroon_length}"] if f"AROOND_{aroon_length}" in aroon_df.columns else aroon_df.iloc[:, 1]
    # Use +DI and -DI for additional confirmation (from ADX calculation with default length 14)
    adx_df = ta.adx(high=high, low=low, close=close, length=14)
    plus_di = adx_df[f"DMP_14"] if f"DMP_14" in adx_df.columns else adx_df.iloc[:, 1]
    minus_di = adx_df[f"DMN_14"] if f"DMN_14" in adx_df.columns else adx_df.iloc[:, 2]
    # Compute TSI
    tsi_series = ta.tsi(close=close, fast=tsi_fast, slow=tsi_slow)
    if isinstance(tsi_series, pd.DataFrame):
        tsi_series = tsi_series.iloc[:, 0]
    # Initialize trend signals
    trend = pd.Series(0, index=df.index, dtype=int)
    # Uptrend if Aroon indicates recent high, and momentum indicators agree
    up_cond = (aroon_up >= aroon_threshold) & (aroon_down <= (100 - aroon_threshold)) & (plus_di > minus_di) & (tsi_series > 0)
    # Downtrend if Aroon indicates recent low, and momentum indicators agree
    down_cond = (aroon_down >= aroon_threshold) & (aroon_up <= (100 - aroon_threshold)) & (minus_di > plus_di) & (tsi_series < 0)
    trend[up_cond] = 1
    trend[down_cond] = -1
    return trend

def trend_donchian_obv(df, donchian_period=20, obv_period=10):
    """
    Determine trend using Donchian channel breakout confirmed by On-Balance Volume (OBV).
    - If price breaks above the Donchian channel upper band and OBV is rising, trend = 1.
    - If price breaks below the Donchian channel lower band and OBV is falling, trend = -1.
    - Otherwise, trend = 0.
    Parameters:
        donchian_period: lookback period for Donchian channel (e.g., 20).
        obv_period: period to measure OBV trend (e.g., 10). OBV trend is determined by comparing current OBV with its value obv_period bars ago.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    # Compute Donchian channel high and low (rolling max/min over the last `donchian_period` bars)
    dc_high = high.rolling(window=donchian_period, min_periods=donchian_period).max()
    dc_low = low.rolling(window=donchian_period, min_periods=donchian_period).min()
    # Use previous period's Donchian levels for breakout comparison
    dc_high_prev = dc_high.shift(1)
    dc_low_prev = dc_low.shift(1)
    # Compute On-Balance Volume (OBV)
    price_diff = close.diff()
    obv = pd.Series(0, index=df.index)
    obv[price_diff > 0] = volume[price_diff > 0]
    obv[price_diff < 0] = -volume[price_diff < 0]
    obv = obv.cumsum()
    # OBV momentum: difference between current OBV and OBV `obv_period` bars ago
    obv_diff = obv - obv.shift(obv_period)
    # Initialize trend signals
    trend = pd.Series(0, index=df.index, dtype=int)
    # Uptrend if price makes a new high above previous Donchian upper and OBV is higher than `obv_period` bars ago
    up_cond = (close > dc_high_prev) & (obv_diff > 0)
    # Downtrend if price makes a new low below previous Donchian lower and OBV is lower than `obv_period` bars ago
    down_cond = (close < dc_low_prev) & (obv_diff < 0)
    trend[up_cond] = 1
    trend[down_cond] = -1
    return trend

def trend_double_supertrend(df, fast_period=10, fast_multiplier=3, slow_period=30, slow_multiplier=5):
    """
    Determine trend using two SuperTrend indicators (fast and slow).
    - Compute a fast SuperTrend (shorter period or lower multiplier) and a slow SuperTrend (longer period or higher multiplier).
    - If both SuperTrends indicate an uptrend, trend = 1.
    - If both indicate a downtrend, trend = -1.
    - If they disagree or no clear trend, trend = 0.
    Parameters:
        fast_period, fast_multiplier: parameters for the fast SuperTrend.
        slow_period, slow_multiplier: parameters for the slow SuperTrend.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    # Compute fast and slow SuperTrend using pandas_ta
    st_fast = ta.supertrend(high=high, low=low, close=close, length=fast_period, multiplier=fast_multiplier)
    st_slow = ta.supertrend(high=high, low=low, close=close, length=slow_period, multiplier=slow_multiplier)
    # pandas_ta.supertrend returns DataFrames with columns: 'SUPERT_{period}_{mult}', 'SUPERTd_{period}_{mult}', 'SUPERTl_{period}_{mult}', 'SUPERTs_{period}_{mult}'
    # Identify the direction columns (SUPERTd) which give 1 or -1 for trend direction.
    # Prepare strings for column lookup (account for possible ".0" in multiplier formatting)
    fast_mult_str = f"{float(fast_multiplier)}"
    slow_mult_str = f"{float(slow_multiplier)}"
    if fast_mult_str.endswith('.0'):
        fast_mult_str = fast_mult_str[:-2]
    if slow_mult_str.endswith('.0'):
        slow_mult_str = slow_mult_str[:-2]
    fast_dir_col = f"SUPERTd_{fast_period}_{fast_mult_str}"
    slow_dir_col = f"SUPERTd_{slow_period}_{slow_mult_str}"
    fast_dir = st_fast[fast_dir_col] if fast_dir_col in st_fast.columns else st_fast.filter(like='SUPERTd').iloc[:, 0]
    slow_dir = st_slow[slow_dir_col] if slow_dir_col in st_slow.columns else st_slow.filter(like='SUPERTd').iloc[:, 0]
    # Fill NaNs (initial periods) with 0 (no trend info yet)
    fast_dir = fast_dir.fillna(0)
    slow_dir = slow_dir.fillna(0)
    # Initialize trend signals
    trend = pd.Series(0, index=df.index, dtype=int)
    # Uptrend if both SuperTrends show uptrend (direction == 1)
    up_cond = (fast_dir == 1) & (slow_dir == 1)
    # Downtrend if both SuperTrends show downtrend (direction == -1)
    down_cond = (fast_dir == -1) & (slow_dir == -1)
    trend[up_cond] = 1
    trend[down_cond] = -1
    return trend

def trend_hma_wavetrend(df, hma_length=50, wt_channel_length=10, wt_average_length=21):
    """
    Determine trend using double confirmation with Hull Moving Average (HMA) and WaveTrend oscillator.
    - If HMA is sloping upward and WaveTrend oscillator is positive, trend = 1.
    - If HMA is sloping downward and WaveTrend oscillator is negative, trend = -1.
    - Otherwise, trend = 0.
    Parameters:
        hma_length: period for Hull Moving Average (e.g., 50).
        wt_channel_length: WaveTrend oscillator channel length (e.g., 10).
        wt_average_length: WaveTrend oscillator average length (e.g., 21).
    """
    close = df['close']
    high = df['high']
    low = df['low']
    # Compute Hull Moving Average (HMA) of closing price
    hma_series = ta.hma(close=close, length=hma_length)
    if isinstance(hma_series, pd.DataFrame):
        hma_series = hma_series.iloc[:, 0]
    hma_diff = hma_series.diff()
    # Compute WaveTrend oscillator (LazyBear's WaveTrend formula)
    tp = (high + low + close) / 3.0  # typical price
    # EMA of typical price
    esa = tp.ewm(span=wt_channel_length, adjust=False).mean()
    # EMA of absolute price deviation
    dep = (tp - esa).abs().ewm(span=wt_channel_length, adjust=False).mean()
    # Channel Index (CI)
    ci = (tp - esa) / (0.015 * dep)
    # WaveTrend values
    wt1 = ci.ewm(span=wt_average_length, adjust=False).mean()
    wt2 = wt1.ewm(span=4, adjust=False).mean()  # signal line (not used directly for trend decision here)
    # Initialize trend signals
    trend = pd.Series(0, index=df.index, dtype=int)
    up_cond = (hma_diff > 0) & (wt1 > 0)
    down_cond = (hma_diff < 0) & (wt1 < 0)
    trend[up_cond] = 1
    trend[down_cond] = -1
    return trend

def get_trend_signal(df, method="kama", **kwargs):
    """
    Get trend signal series for the given DataFrame using the specified method.
    method: one of "kama", "adx_tsi", "aroon_tsi", "donchian_obv", "double_supertrend", "hma_wavetrend".
    Additional keyword arguments are passed to the respective method function for parameter tuning.
    """
    method = method.lower()
    if method == "kama":
        return trend_kama(df, **kwargs)
    elif method == "adx_tsi":
        return trend_adx_tsi(df, **kwargs)
    elif method == "aroon_tsi":
        return trend_aroon_tsi(df, **kwargs)
    elif method == "donchian_obv":
        return trend_donchian_obv(df, **kwargs)
    elif method == "double_supertrend":
        return trend_double_supertrend(df, **kwargs)
    elif method == "hma_wavetrend":
        return trend_hma_wavetrend(df, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}.")

# Example usage (commented out):
# df = pd.read_csv('your_data.csv')  # DataFrame with columns ['open','high','low','close','volume']
# trend_signals = trend_kama(df)  # using KAMA method
# trend_signals = trend_adx_tsi(df, adx_length=14, adx_threshold=25)  # using ADX + TSI method
# trend_signals = get_trend_signal(df, method="double_supertrend", fast_period=10, fast_multiplier=3, slow_period=30, slow_multiplier=5)

def label_effective_trend(df, threshold=0.001):
    """
    Label the 'true' next-bar trend:
      - compute next-bar return: (close_t+1 / close_t) - 1
      - if >  threshold →  1
      - if < -threshold → -1
      - else              →  0
    Returns a Series aligned with df.index (last row will be NaN → drop later).
    """
    ret = df['close'].pct_change().shift(-1)
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[ret > threshold] = 1
    labels[ret < -threshold] = -1
    # last bar can’t be labeled (no future), leave as NaN
    labels.iloc[-1] = np.nan
    return labels


def evaluate_methods_on_df(df, methods, threshold=0.001):
    """
    For each method in methods (dict of name→(method_key, kwargs)),
    compute detection_rate and success_rate vs the effective trend.
    Returns a DataFrame with metrics.
    """
    # 1) label truth
    truth = label_effective_trend(df, threshold).dropna()
    n_bars = len(truth)

    records = []
    for name, (method_key, kwargs) in methods.items():
        sig = get_trend_signal(df, method=method_key, **kwargs)

        df[method_key] = sig

        sig = sig.reindex(truth.index)  # align & drop extra last point
        # only keep bars where method signalled a trend
        mask = sig != 0
        n_signals = mask.sum()
        if n_signals == 0:
            detection_rate = 0.0
            success_rate = np.nan
        else:
            detection_rate = n_signals / n_bars
            # of those, how many match the true label?
            correct = (sig[mask] == truth[mask]).sum()
            success_rate = correct / n_signals

        records.append({
            'method': name,
            'n_bars': n_bars,
            'n_signals': n_signals,
            'detection_rate': detection_rate,
            'n_correct': correct if n_signals > 0 else 0,
            'success_rate': success_rate
        })

    return pd.DataFrame(records), df

def duplicate_ohlcv_files(data_dir: str, start_date: str, symbol: str) -> None:
    """
    In `data_dir`, find all files like
      {symbol}_data_{freq}_{start_date}_*.csv
    for the frequencies 1m, 5m, 15m, 30m, 1h, then copy each to:
      ohlcv_{freq}.csv
    """
    # the timeframes you care about
    freqs = ['1m', '5m', '15m', '30m', '1h']

    for freq in freqs:
        # build a glob pattern to catch the file with your given start_date
        pattern = os.path.join(data_dir, f"{symbol}_data_{freq}_{start_date}_*.csv")
        matches = glob.glob(pattern)

        if not matches:
            print(f"[!] no file found for {freq} (pattern: {pattern})")
            continue

        # pick the first match (if there’re multiples, you might want to tweak this)
        src = matches[0]
        dst = os.path.join(data_dir, f"ohlcv_{freq}.csv")

        # copy (overwrites existing ohlcv_*.csv if present)
        shutil.copy(src, dst)
        print(f"✔ copied {os.path.basename(src)} → {os.path.basename(dst)}")

if __name__ == "__main__":
    # your existing base path
    path = r'C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\tend_analysis'
    os.makedirs(path, exist_ok=True)  # ensure the base exists
    os.chdir(path)  # cd into it
    print(f"Now in directory: {os.getcwd()}")

    # name of your sub-folder
    directory_result = "trend_results"

    # join and create it
    result_path = os.path.join(path, directory_result)
    os.makedirs(result_path, exist_ok=True)  # makes trnd_results (and parents) if needed

    # optionally cd into the new results dir
    os.chdir(result_path)
    print(f"Now in directory: {os.getcwd()}")

    lst_start_date = [
        '2025-04-01T00:00:00Z',
        '2024-01-01T00:00:00Z',
        '2024-12-17T00:00:00Z',
        '2025-01-01T00:00:00Z',
        '2024-03-13T00:00:00Z',
        '2025-03-01T00:00:00Z'

    ]

    dct_start_date = {
        1: "2024-01-01T00:00:00Z",
        2: "2024-12-17T00:00:00Z",
        3: "2025-01-01T00:00:00Z",
        4: "2024-03-13T00:00:00Z",
        5: "2025-03-01T00:00:00Z",
        6: "2025-04-01T00:00:00Z"
    }

    lst_start_date = [
        # '2025-04-01T00:00:00Z'
        "2024-01-01T00:00:00Z"
    ]

    dct_start_date = {
        1: "2024-01-01T00:00:00Z"
    }

    for start_date in lst_start_date:
        symbol = "BTCUSDT"

        # start_date = dct_start_date[6]

        data_dir = os.path.join(result_path, symbol + "_" + start_date.rstrip('Z')[:16].replace('T','-').replace(':','-') + "_data_ohlcv")
        os.makedirs(data_dir, exist_ok=True)

        if False:
            end_date = None
        else:
            end_date = datetime.utcnow()
            print('end date: ', end_date)

        str_end_date = "2025-04-30 10:00:43.933201"
        end_date = datetime.strptime(str_end_date, "%Y-%m-%d %H:%M:%S.%f")
        end_date = end_date.replace(second=0, microsecond=0)

        if True:
            crypto_bkt = CryptoBacktest("BTCUSDT", start_date, end_date=end_date, lst_timeframe=["1m", "5m", "15m", "30m", "1h"], save_dir=data_dir)


            nice_start_date = start_date.rstrip('Z')[:16].replace('T','-').replace(':','-')
            duplicate_ohlcv_files(data_dir, nice_start_date, symbol)

        # 1. Define your CSV files by timeframe
        files = {
            '1m': 'ohlcv_1m.csv',
            '5m': 'ohlcv_5m.csv',
            '15m': 'ohlcv_15m.csv',
            '30m': 'ohlcv_30m.csv',
            '1h': 'ohlcv_1h.csv',
        }

        # 2. Define the methods you want to test
        methods = {
            "KAMA": ("kama", dict(length=10, fast=2, slow=30)),
            "ADX+TSI": ("adx_tsi", dict(adx_length=14, adx_threshold=20, tsi_fast=13, tsi_slow=25)),
            "Aroon+TSI": ("aroon_tsi", dict(aroon_length=25, aroon_threshold=70, tsi_fast=13, tsi_slow=25)),
            "Donchian+OBV": ("donchian_obv", dict(donchian_period=20, obv_period=10)),
            "DoubleST": ("double_supertrend", dict(fast_period=10, fast_multiplier=3, slow_period=30, slow_multiplier=5)),
            "HMA+WT": ("hma_wavetrend", dict(hma_length=50, wt_channel_length=10, wt_average_length=21)),
        }

        # Threshold for flat band (±0.1% here)
        THRESHOLD = 0.001

        file_path_1m = os.path.join(data_dir, files['1m'])
        df_1m = pd.read_csv(file_path_1m, parse_dates=True, index_col=0)
        index_1m = df_1m.index

        all_results = []
        all_dfs = []

        for tf, fname in files.items():
            file_path = os.path.join(data_dir, fname)
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)

            # pick cols & drop NaNs
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

            # run your eval
            results, df = evaluate_methods_on_df(df, methods, threshold=THRESHOLD)
            results.insert(0, 'timeframe', tf)
            all_results.append(results)

            # rename
            df.columns = [f"{c}_{tf}" for c in df.columns]

            # reindex to 1m and forward-fill
            df = df.reindex(index_1m).ffill()

            all_dfs.append(df)

        # finally concat on the common 1m index
        combined_df = pd.concat(all_dfs, axis=1, join='inner')

        final = pd.concat(all_results, ignore_index=True)

        # Display
        pd.set_option('display.float_format', '{:.2%}'.format)
        print(final[['timeframe', 'method', 'detection_rate', 'success_rate']])

        final.to_csv(os.path.join(data_dir, symbol + "_" + nice_start_date + "_results.csv"))
        final.to_csv(os.path.join(data_dir, symbol + "_" + nice_start_date + "_results_excel.csv"), sep=";")

        metric_order = [
            'open', 'high', 'low', 'close', 'volume',
            'kama', 'adx_tsi', 'aroon_tsi',
            'donchian_obv', 'double_supertrend', 'hma_wavetrend'
        ]
        timeframe_order = ['1m', '5m', '15m', '30m', '1h']


        # custom key fn
        def sort_key(col):
            metric, tf = col.rsplit('_', 1)
            return (metric_order.index(metric), timeframe_order.index(tf))


        # compute the new column ordering
        new_cols = sorted(combined_df.columns, key=sort_key)

        # reindex your df
        combined_df = combined_df[new_cols]

        cols_to_sum = [
            'kama_1m', 'kama_5m', 'kama_15m', 'kama_30m', 'kama_1h',
            'adx_tsi_1m', 'adx_tsi_5m', 'adx_tsi_15m', 'adx_tsi_30m', 'adx_tsi_1h',
            'aroon_tsi_1m', 'aroon_tsi_5m', 'aroon_tsi_15m', 'aroon_tsi_30m', 'aroon_tsi_1h',
            'donchian_obv_1m', 'donchian_obv_5m', 'donchian_obv_15m', 'donchian_obv_30m', 'donchian_obv_1h',
            'double_supertrend_1m', 'double_supertrend_5m', 'double_supertrend_15m', 'double_supertrend_30m',
            'double_supertrend_1h',
            'hma_wavetrend_1m', 'hma_wavetrend_5m', 'hma_wavetrend_15m', 'hma_wavetrend_30m', 'hma_wavetrend_1h'
        ]

        combined_df['sum'] = combined_df[cols_to_sum] \
            .apply(pd.to_numeric, errors='coerce') \
            .sum(axis=1)

        # --- compute smoothers ---
        span = 10
        window = 10

        # exponential-weighted moving average
        combined_df['sum_smooth_ewm'] = combined_df['sum'].ewm(span=span, adjust=False).mean()

        # simple moving average (centered)
        combined_df['sum_smooth_sma'] = combined_df['sum'] \
            .rolling(window=window, min_periods=1, center=True).mean()

        # absolute profit
        combined_df['profit'] = combined_df['close_1m'] - combined_df['close_1m'].shift(1)
        # percentage profit
        combined_df['profit_pct'] = combined_df['close_1m'].pct_change() * 100
        combined_df['profit_flag'] = (combined_df['profit_pct'] > 0.05).astype(int)

        combined_df.to_csv(os.path.join(data_dir, symbol + "_" + nice_start_date + "_combined.csv"))
        # combined_df.to_csv(os.path.join(data_dir, symbol + "_" + nice_start_date + "_combined_excel.csv"), sep=";")

        out_fn = f"{symbol}_{nice_start_date}_combined_excel.csv"
        combined_df.to_csv(
            os.path.join(data_dir, out_fn),
            sep=';',  # semicolon as field separator
            decimal=',',  # comma for the decimal point
            index=True
        )

        GRAPH_SUM = False
        EWM = False
        SMA = True
        if GRAPH_SUM:
            fig, ax1 = plt.subplots()

            # Plot close_1m as a line on the left axis
            ax1.plot(combined_df.index, combined_df['close_1m'], color='tab:blue', label='close_1m')
            ax1.set_ylabel('close_1m')
            ax1.set_xlabel('Time')

            # Create a twin axis and plot sum as bars
            ax2 = ax1.twinx()
            ax2.bar(
                combined_df.index,
                combined_df['sum'],
                alpha=0.3,
                width=0.0007,  # tweak this so bars aren't too wide; it's in days (0.0007≈1min)
                label='sum'
            )
            ax2.set_ylabel('sum')

            # Combine legends
            lines, labels = ax1.get_lines() + ax2.containers, [l.get_label() for l in ax1.get_lines()] + [c.get_label() for
                                                                                                          c in
                                                                                                          ax2.containers]
            fig.legend(lines, labels, loc='upper left')

            plt.title('close_1m (line) vs. sum (bars)')
            plt.tight_layout()
            plt.show()
        elif EWM:
            # --- plotting ---
            fig, ax1 = plt.subplots()

            # line for close_1m
            ax1.plot(combined_df.index, combined_df['close_1m'], label='close_1m')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('close_1m')

            # twin axis for the bar chart
            ax2 = ax1.twinx()
            ax2.bar(
                combined_df.index,
                combined_df['sum_smooth_ewm'],
                width=0.0007,  # ~1 minute in days
                alpha=0.6,
                label='sum_smooth_ewm'
            )
            ax2.set_ylabel('sum_smooth_ewm')

            # combined legend
            lines = ax1.get_lines() + ax2.containers
            labels = [l.get_label() for l in ax1.get_lines()] + [c.get_label() for c in ax2.containers]
            fig.legend(lines, labels, loc='upper left')

            plt.title('close_1m (line) vs. Smoothed Sum EWMA (bars)')
            plt.tight_layout()
            plt.show()
        elif SMA:
            # --- build a color list: green for ≥0, red for <0 ---
            colors = [
                'green' if v >= 0 else 'red'
                for v in combined_df['sum_smooth_sma']
            ]

            # --- plotting ---
            fig, ax1 = plt.subplots()

            # line for close_1m
            ax1.plot(combined_df.index, combined_df['close_1m'], label='close_1m')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('close_1m')

            # twin axis for SMA bars colored by sign
            ax2 = ax1.twinx()
            ax2.bar(
                combined_df.index,
                combined_df['sum_smooth_sma'],
                width=0.0007,  # ~1 minute in days
                color=colors,
                alpha=0.8,
                label='sum_smooth_sma'
            )
            ax2.set_ylabel('sum_smooth_sma')

            # combined legend
            lines = ax1.get_lines() + ax2.containers
            labels = [l.get_label() for l in ax1.get_lines()] + [c.get_label() for c in ax2.containers]
            fig.legend(lines, labels, loc='upper left')

            plt.title('close_1m (line) vs. Smoothed Sum SMA (green=positive, red=negative)')
            plt.tight_layout()
            plt.show()