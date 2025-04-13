import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
from datetime import date
import datetime
from . import utils

import concurrent.futures

import requests

def _get_ohlcv_bitget_v1(symbol, timeframe="1h", limit=100):
    """
    Fetch the latest 'limit' OHLCV candles for a given futures symbol, product type, and timeframe
    from Bitget's Futures Market API.

    Args:
        symbol (str): Trading symbol (e.g., "BTCUSDT").
        timeframe (str): Candle timeframe (e.g., "1d", "1h", etc.). Default is "1d".
        limit (int): Number of candles to fetch.

    Returns:
        list: A list of candles (each typically a list or dict depending on API response),
              or None if an error occurred.
    """

    # if we receive BTC/USDT, we convert it into BTCUSDT
    symbol = symbol.replace("/", "")

    base_url = "https://api.bitget.com"
    endpoint = "/api/v2/mix/market/candles"
    url = base_url + endpoint

    # Map common timeframe abbreviations to the API-required format.
    timeframe_mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "4h": "4H",
        "6h": "6H",
        "12h": "12H",
        "1d": "1D",
        "1w": "1W",
        "1M": "1Mutc"
    }
    period = timeframe_mapping.get(timeframe, timeframe)

    params = {
        "symbol": symbol,
        "granularity": period,
        "limit": str(limit),
        "productType": "USDT-FUTURES"  # added productType parameter for futures
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError for unsuccessful responses
        data = response.json()
        if data.get("code") != "00000":
            raise Exception(f"API error: {data.get('msg', 'Unknown error')} (code: {data.get('code')})")

        candle_data = data.get("data", [])
        # Assuming each candle is in the format:
        # [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(candle_data, columns=["timestamp", "open", "high", "low", "close", "volume", "volume_2"])
        df = df.drop(columns=['volume_2'])
        df = df.rename(columns={0: 'timestamp', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].astype(float)

        if df.empty:
            return "no data"
        df = df.set_index(df['timestamp'])
        df.index = df.index.str.replace(r'0{3}$', '', regex=True)
        df.index = pd.to_datetime(df.index.astype(int), unit='s', utc=True, errors='coerce')

        return df
    except Exception as e:
        print("Error fetching futures OHLCV data:", e)
        return None

def _get_ohlcv_bitget_v2(symbol, timeframe="1h", limit=200):
        # symbol = utils.convert_symbol_to_bitget(symbol)
        symbol = symbol.replace("/", "")

        base_url = "https://api.bitget.com"
        endpoint = "/api/v2/mix/market/candles"

        url = base_url + endpoint

        granularity_mapping = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "4h": "4H",
            "6h": "6H",
            "12h": "12H",
            "1d": "1D",
            "1w": "1W"
        }

        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": granularity_mapping[timeframe],
            "limit": str(limit)
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises an HTTPError for unsuccessful responses
            candle_data = response.json()

            # Assuming each candle is in the format:
            # [timestamp, open, high, low, close, volume]
            df = pd.DataFrame(candle_data["data"], columns=["timestamp", "open", "high", "low", "close", "volume", "volume_2"])
            df = df.drop(columns=['volume_2'])
            df = df.rename(columns={0: 'timestamp', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
            cols = ["open", "high", "low", "close", "volume"]
            df[cols] = df[cols].astype(float)
            cols = ["timestamp"]

            if df.empty:
                return "no data"
            df = df.set_index(df['timestamp'])
            df.index = df.index.str.replace(r'0{3}$', '', regex=True)
            df.index = pd.to_datetime(df.index.astype(int), unit='s', utc=True, errors='coerce')
            return df
        except Exception as e:
            print("Error fetching futures OHLCV data:", e)
            return None

def _get_ohlcv_bitget(symbol, timeframe, limit, version="V2"):
    if version == "V2":
        return _get_ohlcv_bitget_v2(symbol, timeframe, limit)
    else:
        return _get_ohlcv_bitget_v1(symbol, timeframe, limit)

def _get_ohlcv_bitget(symbol, timeframe, limit):
    df_api_ohlv = _get_ohlcv_bitget_v2(symbol, timeframe, limit+1).iloc[:-1]
    return df_api_ohlv

def _get_ohlcv(symbol, start, end=None, timeframe="1h", limit=100):
    return _get_ohlcv_bitget(symbol, timeframe, limit)