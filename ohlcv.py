import requests
import time
import concurrent.futures
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import pandas as pd

API_URL = "https://api.bitget.com/api/v2/mix/market/candles"
PRODUCT_TYPE = "USDT-FUTURES"  # Adjust if needed.


def ms_timestamp(dt: datetime) -> int:
    """Convert a timezone-aware datetime object to a Unix timestamp in milliseconds."""
    return int(dt.timestamp() * 1000)


def fetch_candles(symbol: str, timeframe: str, start_ms=None, end_ms=None, limit=None) -> list:
    """
    Fetch candlestick data from Bitget API.

    :param symbol: Trading pair symbol (e.g., "BTCUSDT")
    :param timeframe: Candle interval (e.g., "1m" or "1H")
    :param start_ms: Optional start time in milliseconds
    :param end_ms: Optional end time in milliseconds
    :param limit: Maximum number of records to retrieve (max=1000)
    :return: List of candle records.
    """
    params = {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "granularity": timeframe,
    }
    if start_ms is not None:
        params["startTime"] = str(start_ms)
    if end_ms is not None:
        params["endTime"] = str(end_ms)
    if limit is not None:
        params["limit"] = str(limit)

    response = requests.get(API_URL, params=params)
    data = response.json()
    if data.get("code") != "00000":
        raise Exception(f"Error fetching data for {symbol} with timeframe {timeframe}: {data.get('msg')}")
    return data.get("data", [])


def fetch_1m_candles_for_range(symbol: str, start_dt: datetime, end_dt: datetime) -> list:
    """
    Fetch 1-minute candles for a given time range using pagination,
    ensuring each API call returns at most 1000 records.

    :param symbol: Trading pair symbol (e.g., "BTCUSDT")
    :param start_dt: Start datetime (timezone-aware, UTC)
    :param end_dt: End datetime (timezone-aware, UTC)
    :return: List of candle records for this range.
    """
    candles = []
    start_ms = ms_timestamp(start_dt)
    end_ms = ms_timestamp(end_dt)

    while start_ms < end_ms:
        # Request up to 1000 records per call.
        data_chunk = fetch_candles(symbol, "1m", start_ms=start_ms, end_ms=end_ms, limit=1000)
        if not data_chunk:
            break
        candles.extend(data_chunk)
        # If fewer than 1000 records are returned, then we've got all available data for this segment.
        if len(data_chunk) < 1000:
            break
        # Otherwise, update start_ms to one millisecond after the last candle's timestamp.
        last_ts = int(data_chunk[-1][0])
        new_start_ms = last_ts + 1
        if new_start_ms == start_ms:
            break
        start_ms = new_start_ms
        time.sleep(0.2)  # Respect the rate limits.
    return candles


def fetch_1m_candles_for_one_month(symbol: str) -> list:
    """
    Fetch 1-minute candles for the last one month for the given symbol.
    The one-month period is split into chunks of 1000 minutes each and fetched concurrently.

    :param symbol: Trading pair symbol (e.g., "BTCUSDT")
    :return: List of 1m candle records.
    """
    now = datetime.now(timezone.utc)
    start_period = now - relativedelta(months=1)

    # Generate 1000-minute chunks within the one-month period.
    chunks = []
    current = start_period
    chunk_duration = timedelta(minutes=1000)
    while current < now:
        chunk_end = min(current + chunk_duration, now)
        chunks.append((current, chunk_end))
        current = chunk_end

    all_candles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit a thread for each chunk.
        future_to_chunk = {executor.submit(fetch_1m_candles_for_range, symbol, cs, ce): (cs, ce)
                           for cs, ce in chunks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            cs, ce = future_to_chunk[future]
            try:
                chunk_data = future.result()
                print(f"Chunk from {cs} to {ce} fetched with {len(chunk_data)} records.")
                all_candles.extend(chunk_data)
            except Exception as exc:
                print(f"Chunk from {cs} to {ce} generated an exception: {exc}")
    return all_candles


def fetch_1h_candles_latest_1000(symbol: str) -> list:
    """
    Fetch the latest 1000 1-hour candles for the given symbol.

    :param symbol: Trading pair symbol (e.g., "BTCUSDT")
    :return: List of 1H candle records.
    """
    now_ms = ms_timestamp(datetime.now(timezone.utc))
    print(f"Fetching latest 1000 1H candles for {symbol}...")
    data = fetch_candles(symbol, "1H", end_ms=now_ms, limit=1000)
    return data


def get_candle_dataframe(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given symbol and timeframe and return it as a Pandas DataFrame.

    - For "1m" timeframe: Fetch the last one month using 1000-minute chunks (with multithreading).
    - For "1H" timeframe: Fetch the latest 1000 candles.

    The resulting DataFrame has columns: datetime, open, high, low, close, volume.

    :param symbol: Trading pair symbol (e.g., "BTCUSDT")
    :param timeframe: Candle interval ("1m" or "1H")
    :return: DataFrame with OHLCV data.
    """
    candles = []

    if timeframe == "1m":
        candles = fetch_1m_candles_for_one_month(symbol)
    elif timeframe == "1H":
        candles = fetch_1h_candles_latest_1000(symbol)
    else:
        raise ValueError("Unsupported timeframe. Please choose either '1m' or '1H'.")

    if not candles:
        raise Exception(f"No candle data fetched for {symbol} with timeframe {timeframe}.")

    # The API returns data as:
    # [timestamp, open, high, low, close, base_volume, quote_volume]
    columns = ["timestamp", "open", "high", "low", "close", "base_volume", "quote_volume"]
    df = pd.DataFrame(candles, columns=columns)

    # Convert the Unix timestamp in milliseconds to a datetime column.
    df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms')

    # Convert numerical columns.
    for col in ["open", "high", "low", "close", "base_volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Use base_volume as our volume.
    df = df[["datetime", "open", "high", "low", "close", "base_volume"]]
    df.rename(columns={"base_volume": "volume"}, inplace=True)

    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df.set_index('datetime')
    return df


# Example usage:
if __name__ == "__main__":
    symbol = "BTCUSDT"  # Change symbol if needed.
    timeframe = "1m"  # Can be either "1m" or "1H"
    try:
        df_candles = get_candle_dataframe(symbol, timeframe)
        print(df_candles.head())
        # Optionally, save to a CSV file:
        # df_candles.to_csv(f"{symbol}_{timeframe}_candles.csv", index=False)
    except Exception as e:
        print("An error occurred:", e)
