import pandas as pd
import convert_to_xcel
import csv
from pathlib import Path
import os
import re

# Try importing to_offset for newer versions of Pandas:
try:
    from pandas import to_offset
except ImportError:
    # For older versions of Pandas, use:
    from pandas.tseries.frequencies import to_offset


def infer_timeframe(index) -> str:
    """
    Infer and standardize a frequency string (e.g. '1m', '5m', '1h')
    from a DatetimeIndex using pandas.infer_freq and to_offset.
    """

    # Let pandas figure out the raw frequency alias (like 'T', '5T', 'H', etc.)
    raw_freq = pd.infer_freq(index)
    if raw_freq is None:
        # Pandas couldn't infer a frequency
        return None

    # Convert the inferred alias to a DateOffset, e.g. '5T' -> <5 * Minutes>
    offset = to_offset(raw_freq)

    # 'offset.nanos' is the total number of nanoseconds in each step
    ns = offset.nanos

    # Convert nanoseconds to minutes
    minutes = ns / 1e9 / 60.0

    # If it's a whole number of minutes, see how to map it
    if minutes.is_integer():
        minutes = int(minutes)

        # If less than 60, treat it as N minutes
        if minutes < 60:
            return f"{minutes}m"

        # Otherwise, see if it’s an exact number of hours
        hours = minutes // 60
        leftover = minutes % 60
        if leftover == 0:
            # e.g. 60 -> 1h, 120 -> 2h, etc.
            return f"{int(hours)}h"

        # If leftover != 0, e.g. 90 minutes -> 1.5h
        # You could handle that specially, or just return the original alias:
        return offset.freqstr

    else:
        # Not a whole number of minutes (maybe seconds or fraction).
        # Fallback or custom logic here.
        return offset.freqstr

def add_exel_before_csv(path_str):
    p = Path(path_str)            # Convert string to Path object
    if p.suffix.lower() == ".csv":
        # If the file ends with .csv, insert _exel before .csv
        return str(p.with_name(f"{p.stem}_exel{p.suffix}"))
    else:
        # If it doesn't end with .csv, return the original path (or handle differently)
        return path_str

def add_exel_before_csv(path_str):
    p = Path(path_str)            # Convert string to Path object
    if p.suffix.lower() == ".csv":
        # If the file ends with .csv, insert _exel before .csv
        return str(p.with_name(f"{p.stem}_exel{p.suffix}"))
    else:
        # If it doesn't end with .csv, return the original path (or handle differently)
        return path_str

def save_dataframe(df: pd.DataFrame, directory: str, filename: str, file_format="csv", keep_index=True):
    """
    Saves a pandas DataFrame to a specified directory while preserving the index.
    - Creates the directory if it does not exist.
    - If the file already exists, appends an incrementing number.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        directory (str): The directory where the file should be saved.
        filename (str): The base name of the file (without extension).
        file_format (str): The file format (default: "csv").
        keep_index (bool): Whether to keep the DataFrame index (default: True).

    Returns:
        str: The final saved file path.
    """
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # Construct the initial file path
    file_path = os.path.join(directory, f"{filename}.{file_format}")

    # Handle duplicate filenames by appending an incrementing number
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(directory, f"{filename}_{counter}.{file_format}")
        counter += 1

    # Save DataFrame based on the file format
    if file_format == "csv":
        df.to_csv(file_path, index=keep_index)
        new_excel_path = add_exel_before_csv(file_path)
        convert_to_xcel.convert_csv_for_excel(file_path, new_excel_path)

    elif file_format == "json":
        df.to_json(file_path, orient="records", indent=4)
    elif file_format in ["excel", "xlsx"]:
        df.to_excel(file_path, index=keep_index, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format. Choose from 'csv', 'json', or 'xlsx'.")

    return file_path

def round_time(dt, timeframe):
    """
    Round the datetime object based on the given timeframe.
    """
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        rounded_minutes = (dt.minute // minutes) * minutes
        return dt.replace(second=0, microsecond=0, minute=rounded_minutes)
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        try:
            rounded_hours = (dt.hour // hours) * hours
        except:
            print("toto")
        return dt.replace(second=0, microsecond=0, minute=0, hour=rounded_hours)
    else:
        raise ValueError("Unsupported timeframe. Use formats like '5m', '1h', etc.")

def detect_delimiter(file_path):
    """
    Detects the delimiter of a CSV file, defaulting to ',' if detection fails.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(1024)  # Read a small part of the file
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except csv.Error:
            print("Warning: Could not determine delimiter; defaulting to manual check.")
            return None

def read_csv_thread_safe(file_path, lock):
    """
    Reads a CSV file safely with automatic delimiter detection.
    Uses a lock to ensure thread-safe reading.
    """
    delimiter = detect_delimiter(file_path)
    with lock:
        if delimiter:
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            # Try `;` first (common for Excel), fallback to `,`
            try:
                df = pd.read_csv(file_path, delimiter=";")
            except pd.errors.ParserError:
                df = pd.read_csv(file_path, delimiter=",")
    return df

def get_numeric_prefixes(directory_path):
    """
    Parses all .csv files in the given directory and extracts
    any filename prefixes of the form 'digits_' (e.g. '123_').

    :param directory_path: Path to the directory containing the CSV files.
    :return: A list of matching prefixes (strings like '123_').
    """
    # Regex pattern to match filenames beginning with one or more digits
    # followed by an underscore, and ending with '.csv'
    pattern = re.compile(r'^(\d+_)')

    prefixes = []

    # List all files in the given directory
    for filename in os.listdir(directory_path):
        # We're only interested in CSV files
        if filename.lower().endswith('.png'):
            match = pattern.match(filename)
            if match:
                prefix_without_underscore = match.group(1).replace('_', '')
                prefixes.append(prefix_without_underscore)

    return prefixes


def drop_rows_by_id(df, id_list):
    """
    Drops all rows in the DataFrame df where df['ID'] is in id_list.

    Parameters:
    df (pd.DataFrame): The input DataFrame, which must have an "ID" column.
    id_list (list): List of IDs to remove from df.

    Returns:
    pd.DataFrame: A new DataFrame with rows removed where ID was in id_list.
    """
    # Keep only those rows where 'ID' is NOT in id_list
    filtered_df = df[~df["ID"].isin(id_list)]

    # Optionally, you can use .copy() if you want a completely independent DataFrame
    # filtered_df = df[~df["ID"].isin(id_list)].copy()

    return filtered_df