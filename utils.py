import pandas as pd
import convert_to_xcel
import os
import csv
from pathlib import Path

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