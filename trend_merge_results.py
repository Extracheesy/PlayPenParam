from pathlib import Path

import pandas as pd

# use a raw-string so backslashes aren’t treated as escapes
p = Path(r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\tend_analysis\trend_results")

# 2) Find only the BTCUSDT…_data_ohlcv dirs
filter_prefix = "BTCUSDT"
filter_suffix = "_data_ohlcv"
data_dirs = [
    d for d in p.iterdir()
    if d.is_dir() and d.name.startswith(filter_prefix) and d.name.endswith(filter_suffix)
]

# 3) Loop through each folder, pick up all *_results.csv files, read & horiz-concat
df_result = pd.DataFrame()
for data_dir in data_dirs:
    result_files = [
        f for f in data_dir.iterdir()
        if f.is_file() and f.name.endswith("_results.csv")
    ]
    for file_path in result_files:
        df = pd.read_csv(file_path)  # or sep="," if they really are commas
        # Optional: prefix columns to avoid name collisions
        df.columns = [f"{data_dir.name}_{col}" for col in df.columns]
        df_result = pd.concat([df_result, df], axis=1)

# 4) Write the combined DataFrame to CSV with semicolons
out_file = p / "global_trend_results.csv"
df_result.to_csv(out_file, sep=";", index=True)  # index=True keeps your row indices
